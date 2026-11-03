"""Stage 2 tests: the conditioning op registry, parsing and the no-op rule.

No GPU, no engine, no video: these are pure image operations, so every op can
be checked on a small synthetic frame. The three properties that matter to the
benchmark are tested for *every* registered op rather than a sample, so adding
an op to ``yolocar.filters`` cannot silently break the chain contract:

* it preserves the wire type (BGR uint8 HxWx3, contiguous, source geometry);
* it does not mutate its input (the ROS transport hands out read-only views);
* it is deterministic (two runs on the same frame agree).
"""

from __future__ import annotations

import inspect

import cv2
import numpy as np
import pytest

pytest.importorskip("cv2")

from yolocar import filters  # noqa: E402
from yolocar.preprocessing import (  # noqa: E402
    OPS,
    Op,
    as_bgr_frame,
    condition,
    describe,
    enhance,
    parse,
)

OP_NAMES = sorted(OPS)


def _frame(height: int = 48, width: int = 64, seed: int = 0) -> np.ndarray:
    """A frame with structure and noise, so filters have something to act on."""
    rng = np.random.default_rng(seed)
    gradient = np.linspace(0, 255, width, dtype=np.float32)[None, :]
    noise = rng.normal(0.0, 12.0, (height, width))
    grey = np.clip(gradient + noise, 0, 255).astype(np.uint8)[:, :, None]
    return np.ascontiguousarray(np.repeat(grey, 3, axis=2))


def _read_only(frame: np.ndarray) -> np.ndarray:
    """What the ROS transport actually hands a node: a read-only buffer view."""
    view = np.frombuffer(frame.tobytes(), dtype=np.uint8)
    result = view.reshape(frame.shape)
    assert not result.flags.writeable
    return result


# ------------------------------------------------------------------- registry
def test_registry_covers_every_filter_function() -> None:
    """``OPS`` is the only way a chain names an op; keep it complete."""
    public = {
        name
        for name, value in vars(filters).items()
        if inspect.isfunction(value)
        and value.__module__ == filters.__name__
        and not name.startswith("_")
    }
    assert public == set(OPS), (
        "yolocar.filters exposes functions that the registry does not: "
        f"{sorted(public - set(OPS))}; registry has stale entries: "
        f"{sorted(set(OPS) - public)}"
    )


@pytest.mark.parametrize("name", OP_NAMES)
def test_op_preserves_the_wire_type_and_does_not_mutate(name: str) -> None:
    source = _read_only(_frame())
    before = source.copy()

    result = OPS[name](source)

    assert result.shape == source.shape
    assert result.dtype == np.uint8
    assert result.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(source, before, err_msg=f"{name} mutated its input")


@pytest.mark.parametrize("name", OP_NAMES)
def test_op_is_deterministic(name: str) -> None:
    frame = _frame(seed=7)
    np.testing.assert_array_equal(
        OPS[name](frame), OPS[name](frame), err_msg=f"{name} is not deterministic"
    )


def test_expensive_ops_actually_change_the_image() -> None:
    """A chain that cannot change a frame is not a workload worth measuring."""
    frame = _frame()
    for name in OP_NAMES:
        if name == "gamma":  # gamma=1.0 is the identity by definition
            continue
        assert not np.array_equal(OPS[name](frame), frame), name


# ------------------------------------------------------------------ behaviour
def test_gamma_applies_the_expected_tone_curve() -> None:
    frame = np.full((4, 4, 3), 100, dtype=np.uint8)
    out = filters.gamma(frame, gamma=2.0)
    expected = round(255.0 * (100.0 / 255.0) ** 2.0)
    np.testing.assert_array_equal(out, np.full((4, 4, 3), expected, dtype=np.uint8))


def test_gamma_of_one_is_the_identity() -> None:
    frame = _frame()
    np.testing.assert_array_equal(filters.gamma(frame, gamma=1.0), frame)


def test_bilateral_reduces_noise_variance_on_a_flat_patch() -> None:
    rng = np.random.default_rng(3)
    base = np.full((64, 64, 3), 128, dtype=np.float32)
    noisy = np.clip(base + rng.normal(0, 10, base.shape), 0, 255).astype(np.uint8)
    smoothed = filters.bilateral(noisy, d=7, sigma_color=40, sigma_space=7)
    assert smoothed.std() < noisy.std()


def test_median_removes_salt() -> None:
    frame = np.full((16, 16, 3), 100, dtype=np.uint8)
    frame[8, 8] = 255
    assert int(filters.median(frame, ksize=3)[8, 8, 0]) < 255


@pytest.mark.parametrize(
    ("op", "kwargs", "message"),
    [
        (filters.gamma, {"gamma": 0.0}, "gamma must be"),
        (filters.gaussian, {"ksize": 4}, "odd"),
        (filters.median, {"ksize": 2}, "3 or 5"),
        (filters.median, {"ksize": 7}, "3 or 5"),
        (filters.bilateral, {"d": 0}, "d must be"),
        (filters.clahe, {"tiles": 0}, "tiles must be"),
        (filters.clahe, {"clip_limit": 0.0}, "clip_limit must be"),
    ],
)
def test_ops_reject_impossible_parameters(op, kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        op(_frame(), **kwargs)


# --------------------------------------------------------------------- parsing
def test_parse_empty_is_an_empty_chain() -> None:
    for spec in (None, "", (), []):
        assert parse(spec) == ()


def test_parse_bare_names_and_parameters() -> None:
    chain = parse(("clahe", "bilateral:d=9,sigma_color=50", "gamma:gamma=0.8"))
    assert [op.name for op in chain] == ["clahe", "bilateral", "gamma"]
    assert chain[1].params == {"d": 9.0, "sigma_color": 50.0}
    assert chain[2].params == {"gamma": 0.8}
    assert str(chain[1]) == "bilateral:d=9,sigma_color=50"


def test_parse_accepts_a_single_string() -> None:
    assert [op.name for op in parse("clahe")] == ["clahe"]


def test_parse_splits_a_semicolon_separated_string() -> None:
    """The form a ROS parameter or shell argument can carry."""
    chain = parse("clahe;bilateral:d=9,sigma_color=50;gamma:gamma=0.8")
    assert [op.name for op in chain] == ["clahe", "bilateral", "gamma"]
    assert chain[1].params == {"d": 9.0, "sigma_color": 50.0}
    assert parse("clahe;") == parse("clahe")


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("sharpen", "unknown preprocessing op 'sharpen'"),
        ("clahe:k=3", "does not take k"),
        ("clahe:clip_limit", "malformed parameter"),
        (("clahe", "clahe"), "appears twice"),
    ],
)
def test_parse_rejects_malformed_chains(spec, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse(spec)


def test_error_message_lists_the_available_ops() -> None:
    with pytest.raises(ValueError) as excinfo:
        parse("sharpen")
    for name in OP_NAMES:
        assert name in str(excinfo.value)


# ------------------------------------------------------------ chain execution
def test_empty_chain_is_a_bit_exact_zero_copy_noop() -> None:
    frame = _frame()
    out, timings = condition(frame)
    assert out is frame
    assert timings == {}


def test_condition_coerces_then_applies_the_chain() -> None:
    """Stage 2 owns the wire type: a float frame comes out as BGR uint8."""
    float_frame = _frame().astype(np.float32) * 4.0  # beyond 255 on purpose
    out, timings = condition(float_frame, parse("gamma:gamma=2.0"))
    assert out.dtype == np.uint8
    assert out.shape == float_frame.shape
    assert set(timings) == {"gamma"}


def test_enhance_reports_one_timing_per_op_in_chain_order() -> None:
    chain = parse(("gamma:gamma=1.2", "clahe", "gaussian"))
    out, timings = enhance(_frame(), chain)
    assert list(timings) == [op.name for op in chain]
    assert all(value >= 0.0 for value in timings.values())
    assert out.shape == (48, 64, 3)


def test_enhance_does_not_coerce() -> None:
    """``enhance`` is the pure op loop; ``condition`` is the stage boundary."""
    with pytest.raises(cv2.error):
        enhance(_frame().astype(np.float64), parse("gamma"))


def test_chained_ops_do_not_mutate_the_intermediate_or_source() -> None:
    frame = _frame()
    before = frame.copy()
    enhance(frame, parse(("clahe", "bilateral:d=5")))
    np.testing.assert_array_equal(frame, before)


def test_describe_lists_the_chain_for_provenance() -> None:
    assert describe(parse(("gamma:gamma=1.5", "clahe"))) == [
        "gamma:gamma=1.5",
        "clahe",
    ]


def test_op_apply_matches_the_registry_function() -> None:
    frame = _frame()
    op = Op(name="gaussian", params={"ksize": 3.0})
    np.testing.assert_array_equal(op.apply(frame), filters.gaussian(frame, ksize=3))


def test_as_bgr_frame_still_normalises() -> None:
    assert as_bgr_frame(np.zeros((8, 8), dtype=np.uint8)).shape == (8, 8, 3)
    assert as_bgr_frame(np.zeros((8, 8, 3), dtype=np.float32)).dtype == np.uint8

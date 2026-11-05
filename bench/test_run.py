"""The scenario matrix is a contract; these tests are what keeps it one.

The harness is out-of-process by design, so nothing here launches a workload:
every test works on the planner's output -- cell name, condition, command --
which is the last place a mistake is still cheap. Run with ``pytest bench``.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run as R  # noqa: E402

SCENARIO = json.loads((Path(__file__).resolve().parent / "scenarios.json").read_text())


def cells() -> list[tuple[str, dict[str, object]]]:
    return R.matrix_conditions(SCENARIO)


def by_name() -> dict[str, dict[str, object]]:
    return dict(cells())


# ------------------------------------------------------- one common workload
def test_every_cell_runs_the_one_conditioning_chain() -> None:
    """The seven ops are a constant of the experiment, not a coordinate.

    ``check_cell_conditioning`` is what ``matrix_conditions`` calls, so this
    passing means the monolith cells, the four-node chain, every shape and every
    window carry the same stage 2 -- in the form each front-end takes.
    """
    ops = R.conditioning_ops(SCENARIO)
    assert len(ops) == 7
    for name, condition in cells():
        R.check_cell_conditioning(name, condition, ops)


def test_a_cell_that_skips_an_op_is_rejected() -> None:
    ops = R.conditioning_ops(SCENARIO)
    conditions = by_name()

    mono = conditions["monolith-720p-n"]
    dropped = dict(mono, cmd=list(mono["cmd"]))  # type: ignore[arg-type]
    index = max(i for i, part in enumerate(dropped["cmd"]) if part == "--pre")
    del dropped["cmd"][index : index + 2]
    with pytest.raises(SystemExit, match="conditioning chain"):
        R.check_cell_conditioning("monolith-720p-n", dropped, ops)

    ros = conditions["ros-N4-p2c1-720p-n"]
    # The ROS templates splice `{pre_ops}` from the scenario, so to make a cell
    # disagree we hardcode a chain into its command -- which is also what a
    # hand-edited scenario looks like.
    short = dict(
        ros,
        shell=ros["shell"].replace(  # type: ignore[union-attr]
            "'pre_ops:={pre_ops}'", "'pre_ops:=clahe;bilateral'"
        ),
    )
    with pytest.raises(SystemExit, match="conditioning chain"):
        R.check_cell_conditioning("ros-N4-p2c1-720p-n", short, ops)


def test_a_command_with_no_conditioning_is_rejected() -> None:
    ops = R.conditioning_ops(SCENARIO)
    conditions = by_name()

    mono = conditions["monolith-720p-n"]
    bare = dict(
        mono,
        cmd=[part for part in mono["cmd"] if part != "--pre" and part not in ops],  # type: ignore[operator]
    )
    with pytest.raises(SystemExit, match="conditioning chain"):
        R.check_cell_conditioning("monolith-720p-n", bare, ops)

    ros = conditions["ros-N4-p2c1-720p-n"]
    bare_ros = dict(ros, shell=ros["shell"].replace("'pre_ops:={pre_ops}'", ""))  # type: ignore[union-attr]
    with pytest.raises(SystemExit, match="pre_ops"):
        R.check_cell_conditioning("ros-N4-p2c1-720p-n", bare_ros, ops)


def test_a_scenario_without_a_chain_is_rejected() -> None:
    with pytest.raises(SystemExit, match="conditioning.ops"):
        R.check_cell_conditioning("monolith-720p-n", {"cmd": ["true"]}, [])


# ------------------------------------------------------------- cell naming
def test_the_legacy_four_node_name_is_gone() -> None:
    names = [name for name, _ in cells()]
    assert not any("4node" in name for name in names)
    assert "ros-N4-p2c1-720p-n" in names


def test_the_design_has_no_duplicates_and_one_p2c1_baseline() -> None:
    """`ros-N4-p2c1` is the matrix's ROS cell *and* the shape axis's baseline.

    The two workloads are now one cell, so it must appear exactly once per
    (resolution, model) -- listing it twice is what the old design did when the
    op-free copy was a separate name.
    """
    names = [name for name, _ in cells()]
    assert len(names) == len(set(names))
    assert names.count("ros-N4-p2c1-720p-n") == 1
    assert len(names) == 40


def test_the_window_token_is_checked_for_names_without_a_shape() -> None:
    """Regression: the checker used to return early for non-``N{p}c{c}`` names.

    The frames-in-flight curve is indexed by the ``-iD`` token, so a cell whose
    name says one window and whose command runs another is misfiled silently.
    """
    with pytest.raises(SystemExit, match="name says i4"):
        R.check_cell_shape("monolith-720p-n-i4", {"inflight": 1, "cmd": ["true"]})


def test_a_windowed_ros_cell_must_pass_its_window() -> None:
    """The name, the declared window and the launch argument must all agree.

    The launch defaults to one frame in flight, so a ``-i4`` cell whose command
    forgot ``inflight:=`` would run the demand-driven chain and be filed under a
    window it never had.
    """
    condition = dict(by_name()["ros-N4-p2c1-720p-n"])  # type: ignore[arg-type]
    condition["inflight"] = 4
    condition["shell"] = condition["shell"].replace(  # type: ignore[union-attr]
        " inflight:={inflight}", ""
    )
    with pytest.raises(SystemExit, match="must pass inflight:=4"):
        R.check_cell_shape("ros-N4-p2c1-720p-n-i4", condition)


def test_every_ros_cell_must_declare_its_shape() -> None:
    condition = dict(by_name()["ros-N4-p2c1-720p-n"])
    condition["shell"] = condition["shell"].replace(  # type: ignore[union-attr]
        "'chain:=acq|pre|infer|post' ", ""
    )
    with pytest.raises(SystemExit, match="must pass chain"):
        R.check_cell_shape("ros-N4-p2c1-720p-n", condition)


# ---------------------------------------------------------------- budgeting
def test_the_hint_table_has_a_resolution_level() -> None:
    """The ops and the model scale differently with pixels.

    One scalar per chain cannot describe both ends: at 720p the chain is ~7x the
    inference of the n engine, at 180p the two are comparable, so the model
    spread widens as resolution falls. The table therefore carries a value per
    (chain, resolution, model), and these are the shipped 720p/180p entries.
    """
    conditions = by_name()
    assert R._fps_hint(conditions["monolith-180p-n"], SCENARIO) == 25.0
    assert R._fps_hint(conditions["monolith-180p-x"], SCENARIO) == 12.0
    assert R._fps_hint(conditions["ros-N4-p2c1-180p-n"], SCENARIO) == 13.0
    # Window cells are per-cell overrides (main() merges `cells` on top), so the
    # override must win over the table.
    ladder = dict(conditions["ros-N4-p2c1-720p-n-i4"])
    ladder.update(SCENARIO["cells"]["ros-N4-p2c1-720p-n-i4"])
    assert R._fps_hint(ladder, SCENARIO) == 6.0


# ------------------------------------------------------------- the full design
def full_cells() -> list[tuple[str, dict[str, object]]]:
    return R.matrix_conditions(SCENARIO, design="full")


def test_the_full_design_is_every_combination() -> None:
    """`--design full` is the cross product of the factors, windows included.

    Monolith: 3 resolutions x 3 models at its one window = 9. Every ROS shape:
    3 resolutions x 3 models x 4 windows = 36, times 8 shapes. The point of
    stating the arithmetic in a test is that the number is the *design*: 297
    cells, 891 trials at three passes.
    """
    names = [name for name, _ in full_cells()]
    assert len(names) == len(set(names)) == 297

    chains = Counter(condition["cell"]["chain"] for _, condition in full_cells())  # type: ignore[index]
    assert chains["monolith"] == 9
    assert sum(count for chain, count in chains.items() if chain != "monolith") == 288
    assert len(chains) == 9
    for chain, count in chains.items():
        if chain != "monolith":
            assert count == 3 * 3 * 4, chain

    resolutions = Counter(condition["cell"]["resolution"] for _, condition in full_cells())  # type: ignore[index]
    models = Counter(condition["cell"]["model"] for _, condition in full_cells())  # type: ignore[index]
    windows = Counter(condition["inflight"] for _, condition in full_cells())
    assert resolutions == Counter({"720p": 99, "360p": 99, "180p": 99})
    assert models == Counter({"n": 99, "m": 99, "x": 99})
    # 9 monolith cells + 8 shapes x 3 x 3 at the window of one.
    assert windows == Counter({1: 81, 2: 72, 4: 72, 8: 72})


def test_the_full_design_contains_the_pruned_one() -> None:
    """The default design is a subset of the full matrix, workload and all.

    Same cell names, same command templates, same conditioning: `--design full`
    changes which cells exist, never how any of them is measured.
    """
    pruned = by_name()
    full = dict(full_cells())
    missing = sorted(set(pruned) - set(full))
    assert not missing, missing
    for name in pruned:
        assert pruned[name].get("cmd") == full[name].get("cmd")
        assert pruned[name].get("shell") == full[name].get("shell")


def test_the_monolith_has_no_window() -> None:
    """One process, one frame at a time: its window domain is [1]."""
    assert R.chain_windows(SCENARIO["factors"]["chain"]["monolith"]) == [1]
    assert R.chain_windows(SCENARIO["factors"]["chain"]["ros-N4-p2c1"]) == [1, 2, 4, 8]


def test_a_window_outside_the_chain_domain_is_rejected() -> None:
    """The declared domain and the design list cannot drift apart.

    A cell that asks for a window the chain does not declare would either launch
    a different graph or invite hand-editing `windows` until the name is a lie.
    """
    scenario = json.loads(json.dumps(SCENARIO))
    scenario["design"]["cells"].append(["monolith", "720p", "n", 4])
    with pytest.raises(SystemExit, match="declares windows"):
        R.matrix_conditions(scenario)


# ------------------------------------------------------------ derived hints
def test_derived_hints_are_pessimistic_and_windowed() -> None:
    """A cell with no measured hint is costed, not given a flat default.

    The derived rate reproduces the shipped window-of-one entries (that is the
    sanity check in `fps_model`), and a windowed cell of a family the table does
    cover may not exceed the table's rate by more than the declared gain cap --
    an over-estimate is the failure that costs data, because it shortens the
    ceiling until the analysis window comes up short.
    """
    caps = float(SCENARIO["fps_model"]["window_gain_cap"])
    conditions = by_name()
    assert 4.0 <= R._fps_from_model(conditions["monolith-720p-n"], SCENARIO) <= 6.5
    assert 3.5 <= R._fps_from_model(conditions["ros-N4-p2c1-720p-n"], SCENARIO) <= 5.5

    for name, condition in full_cells():
        hint = R._fps_hint(condition, SCENARIO)
        assert 0.5 <= hint < R.DEFAULT_FPS_HINT, (name, hint)
        inflight = int(condition["inflight"])
        if inflight == 1:
            continue
        same_family = dict(condition, inflight=1)
        one = R._fps_hint(same_family, SCENARIO)
        assert one <= hint <= one * caps * 1.01 + 1e-9, (name, one, hint)


def test_the_frame_cap_never_binds_before_the_ceiling() -> None:
    """Every cell's `max_frames` must cover its own settle at its own hint.

    The cap exists to catch an fps *overshoot* and must never truncate a trial
    that is simply faster than its hint, or the window would come up short and
    the cell would be silently measured over fewer frames.
    """
    for name, condition in full_cells():
        budget = R.trial_budget(
            condition, SCENARIO, settle=None, analysis_frames=None, duration=None
        )
        needed = float(budget["settle_s"]) * float(budget["fps_hint"])
        assert float(budget["max_frames"]) > needed, name
        assert float(budget["duration_s"]) > float(budget["settle_s"]), name


# --------------------------------------------------------------- the window
def test_a_windowed_cell_settles_longer_than_a_demand_driven_one() -> None:
    """The settle rule follows the cell, not a hand-copied override.

    A windowed cell is a different power/thermal state, so `run.py` gives it
    `trial.settle_s_windowed`; `analyze.py` reads that back per trial. A CLI
    override still wins over both.
    """
    conditions = by_name()
    assert R.trial_budget(
        conditions["ros-N4-p2c1-720p-n"], SCENARIO, settle=None,
        analysis_frames=None, duration=None,
    )["settle_s"] == 30.0
    assert R.trial_budget(
        dict(conditions["ros-N4-p2c1-720p-n"], inflight=4), SCENARIO, settle=None,
        analysis_frames=None, duration=None,
    )["settle_s"] == 45.0
    assert R.trial_budget(
        dict(conditions["ros-N4-p2c1-720p-n"], inflight=4), SCENARIO, settle=12.0,
        analysis_frames=500, duration=None,
    )["settle_s"] == 12.0

"""
The credit window, tested as the pure state machine it is.

Every frames-in-flight question that is not about the middleware is a question
about this object: how many frames may be resident, which acks release which
frames, and what happens to an ack that arrives out of order or twice. Getting
these wrong does not fail loudly at runtime -- it either stalls the graph or
silently changes the window -- so they are pinned here, with no ROS context.
"""

from __future__ import annotations

import pytest

from yolocar_ros.window import CreditWindow


def test_a_window_of_one_admits_exactly_one_frame() -> None:
    window = CreditWindow(1)
    assert window.has_room
    window.add(0)
    assert not window.has_room
    assert len(window) == 1
    window.ack(0)
    assert window.has_room
    assert len(window) == 0


def test_a_window_of_k_admits_k_frames_in_publish_order() -> None:
    window = CreditWindow(4)
    for seq in range(4):
        assert window.has_room
        assert window.add(seq) == seq + 1
    assert not window.has_room
    assert window.outstanding == (0, 1, 2, 3)


def test_publishing_past_the_window_is_a_programming_error() -> None:
    window = CreditWindow(2)
    window.add(0)
    window.add(1)
    with pytest.raises(RuntimeError):
        window.add(2)


def test_a_size_below_one_is_clamped_to_one() -> None:
    assert CreditWindow(0).size == 1
    assert CreditWindow(-3).size == 1


def test_an_ack_releases_every_frame_up_to_it() -> None:
    window = CreditWindow(5)
    for seq in range(5):
        window.add(seq)
    assert window.ack(2) == 3  # 0, 1 and 2 are done
    assert window.outstanding == (3, 4)
    assert window.acked_seq == 2


def test_acks_out_of_order_release_the_same_frames() -> None:
    in_order = CreditWindow(4)
    shuffled = CreditWindow(4)
    for seq in range(4):
        in_order.add(seq)
        shuffled.add(seq)
    in_order.ack(0)
    in_order.ack(1)
    in_order.ack(2)
    in_order.ack(3)
    # 2 arrives first, then the older ones: nothing releases twice.
    assert shuffled.ack(2) == 3
    assert shuffled.ack(0) == 0
    assert shuffled.ack(1) == 0
    assert shuffled.ack(3) == 1
    assert len(shuffled) == len(in_order) == 0


def test_a_duplicate_ack_is_harmless() -> None:
    window = CreditWindow(3)
    window.add(0)
    window.add(1)
    assert window.ack(1) == 2
    assert window.ack(1) == 0
    assert window.ack(0) == 0
    assert window.acked_seq == 1


def test_an_ack_for_an_unpublished_frame_is_ignored() -> None:
    """A stray/late ack must not release a frame that has not been sent."""
    window = CreditWindow(2)
    window.add(0)
    assert window.ack(5) == 1  # the frame we did publish is released
    assert window.acked_seq == 5
    window.add(1)  # 1 > acked_seq would be wrong; it must still be tracked
    assert window.outstanding == (1,)


def test_a_lost_frame_is_released_by_a_later_ack() -> None:
    """
    The window cannot see a drop, so contiguity has to be checked on the records.

    Frame 1 never arrives, frame 2 completes: the watermark releases both, the
    graph keeps flowing, and the loss survives as a hole in the seq set. That is
    exactly why ``depth`` is derived from the window size and why the analysis
    counts seq gaps instead of trusting the window to have prevented them.
    """
    window = CreditWindow(4)
    for seq in range(4):
        window.add(seq)
    assert window.ack(2) == 3
    assert window.outstanding == (3,)
    assert window.has_room


def test_a_lost_tail_holds_one_credit_forever() -> None:
    """The last frame of a run is the only one whose loss stalls anything."""
    window = CreditWindow(2)
    window.add(0)
    window.add(1)
    window.ack(0)
    assert window.outstanding == (1,)  # frame 1 never completes
    assert window.has_room  # one slot keeps rotating; one is gone for good

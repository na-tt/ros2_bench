"""
QoS profiles for the pipeline topics.

Every data topic carries the *window* of frames in flight: the acquisition node
publishes while its outstanding frames are below ``inflight``, so at most
``inflight`` samples can be resident on a topic at once. ``depth`` therefore has
to be at least that number -- and exactly that number, because a larger history
can only hide a mistake. A ``KEEP_LAST`` writer is allowed to evict an unacked
sample even under ``RELIABLE`` (reliability orders what is still in history, it
does not retransmit what the writer discarded), so ``depth < inflight`` is a
silent drop that surfaces only as an ack that never arrives. One number,
``inflight``, is plumbed from the launch file and derives both the pacing and
the depth, so the two cannot disagree.

``TRANSIENT_LOCAL`` is added so the first frame is not lost when a downstream
node has not finished discovery yet: a late-joining subscriber receives the
last sample instead of nothing. This stays correct at any window: the writer's
history is exactly the in-flight set, and a consumer's first contact is with
samples it has never seen. (The acquisition node publishes frame 0
synchronously and waits for its ack before opening the window, so by the time
the graph carries more than one frame every node is already matched.)
"""

from __future__ import annotations

from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


def latest(depth: int = 1) -> QoSProfile:
    """
    One topic of the window: reliable, keep-last-``depth``, late-joiner friendly.

    ``depth`` must equal the frames-in-flight window: it is the number of
    samples that may legitimately be resident on the topic, no more and no less.
    """
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=max(1, int(depth)),
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def control(depth: int = 10) -> QoSProfile:
    """Small control messages (acks, stop/done); keeping a few is harmless."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=max(1, int(depth)),
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )

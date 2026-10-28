"""
QoS profiles for the pipeline topics.

Every data topic carries exactly one frame at a time (the terminal node acks
before the acquisition node reads the next frame), so a depth-1 reliable
history is the honest profile: it never hides queueing and never drops a frame
silently.

``TRANSIENT_LOCAL`` is added so the first frame is not lost when a downstream
node has not finished discovery yet: a late-joining subscriber receives the
last sample instead of nothing. With one frame in flight this cannot deliver a
duplicate that is not already the frame being waited on.
"""

from __future__ import annotations

from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


def latest() -> QoSProfile:
    """One in-flight frame: reliable, keep-last-1, late-joiner friendly."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def control(depth: int = 10) -> QoSProfile:
    """Small control messages (acks, stop/done); keeping a few is harmless."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )

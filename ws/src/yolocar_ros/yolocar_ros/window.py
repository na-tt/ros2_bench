"""
The credit window that paces the acquisition node.

A closed-loop pipeline of ``size`` tokens: the acquisition node publishes at most
``size`` frames that the terminal node has not acknowledged yet, and every ack
releases exactly one. ``size == 1`` is the demand-driven chain (one frame
resident, the latency of a single frame); ``size == k`` keeps k frames resident,
which buys throughput by keeping downstream nodes busy and pays for it in
queued latency.

Accounting is identified by ``seq``, never counted. One monotonic watermark
releases every frame at or below it, so an ack that arrives out of order, or a
duplicate of one already seen, changes nothing -- only the latest ack has to
arrive. That also means the window cannot detect a lost frame: a later ack
releases the missing one, the graph keeps flowing, and the only trace is a hole
in the seq set. Contiguity is therefore a property of the *records*, checked
there, not of this object -- which is why the depth of every topic is derived
from the window size instead of enforced by it.

This module holds no ROS: it is the whole state machine, which is why it can be
tested without a middleware context.
"""

from __future__ import annotations

from collections import deque


class CreditWindow:
    """Track at most ``size`` frames that are published but not yet acknowledged."""

    def __init__(self, size: int = 1) -> None:
        self.size = max(1, int(size))
        self._outstanding: deque[int] = deque()
        self._acked_seq = -1

    def __len__(self) -> int:
        """Return how many frames are currently outstanding."""
        return len(self._outstanding)

    def __repr__(self) -> str:
        return (
            f'CreditWindow(size={self.size}, outstanding={len(self._outstanding)}, '
            f'acked_seq={self._acked_seq})'
        )

    @property
    def acked_seq(self) -> int:
        """Return the highest acknowledged seq (``-1`` until the first ack)."""
        return self._acked_seq

    @property
    def has_room(self) -> bool:
        """Return whether one more frame may be published without waiting."""
        return len(self._outstanding) < self.size

    @property
    def outstanding(self) -> tuple[int, ...]:
        """Return the seqs in flight, in publish order (for logging and tests)."""
        return tuple(self._outstanding)

    def add(self, seq: int) -> int:
        """Record a published frame; return the new occupancy."""
        if not self.has_room:
            raise RuntimeError(
                f'window is full ({len(self._outstanding)}/{self.size}); '
                'the publisher must wait for a release first'
            )
        self._outstanding.append(int(seq))
        return len(self._outstanding)

    def ack(self, seq: int) -> int:
        """Release every frame up to ``seq``; return how many were released."""
        if int(seq) > self._acked_seq:
            self._acked_seq = int(seq)
        released = 0
        while self._outstanding and self._outstanding[0] <= self._acked_seq:
            self._outstanding.popleft()
            released += 1
        return released

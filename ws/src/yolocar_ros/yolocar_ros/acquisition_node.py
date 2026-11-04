"""
Stage 1 node: drive a source and publish into a credit window of ``inflight``.

The node owns the pacing. It publishes while fewer than ``inflight`` of its
frames are outstanding (published and not yet acknowledged by the terminal
node), so the graph holds at most that many frames and never drops one:
``inflight:=1`` is the demand-driven chain that measures latency in isolation,
``inflight:=k`` is k frames resident, which spends latency to buy throughput.

Frame 0 is published synchronously -- the node waits for its ack before opening
the window -- because that ack is also the discovery gate (every downstream
node has matched) and the engine-ready gate (it carries ``engine_loaded_t``,
which starts the ``--duration`` clock). After that the window is filled
greedily: the node never waits for an ack it does not need.

The window is identified by ``seq``, not counted. One monotonic watermark
releases every frame at or below it, so a duplicated or reordered ack is
harmless; only the latest one has to arrive. A frame lost in the graph
therefore never releases its credit, which surfaces as ``ack_timeout`` -- the
honest failure, because a lost frame invalidates the measurement rather than
quietly shrinking the sample.

No image processing happens here beyond what :mod:`yolocar.acquisition` does
(decode + the ``--rotate`` correction), which keeps ``acq_ms`` comparable to
the monolith's acquisition bucket.
"""

from __future__ import annotations

import logging
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Empty

from yolocar.acquisition import open_source

from yolocar_interfaces.msg import FrameAck, ImageFrame

from . import qos, topics
from .conversions import frame_meta, image_frame
from .window import CreditWindow

LOGGER = logging.getLogger('yolocar')

_SENTINEL = object()


class AcquisitionNode(Node):
    """Publish the configured source as ``ImageFrame`` messages, within a window."""

    def __init__(self) -> None:
        super().__init__('yolocar_acquisition')

        self.source_path = str(
            self.declare_parameter(
                'source', 'assets/video/0000f77c-6257be58.mov'
            ).value
        )
        self.rotate = int(self.declare_parameter('rotate', 180).value)
        self.loops = int(self.declare_parameter('loops', 1).value)
        duration = float(self.declare_parameter('duration', -1.0).value)
        self.duration_s = duration if duration > 0.0 else None
        max_frames = int(self.declare_parameter('max_frames', -1).value)
        self.max_frames = max_frames if max_frames > 0 else None
        #: Frames in flight. 1 = demand-driven (one frame resident, no window).
        self.inflight = max(1, int(self.declare_parameter('inflight', 1).value))
        self.ack_timeout = float(self.declare_parameter('ack_timeout', 60.0).value)
        self.end_timeout = float(self.declare_parameter('end_timeout', 30.0).value)
        log_level = str(self.declare_parameter('log_level', 'INFO').value)
        LOGGER.setLevel(log_level.upper())

        # The topic depth *is* the window: a smaller history would evict an
        # unacked sample, a larger one could only hide a mistake.
        self.publisher = self.create_publisher(
            ImageFrame, topics.FRAME_TOPIC, qos.latest(self.inflight)
        )
        self.stop_publisher = self.create_publisher(
            Empty, topics.STOP_TOPIC, qos.control(depth=1)
        )
        self.create_subscription(
            FrameAck,
            topics.ACK_TOPIC,
            self._on_ack,
            qos.control(max(10, self.inflight)),
        )
        self.create_subscription(
            Empty, topics.DONE_TOPIC, self._on_done, qos.control(depth=1)
        )

        #: The whole flow-control state: which seqs are in flight and whether
        #: one more may be published. Nothing else paces the node.
        self.window = CreditWindow(self.inflight)
        self._published = 0
        self._done = False
        self._t_measure_start: float | None = None
        self._t_start = time.monotonic()

    # ------------------------------------------------------------- callbacks
    def _on_ack(self, msg: FrameAck) -> None:
        # One watermark releases every frame up to it: acks that arrive out of
        # order, or not at all, cost nothing as long as the latest one lands.
        self.window.ack(int(msg.seq))
        # The inference node reports when its engine became ready; use it as
        # the measurement clock origin (the monolith starts at engine load).
        if self._t_measure_start is None and msg.engine_loaded_t > 0.0:
            self._t_measure_start = float(msg.engine_loaded_t)

    def _on_done(self, _msg: Empty) -> None:
        self._done = True

    # ------------------------------------------------------------------ loop
    def _should_stop(self) -> bool:
        if self.max_frames is not None and self._published >= self.max_frames:
            return True
        if (
            self.duration_s is not None
            and self._t_measure_start is not None
            and (time.monotonic() - self._t_measure_start) >= self.duration_s
        ):
            return True
        return False

    def _wait_for_release(self) -> bool:
        """Spin until one outstanding frame is released (or the graph finishes)."""
        target = len(self.window) - 1
        deadline = time.monotonic() + self.ack_timeout
        while rclpy.ok() and not self._done and len(self.window) > target:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            rclpy.spin_once(self, timeout_sec=min(0.01, remaining))
        return len(self.window) <= target

    def _drain(self) -> None:
        """
        Wait for every published frame, so the records are not short a tail.

        The stop condition is on *published* frames; with a window there are up
        to ``inflight - 1`` frames still in the graph when it triggers. Stopping
        the terminal node before they land would truncate ``records.jsonl`` by
        that many frames and hide a real result behind a ``window_short``
        warning.
        """
        while rclpy.ok() and not self._done and len(self.window):
            if not self._wait_for_release():
                self.get_logger().error(
                    f'{len(self.window)} frame(s) did not complete within '
                    f'{self.ack_timeout:.1f} s while draining '
                    '(is the downstream chain healthy?)'
                )
                return

    def _request_finish(self) -> None:
        """Tell the terminal node to write the summary, then wait for it."""
        if not rclpy.ok():
            return
        deadline = time.monotonic() + self.end_timeout
        next_publish = 0.0
        while rclpy.ok() and not self._done:
            now = time.monotonic()
            if now >= deadline:
                self.get_logger().error(
                    f'no done acknowledgement after {self.end_timeout:.1f} s; exiting'
                )
                return
            if now >= next_publish:
                self.stop_publisher.publish(Empty())
                next_publish = now + 0.5
            rclpy.spin_once(self, timeout_sec=0.01)

    def run(self) -> None:
        source = open_source(self.source_path, self.rotate, loops=self.loops)
        duration = 'none' if self.duration_s is None else f'{self.duration_s:g}s'
        self.get_logger().info(
            f'source: {source.description} (kind={source.kind}, fps={source.fps:.1f}, '
            f'rotate={self.rotate}, loops={self.loops}, duration={duration}, '
            f'inflight={self.inflight})'
        )
        iterator = iter(source)
        try:
            while rclpy.ok() and not self._done:
                if not self._should_stop() and self.window.has_room:
                    # Capture timing starts before decoding, exactly like the
                    # monolith: acq_ms is the decoder's + rotation's cost.
                    t_acq = time.monotonic()
                    frame = next(iterator, _SENTINEL)
                    if frame is _SENTINEL:
                        break
                    acq_ms = (time.monotonic() - t_acq) * 1e3

                    occupancy = self.window.add(frame.seq)
                    self._published += 1
                    meta = frame_meta(
                        seq=frame.seq,
                        loop=frame.loop,
                        t_mono=t_acq,
                        acq_ms=acq_ms,
                        inflight=occupancy,
                    )
                    # Stamped before the payload is built: conversion is part of
                    # what a hop costs, and hop_ms is measured up to it.
                    meta.t_pub = time.monotonic()
                    self.publisher.publish(image_frame(frame.image, meta))

                    if self._published == 1:
                        # Frame 0 is synchronous: its ack is the discovery gate
                        # and the engine-ready gate, so no later frame is queued
                        # in front of a graph that is still forming.
                        if not self._wait_for_release():
                            if rclpy.ok() and not self._done:
                                self.get_logger().error(
                                    f'frame {frame.seq} was not acknowledged within '
                                    f'{self.ack_timeout:.1f} s '
                                    '(is the downstream chain healthy?)'
                                )
                            break
                    continue

                if not len(self.window):
                    break  # source exhausted (or stopped) with an empty window
                if not self._wait_for_release():
                    if rclpy.ok() and not self._done:
                        self.get_logger().error(
                            f'{len(self.window)} frame(s) were not acknowledged '
                            f'within {self.ack_timeout:.1f} s '
                            '(is the downstream chain healthy?)'
                        )
                    break
        finally:
            source.close()
            self._drain()
            self._request_finish()

        elapsed = time.monotonic() - self._t_start
        fps = (self._published / elapsed) if elapsed > 0 else 0.0
        self.get_logger().info(
            f'published {self._published} frames in {elapsed:.1f} s '
            f'({fps:.2f} fps incl. startup, inflight={self.inflight})'
        )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )
    rclpy.init(args=argv)
    node = AcquisitionNode()
    try:
        node.run()
    except (KeyboardInterrupt, ExternalShutdownException):
        # Expected when the launch shuts the chain down after a run.
        if rclpy.ok():
            node.get_logger().info('shutting down')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

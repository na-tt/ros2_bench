"""
Stage 1 node: drive a source and publish one ``ImageFrame`` at a time.

The node owns the pacing. It publishes a frame and waits for the terminal
node's ack before decoding the next one, so the graph is demand-driven exactly
like the monolith's blocking loop: one frame in flight, no queue consuming the
latency it is supposed to measure, no silently dropped frames.

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

LOGGER = logging.getLogger('yolocar')

_SENTINEL = object()


class AcquisitionNode(Node):
    """Publish the configured source as ``ImageFrame`` messages, one at a time."""

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
        self.ack_timeout = float(self.declare_parameter('ack_timeout', 60.0).value)
        self.end_timeout = float(self.declare_parameter('end_timeout', 30.0).value)
        log_level = str(self.declare_parameter('log_level', 'INFO').value)
        LOGGER.setLevel(log_level.upper())

        self.publisher = self.create_publisher(
            ImageFrame, topics.FRAME_TOPIC, qos.latest()
        )
        self.stop_publisher = self.create_publisher(
            Empty, topics.STOP_TOPIC, qos.control(depth=1)
        )
        self.create_subscription(
            FrameAck, topics.ACK_TOPIC, self._on_ack, qos.control()
        )
        self.create_subscription(
            Empty, topics.DONE_TOPIC, self._on_done, qos.control(depth=1)
        )

        self._acked_seq = -1
        self._done = False
        self._t_measure_start: float | None = None
        self._frames = 0
        self._t_start = time.monotonic()

    # ------------------------------------------------------------- callbacks
    def _on_ack(self, msg: FrameAck) -> None:
        if int(msg.seq) > self._acked_seq:
            self._acked_seq = int(msg.seq)
        # The inference node reports when its engine became ready; use it as
        # the measurement clock origin (the monolith starts at engine load).
        if self._t_measure_start is None and msg.engine_loaded_t > 0.0:
            self._t_measure_start = float(msg.engine_loaded_t)

    def _on_done(self, _msg: Empty) -> None:
        self._done = True

    # ------------------------------------------------------------------ loop
    def _should_stop(self) -> bool:
        if self.max_frames is not None and self._frames >= self.max_frames:
            return True
        if (
            self.duration_s is not None
            and self._t_measure_start is not None
            and (time.monotonic() - self._t_measure_start) >= self.duration_s
        ):
            return True
        return False

    def _wait_for_ack(self, seq: int) -> bool:
        """Spin until the terminal node acknowledges ``seq``."""
        deadline = time.monotonic() + self.ack_timeout
        while rclpy.ok() and not self._done and self._acked_seq < seq:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            rclpy.spin_once(self, timeout_sec=min(0.01, remaining))
        return self._acked_seq >= seq

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
            f'rotate={self.rotate}, loops={self.loops}, duration={duration})'
        )
        iterator = iter(source)
        try:
            while rclpy.ok() and not self._done:
                if self._should_stop():
                    break

                # Capture timing starts before decoding, exactly like the
                # monolith: acq_ms is the decoder's + rotation's cost.
                t_acq = time.monotonic()
                frame = next(iterator, _SENTINEL)
                if frame is _SENTINEL:
                    break
                acq_ms = (time.monotonic() - t_acq) * 1e3

                self._acked_seq = -1
                self.publisher.publish(
                    image_frame(
                        frame.image,
                        frame_meta(
                            seq=frame.seq,
                            loop=frame.loop,
                            t_mono=t_acq,
                            acq_ms=acq_ms,
                        ),
                    )
                )
                if not self._wait_for_ack(frame.seq):
                    if rclpy.ok() and not self._done:
                        self.get_logger().error(
                            f'frame {frame.seq} was not acknowledged within '
                            f'{self.ack_timeout:.1f} s '
                            '(is the downstream chain healthy?)'
                        )
                    break
                self._frames += 1
        finally:
            source.close()
            self._request_finish()

        elapsed = time.monotonic() - self._t_start
        fps = (self._frames / elapsed) if elapsed > 0 else 0.0
        self.get_logger().info(
            f'published {self._frames} frames in {elapsed:.1f} s '
            f'({fps:.2f} fps incl. startup)'
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

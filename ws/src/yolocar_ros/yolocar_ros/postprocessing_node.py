"""
Stages 4-6 node: NMS -> domain filter -> coordinate adaptation.

An alias for :class:`yolocar_ros.chain_node.ChainNode` pinned to the ``post``
stage, running all three stages by default. ``post_stages:=nms`` /
``filter,coord`` / ``filter`` / ``coord`` cuts the stage instead, and whichever
node ends up running ``coord`` becomes the terminal one -- it writes
``records.jsonl``/``summary.json``, sends the ack that paces the acquisition
node, and owns the detections sidecar.

Stage order is the canonical one from ``AGENTS.md`` and the monolith: NMS over
all classes, then the road-class filter, then the inverse letterbox. All three
are imported from :mod:`yolocar.postprocessing`; nothing is reimplemented here.
"""

from __future__ import annotations

from .chain_node import entry_point

main = entry_point('post', 'yolocar_postprocessing')

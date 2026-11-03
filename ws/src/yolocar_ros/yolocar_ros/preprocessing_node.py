"""
Stage 2 node: image conditioning at source resolution.

An alias for :class:`yolocar_ros.chain_node.ChainNode` pinned to the ``pre``
stage, so this executable keeps its name, its console-script entry and its
parameter surface while the implementation lives in one place. ``chain_node``
explains the design; the short version is that a stage span is data, so a shape
like ``acq | pre+infer | post`` needs no new binary.

In the four-node chain this node reads ``yolocar/frame`` and writes
``yolocar/input``; when stage 2 is split, each instance is given its own
``in_topic``/``out_topic`` and its own ``pre_ops`` subset.
"""

from __future__ import annotations

from .chain_node import entry_point

main = entry_point('pre', 'yolocar_preprocessing')

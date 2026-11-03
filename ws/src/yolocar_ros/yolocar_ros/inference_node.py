"""
Stage 3 node: model input contract + TensorRT invocation + decode.

An alias for :class:`yolocar_ros.chain_node.ChainNode` pinned to the ``infer``
stage. Two properties of the generalised node matter here and are worth keeping
in view at the stage that owns the engine:

* the engine is loaded by the node that runs ``infer`` and by nobody else, so
  ``stages:=pre`` still never touches TensorRT;
* the *input contract* comes with it -- :func:`yolocar.model_input.prepare`
  letterboxes the incoming BGR frame, converts it to RGB NCHW and casts it to
  the engine's dtype. That is why ``input_dtype`` is not a launch argument: the
  engine is the source of truth, so a mismatch cannot be passed in by hand.

The stage boundary is *after* the decode/threshold step
(:func:`yolocar.inference.decode_predictions` + ``select_candidates``), so
``infer_ms`` includes the contract and the decode: compare ``infer + post`` as
one block against the monolith, not the two buckets individually.
"""

from __future__ import annotations

from .chain_node import entry_point

main = entry_point('infer', 'yolocar_inference')

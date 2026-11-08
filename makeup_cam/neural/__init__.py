"""Learned finishing stage (Plan B): the runtime that runs an optional ONNX model after the classic effects.

Binding rules: ``docs/dev/contract-plan-b.md``. This package ships without any model, and no model it can
load today says anything about a look. Everything here is optional: without ``onnxruntime`` or a model
the stage stays off and the output is the classic output, bit for bit.

This file imports nothing, so that ``makeup_cam.neural.geometry`` also works in the training environment
and ``makeup_cam.neural.worker`` starts without Qt or MediaPipe.

- ``geometry``: the aligned crop, the edge window, the weight of what a model may change.
- ``model_files``: where models live, ``model.json``, and the gate that decides what reaches a virtual camera.
- ``worker``: the inference process and its pipe protocol (at import only numpy; onnxruntime and cv2 when it runs).
- ``delta_filter``: checks, clamps, windows and smooths a model's output. Runs in the worker.
- ``backend``: the ``Backend`` protocol, and the backend that drives the worker process.
- ``stage``: ``NeuralFinishEffect``: health, fading, restarts. Registered by ``makeup_cam.effects.neural``.
"""

"""Training and export for the learned finishing stage (docs/dev/contract-plan-b.md, section 10).

The app never imports this package. Importing it pulls in nothing: each module states what it needs.

Without torch (also in the app's own environment):
- ``config``: one JSON file per run; strict, and free of paths and personal data.
- ``layout``: the dataset layout of the contract, and the capture guide's session folders.
- ``masks``, ``augment``, ``samples``: crops, what a model may change, webcam lighting and noise, one training sample.
- ``metadata``: ``model.json``, what makes a model a stand-in, the operator rules of the contract.
- ``heldout``: how close a model gets to the targets, against leaving the crop unchanged.
- ``latency``: a model's latency under ONNX Runtime (run in ``.venv-neural``).
- ``pairs``: makes datasets (run in ``.venv-neural``: it needs the classic pipeline and MediaPipe).

With torch (``.venv-train``):
- ``models``, ``losses``, ``checkpoint``, ``train``, ``export``.
"""

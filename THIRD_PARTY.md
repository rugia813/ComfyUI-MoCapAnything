# Third-party code and assets

The inference model in `model/` is adapted from [phongdaot/MocapAnything](https://github.com/phongdaot/MocapAnything), revision `c99fb22c2f6001958c73c197065322b98ca0b93b`, under the MIT license (Copyright 2026 Kehong Gong). Its license is retained in `LICENSE`.

Changes remove training and download paths, preserve checkpoint keys, use ComfyUI operations/attention/prefetch, and provide a local inference integration. The graph biases and temporal boundary behavior follow that revision. Native ComfyUI supplies DINOv2 and the optimized RoPE implementation.

Weights and reference data are separate downloads and are not included in this source package:

- [MoCapAnything V2 weights](https://huggingface.co/kehong/MoCapAnythingV2-weights)
- [DINOv2-Large](https://huggingface.co/facebook/dinov2-large)
- [MoCapAnything example data](https://huggingface.co/datasets/kehong/MoCapAnythingV2-data-sample); animal assets derive from Truebones Zoo and retain their own terms.

No BRIA RMBG, SAM2, TripoSG, Gradio, Blender, or upstream training implementation is bundled. Use a separately installed foreground-mask node when needed.

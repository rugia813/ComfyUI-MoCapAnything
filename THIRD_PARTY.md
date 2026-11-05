# Third-party code and assets

The inference model in `model/` is adapted from [phongdaot/MocapAnything](https://github.com/phongdaot/MocapAnything), revision `c99fb22c2f6001958c73c197065322b98ca0b93b`, under the MIT license (Copyright 2026 Kehong Gong). Its license is retained in `LICENSE`.

Changes remove training and download paths, preserve checkpoint keys, use ComfyUI operations/attention/prefetch, and provide a local inference integration. The graph biases and temporal boundary behavior follow that revision. Native ComfyUI supplies DINOv2 and the optimized RoPE implementation.

Joint name cleanup and topology relations in `rig.py` are adapted from upstream's `preprocess/build_species_info.py` under the same license. Custom-rig preparation uses the rest pose or embedded animation instead of upstream's animation dataset. Native ComfyUI supplies T5-base and GLB serialization helpers. Blender is an optional external application, not bundled.

Weights and reference data are separate downloads and are not included in this source package:

- [MoCapAnything V2 weights](https://huggingface.co/kehong/MoCapAnythingV2-weights)
- [DINOv2-Large](https://huggingface.co/facebook/dinov2-large)
- [Google T5-base](https://huggingface.co/google-t5/t5-base), for custom-rig joint-name features
- [MoCapAnything example data](https://huggingface.co/datasets/kehong/MoCapAnythingV2-data-sample); animal assets derive from Truebones Zoo and retain their own terms.

No BRIA RMBG, SAM, TripoSG, Gradio, Blender, or upstream training implementation is bundled. The example workflow uses ComfyUI's built-in background removal and SAM3 nodes.

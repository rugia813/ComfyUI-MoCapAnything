# Third-party code and assets

The inference model in `model/` is adapted from [phongdaot/MocapAnything](https://github.com/phongdaot/MocapAnything), revision `c99fb22c2f6001958c73c197065322b98ca0b93b`, under the MIT license (Copyright 2026 Kehong Gong). Its license is retained in `LICENSE`.

Changes remove training and download paths, preserve checkpoint keys, use ComfyUI operations/attention/prefetch, and provide a local inference integration. The graph biases and temporal boundary behavior follow that revision. ComfyUI loads the DINOv2 checkpoint and supplies the optimized RoPE implementation.

Joint name cleanup and topology relations in `rig.py` are adapted from upstream's `preprocess/build_species_info.py` under the same license. Custom-rig preparation uses the rest pose or embedded animation instead of upstream's animation dataset. Native ComfyUI supplies T5-base and GLB serialization helpers. Blender is required for rig preparation and animated character export; it is not bundled.

Model weights are separate downloads and are not included in this source package. Their licenses are separate from this repository's MIT license:

- [MoCapAnything V2 weights](https://huggingface.co/kehong/MoCapAnythingV2-weights): MIT.
- [DINOv2-Large](https://huggingface.co/facebook/dinov2-large): Apache 2.0.
- [Google T5-base](https://huggingface.co/google-t5/t5-base): Apache 2.0.
- [BiRefNet](https://huggingface.co/Comfy-Org/BiRefNet): MIT.
- [SAM3.1](https://huggingface.co/Comfy-Org/sam3.1): Meta's [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE), which has use and redistribution conditions. Review it before using the example workflow.

No BRIA RMBG, SAM, TripoSG, Gradio, Blender, or upstream training implementation is bundled. The example workflow uses ComfyUI's built-in background removal and SAM3 nodes.

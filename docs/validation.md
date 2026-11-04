# Validation

Tested on 2026-11-04 with the installed ComfyUI checkout, Python 3.12, PyTorch 2.10.0+cu130, and an NVIDIA RTX 4090. Upstream comparison revision: `c99fb22c2f6001958c73c197065322b98ca0b93b`.

- The released checkpoint loads strictly: all keys match, 38,547,181 parameters.
- Nine unit/contract checks cover V3 registration, UI/API workflow links, reference content fingerprints, safe paths, rejected pickled references, hierarchy cycles, normalization, mask cropping, BVH rotations/fps, and non-overwriting saves.
- CUDA model comparison against upstream passes for short and long sequences. The initial 1/8/97-frame comparison had maximum pose error below `3.6e-7` and rotation-component error below `1.9e-6`. Chunked 97-frame output differed from full-sequence output by less than `7.4e-7`.
- The final 305-frame regression also passes: pose error below `4.8e-7`, rotation-component error below `1.3e-6`, and chunk/full rotation difference below `6.2e-7`. All 305 frames are retained.
- CPU model comparison passes for 1 and 8 frames; maximum rotation-component error below `6.3e-6`.
- A complete 109-frame, 37-joint, 30-fps sample passed through the actual node methods and ComfyUI's prompt validator/graph executor, using the native video and CLIP Vision nodes. The BVH hierarchy round-trips; all exported positions and rotations are finite.
- A three-frame preview reconstructed from the exported rotations and rest offsets was visually inspected; it shows the sample's upright/rolling/upright motion. This is a skeleton preview, not a skinned-mesh or human quality acceptance test.
- The direct sample capture took 4.61 seconds after constructing the model objects; peak PyTorch CUDA allocation was approximately 978 MiB. This is one local smoke measurement, not a general performance or VRAM guarantee. Model download/initialization, other applications, driver allocations, input resolution, skeleton size, and hardware change the totals.

The UI workflow is checked structurally against the executed API graph. The existing live ComfyUI server was not restarted, and browser canvas rendering has not been inspected. Reloading ComfyUI is required to discover the new extension.

CUDA and CPU numerical agreement does not establish motion quality for unseen videos/rigs. Automatic subject segmentation, direct FBX import/rig preparation, world-space root tracking, and mesh rendering are outside this integration. ROCm, MPS, DirectML, XPU, and NPU have not been hardware-tested. Native dispatch is used without hard-coded CUDA transfers in node/model code.

The local Torch installation emitted warnings about missing `cuobjdump.exe` and `nvdisasm.exe`; the tested paths still completed successfully.

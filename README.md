# ComfyUI-MoCapAnything

Local [MoCapAnything V2](https://github.com/phongdaot/MocapAnything) motion capture: video frames and a reference skeleton → joint positions, local rotations, and BVH animation.

Four V3 nodes: **Load MoCapAnything V2**, **Load MoCapAnything Reference**, **MoCapAnything Capture**, and **Save MoCapAnything Motion**. The workflow reuses native **Load Video**, **Get Video Components**, and **CLIP Vision Loader**. Python dependencies are declared in `requirements.txt` and `pyproject.toml`. No background downloads, web services, or Blender installation are required.

## Install

Install ComfyUI first, including the PyTorch build appropriate for your hardware. Put this directory in `ComfyUI/custom_nodes/ComfyUI-MoCapAnything`. Requires Python 3.11+ and a current ComfyUI with V3 schemas, native DINOv2-Large, `AttentionTensorContainer`, `ComfyAttention`, and `comfy.model_prefetch`. Older ComfyUI versions are not supported; installing Python dependencies does not provide these ComfyUI APIs.

Install this package's dependencies in **ComfyUI's Python environment**, then restart ComfyUI. From this node directory, with ComfyUI's virtual environment activated:

```bash
python -m pip install -r requirements.txt
```

For Windows portable, run this from the node directory instead:

```powershell
& "..\..\..\python_embeded\python.exe" -m pip install -r requirements.txt
```

This installs missing dependencies and retains already-satisfied packages. Torch and torchvision are not pinned to a particular CUDA/platform build. On a new installation, follow ComfyUI's hardware-specific PyTorch setup before running this command.

Use ComfyUI's Python for the commands below; Windows portable installations use `python_embeded/python.exe`. Run conversion commands from this node directory.

1. Download [`video2pos2rot_epoch60.pt`](https://huggingface.co/kehong/MoCapAnythingV2-weights/blob/main/video2pos2rot_epoch60.pt) and convert it:

   ```bash
   python tools/convert_checkpoint.py /path/to/video2pos2rot_epoch60.pt ../../models/diffusion_models/mocapanything/mocapanything_v2.safetensors
   ```

2. Download DINOv2-Large's [`model.safetensors`](https://huggingface.co/facebook/dinov2-large/blob/main/model.safetensors) to `ComfyUI/models/clip_vision/dinov2-large.safetensors`. Select it in the native CLIP Vision loader. Other CLIP/DINO models do not provide the required features.

3. Prepare a reference as described below, placing its `.npz` in `ComfyUI/input/mocapanything/`.

4. Import [`example_workflows/video_to_bvh.json`](example_workflows/video_to_bvh.json). Select your video, checkpoint, encoder, and reference. Queue the workflow. The native video components node supplies the frame rate automatically.

The runtime only reads local files. Downloading and conversion are explicit setup steps. The checkpoint converter uses `torch.load(weights_only=True)`; nodes load only safetensors and non-pickled NPZ files.

## Reference skeleton

A reference needs the actual skeleton, a matching reference image's DINOv2 features, reference joint positions/rotations, joint semantic embeddings, and the training scale. A BVH alone does not contain all of these values.

For the published example, obtain these files from the [sample dataset](https://huggingface.co/datasets/kehong/MoCapAnythingV2-data-sample/tree/main/zoo1030), preserving their layout beneath a local `zoo1030` directory:

```text
species_info_dict.npy
cache/__mesh2pose1002_species_scale_cache.pkl
bvh/Hamster#Hamster-RollAttack/y0.bvh
bvh_pose/Hamster#Hamster-RollAttack/y0.npz
npz_train_image_only/Hamster#Hamster-RollAttack/y0.npz
```

Then run:

```bash
python tools/convert_reference.py /path/to/zoo1030 "Hamster#Hamster-RollAttack/y0" ../../input/mocapanything/hamster.npz --trust-upstream-pickle
```

The flag is required because upstream's species and scale metadata contain Python pickle objects. Use it only for trusted local upstream data. The converted workflow reference contains ordinary numeric/string arrays and never executes pickle. `--frame` selects a matching pose/image frame; the small released example feature files generally contain only frame 0.

Custom rigs must first be prepared with [upstream's custom-rig workflow](https://github.com/phongdaot/MocapAnything/tree/main/examples/custom_rig), then packed in the same format. FBX import, rigging, and generation of reference semantic features are not performed by these nodes. See [`docs/reference_format.md`](docs/reference_format.md).

## Capture and outputs

- Supply an `IMAGE` batch in temporal order. A single image works too. The model uses float32; the native encoder follows ComfyUI's precision/device policy.
- Prefer a clearly visible subject with a clean background. Optionally connect `foreground_mask` from an existing segmentation node: **white = subject**, black = background. One mask broadcasts to all frames. Native Load Image's transparency mask has the opposite meaning; invert it first.
- Masks control foreground compositing and cropping. Without a mask, the complete image is used. Both paths add a white border and apply upstream's DINOv2 256-resize/224-crop preprocessing.
- `batch_size` controls encoder batching; reduce it for low VRAM. `chunk_size` controls motion inference with 40 frames of context on each side. All supplied frames are processed, including clips longer than 301 frames.
- BVH contains local rotations and the reference's bone lengths, with a stationary root. This model does **not** recover camera/world-space root translation. NPZ stores root-relative positions in reference units, `rot6d`, parents, offsets, names, and fps.
- Outputs go to `ComfyUI/output/mocapanything/` by default. Existing files are preserved. Output filenames appear on the save node; its two string outputs provide full paths.

Models use ComfyUI loading/offloading and optimized attention. CPU and CUDA were tested; other ComfyUI devices use native dispatch but have not been hardware-tested. Motion quality still depends on matching the subject to a useful reference and on input segmentation.

## Validation

Run with the ComfyUI Python environment:

```bash
python -m unittest discover -s tests -p "test_*.py"
python tests/compare_upstream.py --upstream /path/to/MocapAnything --checkpoint /path/to/mocapanything_v2.safetensors --reference /path/to/hamster.npz --device cuda
python tests/smoke_capture.py --checkpoint /path/to/mocapanything_v2.safetensors --encoder /path/to/dinov2-large.safetensors --reference /path/to/hamster.npz --video /path/to/hamster.mp4
```

Windows embedded Python omits the script directory from `sys.path`. For unit tests, use `python_embeded/python.exe -c "import sys,unittest; sys.path.insert(0,'tests'); unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests'))"` from this node directory.

See [`docs/validation.md`](docs/validation.md) for measured results and limits. Upstream attribution and asset terms are in [`THIRD_PARTY.md`](THIRD_PARTY.md).

# ComfyUI-MoCapAnything

Turn a video into character motion with [MoCapAnything V2](https://github.com/phongdaot/MocapAnything). Use a prepared reference to save BVH, or use your own rigged character and preview the animated GLB in ComfyUI.

## Install

This node needs a current ComfyUI and Python 3.11+. Install ComfyUI and its hardware appropriate PyTorch build first. Clone this repository into `ComfyUI/custom_nodes/`, then install dependencies with **the Python that runs ComfyUI**:

```bash
python -m pip install -r requirements.txt
```

For Windows portable, run this command from the node folder:

```powershell
& "..\..\..\python_embeded\python.exe" -m pip install -r requirements.txt
```

Download the [MoCapAnything V2 checkpoint](https://huggingface.co/kehong/MoCapAnythingV2-weights/blob/main/video2pos2rot_epoch60.pt) and convert it with the same Python:

```bash
python tools/convert_checkpoint.py /path/to/video2pos2rot_epoch60.pt ../../models/diffusion_models/mocapanything/mocapanything_v2.safetensors
```

Download [DINOv2-Large](https://huggingface.co/facebook/dinov2-large/blob/main/model.safetensors) to `ComfyUI/models/clip_vision/dinov2-large.safetensors`. For your own rig, also download the original [Google T5-base](https://huggingface.co/google-t5/t5-base/blob/main/model.safetensors) to `ComfyUI/models/text_encoders/t5-base.safetensors`. Restart ComfyUI after installing the node and models.

## Animate your own character

Open [custom_rig_to_3d.json](example_workflows/custom_rig_to_3d.json) in ComfyUI.

1. Install Blender 4.2+. The node finds `blender` on PATH, a normal Windows installation, or `Blender.app` in macOS Applications. There is no Blender path setting. If you use a portable Blender build, put its executable on PATH and restart ComfyUI.
2. In **Load 3D (Advanced)**, choose a rigged FBX, GLB, or glTF with **one skeleton, one root bone, and a skinned mesh**. A model visible in the 3D viewer may still be an unrigged mesh. For glTF, keep its `.bin` and texture files beside it; GLB is easier to share.
3. In **CLIP Loader**, select `t5-base.safetensors` with type `stable_audio`. Select `dinov2-large.safetensors` in **CLIP Vision Loader**. Choose the MoCapAnything checkpoint and your video.
4. Set `forward_axis` on **Prepare Rig Reference** to the direction the character faces in the Y-up 3D viewer, then queue the workflow. **Preview 3D (Advanced)** plays the animated character. Connect the animated `model_3d` to ComfyUI's **Save 3D** to keep the GLB.

The custom rig route builds a reference from one rest pose. It is experimental: motion quality may differ from a reference prepared with upstream animation data. The rig must already have bones and skin weights; this node does not rig a static mesh or transfer motion between unrelated skeletons. Root movement through the scene is not recovered.

## Use a prepared reference

Open [video_to_bvh.json](example_workflows/video_to_bvh.json). Select the video, checkpoint, DINOv2-Large encoder, and reference. It saves BVH and offers a skeleton preview without Blender or T5-base.

A reference `.npz` is a NumPy archive of skeleton and model features; it is **not** a 3D model. To make one from the [upstream sample dataset](https://huggingface.co/datasets/kehong/MoCapAnythingV2-data-sample/tree/main/zoo1030), keep the downloaded `zoo1030` folder structure and run:

```bash
python tools/convert_reference.py /path/to/zoo1030 "Hamster#Hamster-RollAttack/y0" ../../input/mocapanything/hamster.npz --trust-upstream-pickle
```

The flag permits reading pickle metadata from the upstream dataset; use it only with files you trust. The resulting reference loads without pickle. Motion `.npz` files saved by this node contain animation results and cannot be loaded as references.

The optional `foreground_mask` input uses **white for the subject**. BVH and motion files are saved under `ComfyUI/output/mocapanything/`. Model attribution is in [THIRD_PARTY.md](THIRD_PARTY.md).

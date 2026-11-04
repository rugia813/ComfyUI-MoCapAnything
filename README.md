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

Download the [MoCapAnything V2 checkpoint](https://huggingface.co/kehong/MoCapAnythingV2-weights/blob/main/video2pos2rot_epoch60.pt) to `ComfyUI/models/diffusion_models/mocapanything/video2pos2rot_epoch60.pt`. Select it in **Load MoCapAnything V2**. The first load converts it to `.safetensors` in the same folder; later loads reuse that file. You can also select an already converted `.safetensors` file.

Download [DINOv2-Large](https://huggingface.co/facebook/dinov2-large/blob/main/model.safetensors) to `ComfyUI/models/clip_vision/dinov2-large.safetensors`. For your own rig, also download the original [Google T5-base](https://huggingface.co/google-t5/t5-base/blob/main/model.safetensors) to `ComfyUI/models/text_encoders/t5-base.safetensors`. Restart ComfyUI after installing the node and models.

Install Blender 4.2+. The node finds `blender` on PATH, a normal Windows installation, or `Blender.app` in macOS Applications. There is no Blender path setting. If you use a portable Blender build, put its executable on PATH and restart ComfyUI.

## Animate your own character

Open [custom_rig_to_3d.json](example_workflows/custom_rig_to_3d.json) in ComfyUI.

The custom rig route builds a reference from one rest pose. It is experimental: motion quality may differ from a reference prepared with upstream animation data. The rig must already have bones and skin weights; this node does not rig a static mesh or transfer motion between unrelated skeletons. Root movement through the scene is not recovered.

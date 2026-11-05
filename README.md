# ComfyUI-MoCapAnything

Turn a video and a rigged character into an animated GLB with [MoCapAnything V2](https://github.com/phongdaot/MocapAnything). Preview the result in ComfyUI.

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

Download [DINOv2-Large](https://huggingface.co/facebook/dinov2-large/blob/main/model.safetensors) to `ComfyUI/models/clip_vision/dinov2-large.safetensors` and [Google T5-base](https://huggingface.co/google-t5/t5-base/blob/main/model.safetensors) to `ComfyUI/models/text_encoders/t5-base.safetensors`. Restart ComfyUI after installing the node and models.

Install Blender 4.2+. The node finds `blender` on PATH, a normal Windows installation, or `Blender.app` in macOS Applications. There is no Blender path setting. If you use a portable Blender build, put its executable on PATH and restart ComfyUI.

## Animate your own character

Open [custom_rig_to_3d.json](example_workflows/custom_rig_to_3d.json) in ComfyUI.

Load a video and a skinned FBX, GLB, or glTF character. The workflow prepares the rig, captures motion from the video, applies it to the character, previews the animation, and saves a GLB. No reference NPZ or matching example skeleton is needed.

The character needs bones and skin weights; a mesh without an armature cannot be animated by this node. Set `forward_axis` to the direction the character faces. If the FBX or GLB contains skeletal animation on the skinned rig, its poses help build the reference. For a static rig, the node uses its rest pose.

Upstream's [custom-rig pipeline](https://github.com/phongdaot/MocapAnything/tree/main/examples/custom_rig) builds references from a base FBX plus motion FBX files. A rest-pose reference is usable here, but animation embedded in the character can give better results. Root translation through the scene is not recovered.

Animated GLBs produced by this node store the ComfyUI prompt and workflow in `asset.extras.comfyui`.

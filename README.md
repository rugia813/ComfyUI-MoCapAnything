# ComfyUI-MoCapAnything

Turn a video and a rigged character into an animated GLB with [MoCapAnything V2](https://github.com/phongdaot/MocapAnything).

![MoCapAnything workflow showing video input, rig preparation, motion capture, and the animated 3D result](assets/Screenshot%202026-11-05%20160520.png)


## Install

Python 3.11+ is required. Install ComfyUI and its PyTorch build for your hardware first. Clone this repository into `ComfyUI/custom_nodes/`, then install the remaining dependencies with **the Python that runs ComfyUI**:

```bash
python -m pip install -r requirements.txt
```

For Windows portable, run this command from the node folder:

```powershell
& "..\..\..\python_embeded\python.exe" -m pip install -r requirements.txt
```

Download these five model files:

| Model | Place in `ComfyUI/models/` |
| --- | --- |
| [MoCapAnything V2](https://huggingface.co/kehong/MoCapAnythingV2-weights/blob/main/video2pos2rot_epoch60.pt) | `diffusion_models/video2pos2rot_epoch60.pt` |
| [DINOv2-Large](https://huggingface.co/facebook/dinov2-large/blob/main/model.safetensors) | `clip_vision/dinov2-large.safetensors` |
| [T5-base](https://huggingface.co/google-t5/t5-base/blob/main/model.safetensors) | `text_encoders/t5-base.safetensors` |
| [BiRefNet](https://huggingface.co/Comfy-Org/BiRefNet/blob/main/background_removal/birefnet.safetensors) | `background_removal/birefnet.safetensors` |
| [SAM3.1](https://huggingface.co/Comfy-Org/sam3.1/blob/main/sam3.1_multiplex_fp16.safetensors) | `checkpoints/sam3.1_multiplex_fp16.safetensors` |

The first MoCapAnything load converts the `.pt` file to `.safetensors` beside it, so the model folder must be writable and have room for both files. Later loads use the converted file. Restart ComfyUI after installing models.

Install Blender 4.2+. The node finds `blender` on PATH, a normal Windows installation, or `Blender.app` in macOS Applications. For a portable Blender build, put its executable on PATH and restart ComfyUI.

## Animate your own character

Open [custom_rig_to_3d.json](example_workflows/custom_rig_to_3d.json) in ComfyUI.

Upload your video and your skinned FBX, GLB, or glTF character. The workflow prepares the rig, uses BiRefNet and SAM3.1 to track the foreground, captures motion, and saves an animated GLB.

The character needs bones and skin weights; a mesh without an armature cannot be animated by this node. Set `forward_axis` to the direction the character faces. If the FBX or GLB contains skeletal animation on the skinned rig, its poses help build the reference. For a static rig, the node uses its rest pose.

Inspect the first-frame mask in **Mask Preview**. If it selects the wrong subject, connect a corrected first-frame mask to **SAM3 Video Track**. Capture also accepts a per-frame mask directly. The upstream demo uses SAM2; this example uses ComfyUI's SAM3.1, so masking and results may differ.

Upstream's [custom-rig pipeline](https://github.com/phongdaot/MocapAnything/tree/main/examples/custom_rig) builds references from a base FBX plus motion FBX files. A rest-pose reference is usable here, but animation embedded in the character can give better results. Root translation through the scene is not recovered.

Animated GLBs produced by this node store the ComfyUI prompt and workflow in `asset.extras.comfyui`.

The full workflow was tested on Windows with CUDA. Blender discovery has Windows, macOS, and Linux tests; full inference on macOS and Linux has not been verified here. GPU memory requirements depend on the video length, model, and other ComfyUI workloads.

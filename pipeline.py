import numpy as np
import torch
from PIL import Image
from transformers import BitImageProcessor, Dinov2Config, Dinov2Model

import comfy.model_management as mm
import comfy.model_patcher
import comfy.utils

from .model.network import MoCapAnything
from .motion import Motion


def load_model(path):
    state = comfy.utils.load_torch_file(path, safe_load=True)
    offload_device = mm.unet_offload_device()
    with torch.device(offload_device):
        model = MoCapAnything()
    model.load_state_dict(state, strict=True)
    model.eval()
    return comfy.model_patcher.ModelPatcher(model, load_device=mm.get_torch_device(), offload_device=offload_device)


def make_batch(reference, features, device):
    tensors = {key: torch.from_numpy(value).unsqueeze(0).to(device=device, dtype=torch.float32 if value.dtype.kind == "f" else None)
               for key, value in reference.tensors.items()}
    joints = len(reference.names)
    ancestors = np.eye(joints, dtype=bool)
    for joint, parent in enumerate(reference.parents):
        if parent >= 0:
            ancestors[joint] |= ancestors[parent]
    tensors.update(
        image_embed=features.unsqueeze(0).to(device=device, dtype=torch.float32),
        joint_mask=torch.ones((1, joints), device=device, dtype=torch.bool),
        ancestor_mask=torch.from_numpy(ancestors).unsqueeze(0).to(device),
        offset_a=torch.from_numpy((reference.offsets - reference.offsets[:1]) / reference.scale).unsqueeze(0).to(device),
    )
    tensors["memory_pose"] = tensors["ref_position"].unsqueeze(1)
    tensors["memory_rot6d"] = tensors["ref_rot6d_a"].unsqueeze(1)
    return tensors


def prepare_frames(images, masks=None):
    for index, image in enumerate(images):
        rgb = image[..., :3].detach().cpu().float()
        height, width = rgb.shape[:2]
        x0, y0, x1, y1 = 0, 0, width, height
        if masks is not None:
            alpha = masks[0 if len(masks) == 1 else index].detach().cpu().float()
            if alpha.shape != (height, width):
                alpha = torch.nn.functional.interpolate(alpha[None, None], size=(height, width), mode="bilinear", align_corners=False)[0, 0]
            rgb = rgb * alpha[..., None] + (1 - alpha[..., None])
            ys, xs = torch.where(alpha > 0)
            if len(xs):
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                y0, y1 = int(ys.min()), int(ys.max()) + 1
        crop = (rgb[y0:y1, x0:x1].clamp(0, 1).numpy() * 255).astype(np.uint8)
        height, width = crop.shape[:2]
        if width > height:
            px = int(width * 0.1)
            py = int(px + (width - height) / 2)
        else:
            py = int(height * 0.1)
            px = int(py + (height - width) / 2)
        yield Image.fromarray(np.pad(crop, ((py, py), (px, px), (0, 0)), constant_values=255))


def capture(model, clip_vision, reference, images, fps, batch_size, chunk_size, masks=None):
    if fps <= 0 or chunk_size <= 0:
        raise ValueError("fps and chunk_size must be positive.")
    progress = comfy.utils.ProgressBar(len(images) + (len(images) + chunk_size - 1) // chunk_size)
    features = encode_images(clip_vision, images, batch_size, masks, progress)
    positions, rotations = [], []
    # Conservative context for the released model's temporal receptive field.
    # Extra context makes long sequences match full-sequence inference at seams.
    halo = 40
    device = model.load_device
    for start in range(0, len(images), chunk_size):
        mm.throw_exception_if_processing_interrupted()
        end = min(start + chunk_size, len(images))
        left, right = max(0, start - halo), min(len(images), end + halo)
        mm.load_models_gpu([model], memory_required=(right - left) * len(reference.names) ** 2 * 256)
        batch = make_batch(reference, features[left:right], device)
        pos, rot = model.model(batch, transformer_options=model.model_options["transformer_options"])
        positions.append(pos[0, start-left:end-left].detach().to(device="cpu", dtype=torch.float32).numpy().copy() * reference.scale)
        rotations.append(rot[0, start-left:end-left].detach().to(device="cpu", dtype=torch.float32).numpy().copy())
        del batch, pos, rot
        progress.update(1)
    return Motion(reference, np.concatenate(positions), np.concatenate(rotations), fps)


def encode_images(clip_vision, images, batch_size=8, masks=None, progress=None):
    if clip_vision.model_type != "dinov2" or clip_vision.config["hidden_size"] != 1024:
        raise ValueError("Load facebook/dinov2-large with the native CLIP Vision loader.")
    if not len(images):
        raise ValueError("Capture needs at least one video frame.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if masks is not None and len(masks) not in (1, len(images)):
        raise ValueError("Provide one foreground mask or one mask per frame.")
    # The HF DINOv2 processor uses 256 resize / 224 crop. Native encode_image
    # uses 518 instead, so call the loaded encoder with the original preprocessing.
    processor = BitImageProcessor(size={"shortest_edge": 256}, crop_size={"height": 224, "width": 224},
                                  image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225], resample=Image.Resampling.BICUBIC)
    progress = progress if progress is not None else comfy.utils.ProgressBar(len(images))
    prepared = list(prepare_frames(images, masks))
    # The native ComfyUI DINOv2 backbone produces materially different tokens
    # from upstream's Hugging Face Dinov2Model even with identical weights and
    # pixel values. Reuse the loaded checkpoint, but run the upstream backbone.
    config = Dinov2Config(**clip_vision.config,
                         intermediate_size=clip_vision.config["hidden_size"] * clip_vision.config["mlp_ratio"])
    with torch.device("meta"):
        encoder = Dinov2Model(config)
    encoder.load_state_dict(clip_vision.get_sd(), strict=True, assign=True)
    dtype = clip_vision.dtype if clip_vision.load_device.type != "cpu" else torch.float32
    encoder = encoder.to(device=clip_vision.load_device, dtype=dtype).eval()
    features = []
    try:
        with torch.inference_mode():
            for start in range(0, len(images), batch_size):
                mm.throw_exception_if_processing_interrupted()
                end = min(start + batch_size, len(images))
                pixels = processor(images=prepared[start:end], return_tensors="pt")["pixel_values"]
                pixels = pixels.to(device=clip_vision.load_device, dtype=dtype)
                hidden = encoder(pixel_values=pixels).last_hidden_state
                features.append(hidden.detach().to(device="cpu", dtype=torch.float32))
                progress.update(end - start)
    finally:
        del encoder
        mm.soft_empty_cache()
    return torch.cat(features)

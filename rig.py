"""Local rig preparation and Blender bridge. No downloads or shell commands."""
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import json
import os
import re
import subprocess
import tempfile

import numpy as np
import torch
from PIL import Image
from comfy_api.latest import Types
import comfy.model_management as mm
from comfy.text_encoders.sa_t5 import SAT5Model

from .motion import Reference, rotation_matrices
from .pipeline import encode_images
from .blender import find_blender


@dataclass
class PreparedRig:
    blend: bytes
    metadata: dict
    reference: Reference
    blender: str


def run_blender(executable, directory, config):
    job = directory / "job.json"
    job.write_text(json.dumps(config), encoding="utf8")
    command = [executable, "--background", "--factory-startup", "--disable-autoexec", "--python-exit-code", "1",
               "--python", str(Path(__file__).parent / "tools/blender_rig.py"), "--", str(job)]
    with (directory / "blender.log").open("w", encoding="utf8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            while True:
                mm.throw_exception_if_processing_interrupted()
                try:
                    process.wait(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    if process.returncode:
        detail = (directory / "blender.log").read_text(encoding="utf8", errors="replace")[-6000:]
        raise RuntimeError(f"Blender rig processing failed:\n{detail}")
    return executable


def topology(parents):
    # Same edge IDs and capped tree distances as upstream build_species_info.py.
    count = len(parents)
    hop = np.zeros((count, count), dtype=np.int64)
    edge = np.full((count, count), 4, dtype=np.int64)
    for i in range(count):
        leaf = True
        for j in range(count):
            if i == j:
                edge[i, j] = 0
            elif parents[j] == i:
                edge[i, j], leaf = 2, False
            elif j == parents[i]:
                edge[i, j] = 1
            elif parents[j] == parents[i]:
                edge[i, j] = 3
            if j < i:
                hop[i, j] = hop[j, i]
            elif j > i:
                hop[i, j] = 1 if parents[j] == i else hop[i, parents[j]] + 1
        if leaf:
            edge[i, i] = 5
    return np.minimum(hop, 5), edge


def clean_joint_name(name):
    original = name
    for prefix in ('BN_Bip01', 'Bip01', 'BN', 'NPC', 'jt', 'Sabrecat', 'Elk', 'mixamorig:', 'mixamorig'):
        if name.startswith(prefix):
            name = name[len(prefix):]
    name = name or original
    replacements = {'L': 'Left', 'l': 'Left', 'R': 'Right', 'r': 'Right', 'momo': 'Thigh',
                    'sippo': 'Tail', 'mune': 'Chest', 'hiza': 'Knee', 'hara': 'Stomach', 'ashi': 'Leg',
                    'hiji': 'Elbow', 'koshi': 'Hips', 'te': 'Hand', 'kubi': 'Neck', 'atama': 'Head',
                    'ago': 'Jaw', 'kata': 'Shoulder', 'Tai': 'Tail'}
    parts = [re.sub(r'[\d_]+', '', part) for part in re.split('(?=[A-Z]|_)', name)]
    return ' '.join(replacements.get(part, part) for part in parts if part) or name


def joint_features(clip, names):
    if not isinstance(clip.cond_stage_model, SAT5Model):
        raise ValueError("Use the native CLIP Loader with google-t5/t5-base weights and type stable_audio. T5-XXL/UMT5 are incompatible.")
    features = []
    for name in names:
        mm.throw_exception_if_processing_interrupted()
        tokens = clip.tokenize(clean_joint_name(name), disable_weights=True)
        count = sum(token[0] != 0 for section in tokens['t5base'] for token in section)
        hidden = clip.encode_from_tokens(tokens)
        # Native T5-base zeros padding. Include EOS, matching upstream's masked mean.
        features.append((hidden.sum(dim=1)[0] / count).detach().float().cpu().numpy())
    return np.stack(features)


def prepare_rig(model_3d, clip, clip_vision, forward_axis):
    if model_3d is None or model_3d.format.lower() not in ('fbx', 'glb', 'gltf'):
        raise ValueError("Connect a rigged FBX, GLB or glTF from Load 3D (Advanced).")
    if model_3d.format.lower() == 'gltf' and not model_3d.is_disk_backed:
        raise ValueError("glTF requires its local .bin/textures beside the source file. Use a disk-backed glTF or embedded GLB.")
    with tempfile.TemporaryDirectory(prefix="mocap_rig_") as temp:
        directory = Path(temp)
        if model_3d.is_disk_backed:
            source = str(Path(model_3d.get_source()).resolve())
        else:
            source = model_3d.save_to(str(directory / f"source.{model_3d.format}"))
        blender = run_blender(find_blender(), directory, {"mode": "prepare", "source": source, "forward_axis": forward_axis})
        metadata = json.loads((directory / "rig.json").read_text(encoding="utf8"))
        names = metadata['names']
        parents = np.array(metadata['parents'], dtype=np.int64)
        positions = np.array(metadata['positions'], dtype=np.float32)
        offsets = np.array(metadata['offsets'], dtype=np.float32)
        with Image.open(directory / "reference.png") as image:
            rgba = torch.from_numpy(np.array(image.convert('RGBA'), dtype=np.float32) / 255)[None]
        rgb = rgba[..., :3] * rgba[..., 3:] + 1 - rgba[..., 3:]
        image_features = encode_images(clip_vision, rgb, masks=rgba[..., 3])[0].numpy()
        semantic_features = joint_features(clip, names)
        # Upstream measures all training motions over 12 yaws. Here only the rest pose is available.
        yaw = np.arange(12) * np.pi / 6
        rotated_x = positions[:, 0, None] * np.cos(yaw) + positions[:, 2, None] * np.sin(yaw)
        rotated_z = -positions[:, 0, None] * np.sin(yaw) + positions[:, 2, None] * np.cos(yaw)
        scale = float(max(np.abs(rotated_x).max(), np.abs(rotated_z).max(), np.abs(positions[:, 1]).max()))
        if scale <= 1e-8:
            raise ValueError("Rig has no spatial extent. Use a rig with distinct joint positions.")
        hop, edge = topology(parents)
        static_pos = np.zeros(len(names), dtype=bool)
        static_pos[0] = True
        ref = Reference(names, parents, offsets, scale, {
            'ref_position': positions / scale,
            'ref_rot6d_a': np.tile([1, 0, 0, 0, 1, 0], (len(names), 1)).astype(np.float32),
            'ref_image_embed': image_features, 'joint_t5embed': semantic_features,
            'graph_hop': hop, 'graph_edge': edge, 'static_pos_joint_mask': static_pos,
            'static_rot_joint_mask': np.zeros(len(names), dtype=bool)})
        rig = PreparedRig((directory / "rig.blend").read_bytes(), metadata, ref, blender)
    return ref, rig, rgb


def animate_rig(motion, rig):
    if motion.reference is not rig.reference:
        raise ValueError("Connect motion captured with this Prepare Rig Reference node's reference. Unrelated rigs require retargeting.")
    with tempfile.TemporaryDirectory(prefix="mocap_animate_") as temp:
        directory = Path(temp)
        (directory / "rig.blend").write_bytes(rig.blend)
        (directory / "rig.json").write_text(json.dumps(rig.metadata), encoding="utf8")
        np.save(directory / "rotations.npy", rotation_matrices(motion.rotations), allow_pickle=False)
        run_blender(rig.blender, directory, {"mode": "animate", "fps": motion.fps})
        result = (directory / "animated.glb").read_bytes()
    return Types.File3D(BytesIO(result), "glb")

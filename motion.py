from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


@dataclass
class Reference:
    names: list[str]
    parents: np.ndarray
    offsets: np.ndarray
    scale: float
    tensors: dict[str, np.ndarray]


@dataclass
class Motion:
    reference: Reference
    positions: np.ndarray
    rotations: np.ndarray
    fps: float


def read_bvh_skeleton(path):
    names, parents, offsets, stack = [], [], [], []
    pending = None
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "MOTION":
            break
        if fields[0] in ("ROOT", "JOINT"):
            pending = len(names)
            names.append(" ".join(fields[1:]))
            parents.append(stack[-1] if stack else -1)
            offsets.append(None)
        elif fields[0] == "End":
            pending = None
        elif fields[0] == "{":
            stack.append(pending)
        elif fields[0] == "}":
            stack.pop()
        elif fields[0] == "OFFSET" and stack[-1] is not None:
            offsets[stack[-1]] = [float(v) for v in fields[1:]]
    return names, np.asarray(parents, dtype=np.int64), np.asarray(offsets, dtype=np.float32)


def load_reference(path):
    with np.load(path, allow_pickle=False) as data:
        names = data["names"].tolist()
        parents = data["parents"].astype(np.int64)
        offsets = data["offsets"].astype(np.float32)
        scale = float(data["global_scale"])
        tensors = {key: data[key].copy() for key in (
            "ref_position", "ref_rot6d_a", "ref_image_embed", "joint_t5embed",
            "graph_hop", "graph_edge", "static_pos_joint_mask", "static_rot_joint_mask",
        )}
    joints = len(names)
    if not joints or parents.shape != (joints,) or parents[0] != -1 or any(p < 0 or p >= i for i, p in enumerate(parents[1:], 1)):
        raise ValueError("Reference must contain one root and parents ordered before their children.")
    if any(not isinstance(name, str) or not re.fullmatch(r"[^\s{}]+", name) for name in names):
        raise ValueError("BVH joint names must be non-empty and contain no whitespace or braces.")
    shapes = {"ref_position": (joints, 3), "ref_rot6d_a": (joints, 6), "joint_t5embed": (joints, 768),
              "graph_hop": (joints, joints), "graph_edge": (joints, joints),
              "static_pos_joint_mask": (joints,), "static_rot_joint_mask": (joints,)}
    if offsets.shape != (joints, 3) or not np.isfinite(offsets).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Reference offsets must be finite [J,3] values and global_scale must be positive.")
    for key, shape in shapes.items():
        if tensors[key].shape != shape or not np.isfinite(tensors[key]).all():
            raise ValueError(f"Reference {key} must contain finite values with shape {shape}.")
    image = tensors["ref_image_embed"]
    if image.ndim != 2 or image.shape[1] != 1024 or not image.shape[0] or not np.isfinite(image).all():
        raise ValueError("Reference image features must have shape [patches,1024] from DINOv2-Large.")
    for key in ("graph_hop", "graph_edge"):
        values = tensors[key]
        if (values < 0).any() or (values > 5).any() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"Reference {key} must contain integer embedding indices from 0 to 5.")
        tensors[key] = values.astype(np.int64)
    for key in ("static_pos_joint_mask", "static_rot_joint_mask"):
        tensors[key] = tensors[key].astype(bool)
    return Reference(names, parents, offsets, scale, tensors)


def rotation_matrices(rotations):
    # Upstream's 6D representation stores the first two ROWS of the matrix.
    first, second = rotations[..., :3], rotations[..., 3:]
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-12)
    second = second - (first * second).sum(axis=-1, keepdims=True) * first
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-12)
    return np.stack((first, second, np.cross(first, second)), axis=-2)

"""Animated, visible bone geometry for ComfyUI's native 3D viewer."""
from io import BytesIO

import numpy as np
from scipy.spatial.transform import Rotation
from comfy_api.latest import Types
from comfy_extras.sam3d_body.export.glb_shared import GLBWriter, quat_sign_fix_per_joint

from .motion import rotation_matrices
from .metadata import embed_glb_metadata


def skeleton_glb(motion, metadata=None):
    ref = motion.reference
    matrices = rotation_matrices(motion.rotations)
    quats = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_quat().reshape(len(matrices), len(ref.names), 4)
    quats = quat_sign_fix_per_joint(quats)
    writer = GLBWriter()
    nodes = [{"name": name, "translation": offset.tolist(), "rotation": quats[0, j].tolist()}
             for j, (name, offset) in enumerate(zip(ref.names, ref.offsets))]
    meshes = []
    faces = np.array([[0, 2, 4], [0, 5, 2], [0, 3, 5], [0, 4, 3],
                      [1, 4, 2], [1, 2, 5], [1, 5, 3], [1, 3, 4]])
    for j, parent in enumerate(ref.parents[1:], 1):
        nodes[parent].setdefault("children", []).append(j)
        offset = ref.offsets[j]
        length = np.linalg.norm(offset)
        if length < 1e-8:
            continue
        direction = offset / length
        axis = np.eye(3)[np.argmin(np.abs(direction))]
        side = np.cross(direction, axis)
        side = side / np.linalg.norm(side) * length * 0.07
        other = np.cross(direction, side)
        middle = offset * 0.2
        vertices = np.array([np.zeros(3), offset, middle + side, middle - side, middle + other, middle - other])
        triangles = vertices[faces].reshape(-1, 3)
        normal = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
        normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
        meshes.append({"primitives": [{"attributes": {
            "POSITION": writer.add_vec3_f32(triangles),
            "NORMAL": writer.add_vec3_f32(np.repeat(normal, 3, axis=0))}, "material": 0}]})
        nodes[parent]["children"].append(len(nodes))
        nodes.append({"name": f"Bone {j}", "mesh": len(meshes) - 1})
    times = writer.add_scalar_f32(np.arange(len(matrices), dtype=np.float32) / motion.fps)
    animation = {"name": "MoCapAnything", "samplers": [], "channels": []}
    for j in range(len(ref.names)):
        animation["samplers"].append({"input": times, "output": writer.add_vec4_f32(quats[:, j]), "interpolation": "LINEAR"})
        animation["channels"].append({"sampler": j, "target": {"node": j, "path": "rotation"}})
    gltf = {"asset": {"version": "2.0", "generator": "ComfyUI-MoCapAnything"},
            "scene": 0, "scenes": [{"nodes": [0]}], "nodes": nodes, "meshes": meshes,
            "materials": [{"doubleSided": True, "pbrMetallicRoughness": {
                "baseColorFactor": [0.12, 0.65, 0.95, 1], "metallicFactor": 0, "roughnessFactor": 0.7}}],
            "animations": [animation]}
    return Types.File3D(BytesIO(embed_glb_metadata(writer.to_bytes(gltf), metadata)), "glb")

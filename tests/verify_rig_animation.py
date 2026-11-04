"""Blender reimport: independently verify exported skin deformation against FK."""
from pathlib import Path
import json
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector


directory = Path(sys.argv[sys.argv.index('--') + 1]).resolve()
for suffix in ('glb', 'gltf', 'fbx'):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    source = str(directory / f'character.{suffix}')
    if suffix == 'fbx':
        bpy.ops.import_scene.fbx(filepath=source, use_anim=False)
    else:
        bpy.ops.import_scene.gltf(filepath=source)
    data = json.loads((directory / f'{suffix}_metadata.json').read_text())
    basis = Matrix(data['basis'])
    rest = [Matrix(value) for value in data['rest']]
    source_vertices = []
    source_joints = []
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH' or not any(mod.type == 'ARMATURE' for mod in obj.modifiers):
            continue
        for vertex in obj.data.vertices:
            source_vertices.append(basis @ obj.matrix_world @ vertex.co)
            group = max(vertex.groups, key=lambda group: group.weight)
            source_joints.append(data['bone_names'].index(obj.vertex_groups[group.group].name))
    expected = []
    for degrees in (0, 1):
        global_rot = [Matrix.Rotation(np.deg2rad(30 * degrees), 3, 'Z'), Matrix.Rotation(np.deg2rad(80 * degrees), 3, 'Z')]
        positions = [rest[0].translation, rest[0].translation + global_rot[0] @ Vector(data['offsets'][1])]
        expected.append(np.array([global_rot[j] @ (v - rest[j].translation) + positions[j] for v, j in zip(source_vertices, source_joints)]))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(directory / f'{suffix}_animated.glb'))
    # glTF times are seconds; use 30fps to select exactly the two original frames.
    bpy.context.scene.render.fps = 30
    bpy.context.scene.render.fps_base = 1
    # Importer used the scene default 24fps at import; last key is at 0.8 frames.
    for frame, target in zip((0, 0.8), expected):
        bpy.context.scene.frame_set(int(frame), subframe=frame % 1)
        deps = bpy.context.evaluated_depsgraph_get()
        vertices = []
        for obj in bpy.context.scene.objects:
            if obj.type != 'MESH' or not any(mod.type == 'ARMATURE' for mod in obj.modifiers):
                continue
            evaluated = obj.evaluated_get(deps)
            mesh = evaluated.to_mesh()
            vertices.extend(basis @ evaluated.matrix_world @ v.co for v in mesh.vertices)
            evaluated.to_mesh_clear()
        vertices = np.array(vertices)
        distances = np.linalg.norm(vertices[:, None] - target[None], axis=-1)
        error = max(distances.min(axis=0).max(), distances.min(axis=1).max())
        assert error < 0.0002, (suffix, frame, error)
        print(f'SKIN_DEFORMATION_PASSED {suffix} frame={frame} error={error:.8f}')

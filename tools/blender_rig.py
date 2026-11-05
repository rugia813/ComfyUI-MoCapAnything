"""Invoked only in an isolated, background Blender process by rig.py."""
from pathlib import Path
import json
import math
import re
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector


def prepare(directory, config):
    source = Path(config['source'])
    if source.suffix.lower() == '.fbx':
        bpy.ops.import_scene.fbx(filepath=str(source), use_anim=True)
    else:
        bpy.ops.import_scene.gltf(filepath=str(source))
    arms = [obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE' and
            any(mesh.type == 'MESH' and any(mod.type == 'ARMATURE' and mod.object == obj for mod in mesh.modifiers)
                for mesh in bpy.context.scene.objects)]
    if not arms:
        raise ValueError(
            f"No skeleton was imported from {source.name!r}. Prepare Rig Reference requires an already-rigged "
            "model with bones and mesh skin weights; it does not auto-rig a static mesh. "
            "Rig and skin the model first, or re-export it with its existing armature and skinning included. "
            "Being visible in Load 3D (Advanced) does not mean a model contains a rig."
        )
    if len(arms) != 1:
        raise ValueError(f"Expected one skinned skeleton; found {len(arms)}. Export a single rigged character.")
    arm = arms[0]
    linear = np.array(arm.matrix_world.to_3x3())
    metric = linear.T @ linear
    if not np.allclose(metric, np.eye(3) * np.trace(metric) / 3, rtol=1e-5, atol=1e-7) or np.linalg.det(linear) <= 0:
        raise ValueError('Apply nonuniform/negative armature and parent scales before exporting the rig.')
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH' and
              any(mod.type == 'ARMATURE' and mod.object == arm for mod in obj.modifiers)]
    if not meshes:
        raise ValueError('No mesh with skin weights bound to the skeleton was found.')
    roots = [bone for bone in arm.data.bones if bone.parent is None]
    if len(roots) != 1:
        raise ValueError('The skeleton must have exactly one root bone. Parent disconnected bone chains before exporting.')
    ordered = []

    def visit(bone):
        ordered.append(bone)
        for child in bone.children:
            visit(child)

    visit(roots[0])
    index = {bone.name: j for j, bone in enumerate(ordered)}
    parents = [-1 if bone.parent is None else index[bone.parent.name] for bone in ordered]
    names = [re.sub(r'[\s{}]', '_', bone.name) for bone in ordered]
    if len(set(names)) != len(names):
        raise ValueError('Bone names collide after removing whitespace/braces. Rename these bones before importing.')
    yaw = {'+Z': 0, '+X': -math.pi / 2, '-Z': math.pi, '-X': math.pi / 2}[config['forward_axis']]
    basis = Matrix.Rotation(yaw, 4, 'Y') @ Matrix.Rotation(-math.pi / 2, 4, 'X')
    rest = [basis @ arm.matrix_world @ bone.matrix_local for bone in ordered]
    origin = rest[0].translation.copy()
    positions = [matrix.translation - origin for matrix in rest]
    offsets = [Vector((0, 0, 0)) if j == 0 else positions[j] - positions[parents[j]] for j in range(len(ordered))]
    metadata = {'names': names, 'bone_names': [bone.name for bone in ordered], 'parents': parents,
                'positions': [list(v) for v in positions], 'offsets': [list(v) for v in offsets],
                'basis': [list(row) for row in basis], 'armature': arm.name,
                'rest': [[list(row) for row in matrix] for matrix in rest]}
    (directory / 'rig.json').write_text(json.dumps(metadata), encoding='utf8')

    motion_data = arm.animation_data
    action = motion_data.action if motion_data else None
    strips = [strip for track in motion_data.nla_tracks for strip in track.strips] if motion_data else []
    if action:
        first, last = int(math.ceil(action.frame_range[0])), int(math.floor(action.frame_range[1]))
    elif strips:
        first, last = int(min(s.frame_start for s in strips)), int(max(s.frame_end for s in strips))
    else:
        first = last = 1
        for bone in arm.pose.bones:
            bone.matrix_basis = Matrix.Identity(4)
    all_positions, all_rotations = [], []
    for frame in range(first, last + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        posed = [basis @ arm.matrix_world @ arm.pose.bones[bone.name].matrix for bone in ordered]
        root = posed[0].translation
        all_positions.append([list(matrix.translation - root) for matrix in posed])
        global_delta = [(matrix.to_3x3() @ rest[j].to_3x3().inverted()).normalized()
                        for j, matrix in enumerate(posed)]
        local_delta = [delta if parents[j] < 0 else global_delta[parents[j]].inverted() @ delta
                       for j, delta in enumerate(global_delta)]
        all_rotations.append([[*matrix[0], *matrix[1]] for matrix in local_delta])
    all_positions = np.asarray(all_positions, dtype=np.float32)
    all_rotations = np.asarray(all_rotations, dtype=np.float32)
    has_motion = bool(last > first and (np.max(np.abs(all_positions - all_positions[:1])) > 1e-6 or
                                        np.max(np.abs(all_rotations - all_rotations[:1])) > 1e-6))
    np.savez(directory / 'motion_reference.npz', positions=all_positions, rot6d=all_rotations)
    metadata['has_motion'] = has_motion
    (directory / 'rig.json').write_text(json.dumps(metadata), encoding='utf8')
    bpy.context.scene.frame_set(first)
    # Render the reference pose for DINOv2.
    points = [basis @ obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
    low = Vector(tuple(min(p[a] for p in points) for a in range(3)))
    high = Vector(tuple(max(p[a] for p in points) for a in range(3)))
    center = (low + high) / 2
    extent = max(high - low)
    camera_data = bpy.data.cameras.new('Reference Camera')
    camera = bpy.data.objects.new('Reference Camera', camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = basis.inverted() @ (center + Vector((0, extent * 0.05, extent * 3)))
    camera.rotation_euler = ((basis.inverted() @ center) - camera.location).to_track_quat('-Z', 'Y').to_euler()
    camera_data.type = 'ORTHO'
    camera_data.ortho_scale = extent * 1.25
    camera_data.clip_end = max(extent * 20, 100)
    camera_data.clip_start = max(extent * 0.001, 0.00001)
    scene = bpy.context.scene
    scene.camera = camera
    scene.world = bpy.data.worlds.new('Reference World')
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.8, 0.8, 0.8, 1)
    for i, direction in enumerate([(1, 2, 2), (-1, 1, 1)]):
        light_data = bpy.data.lights.new(f'Reference Light {i}', 'AREA')
        light_data.energy = 150 * extent * extent
        light_data.shape = 'DISK'
        light_data.size = extent * 2
        light = bpy.data.objects.new(light_data.name, light_data)
        bpy.context.collection.objects.link(light)
        light.location = basis.inverted() @ (center + Vector(direction) * extent)
        light.rotation_euler = ((basis.inverted() @ center) - light.location).to_track_quat('-Z', 'Y').to_euler()
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 16
    scene.cycles.device = 'CPU'
    scene.render.resolution_x = scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.filepath = str(directory / 'reference.png')
    bpy.ops.render.render(write_still=True)
    for obj in list(bpy.context.scene.objects):
        if obj.type in {'CAMERA', 'LIGHT'} or (obj.type == 'MESH' and obj not in meshes) or (obj.type == 'ARMATURE' and obj is not arm):
            bpy.data.objects.remove(obj, do_unlink=True)
            continue
        obj.animation_data_clear()
        if obj.type == 'MESH' and obj.data.shape_keys is not None:
            obj.data.shape_keys.animation_data_clear()
        for constraint in list(obj.constraints):
            obj.constraints.remove(constraint)
    for bone in arm.pose.bones:
        for constraint in list(bone.constraints):
            bone.constraints.remove(constraint)
        bone.matrix_basis = Matrix.Identity(4)
    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(directory / 'rig.blend'))


def animate(directory, config):
    bpy.ops.wm.open_mainfile(filepath=str(directory / 'rig.blend'), load_ui=False, use_scripts=False)
    data = json.loads((directory / 'rig.json').read_text(encoding='utf8'))
    rotations = np.load(directory / 'rotations.npy', allow_pickle=False)
    arm = bpy.data.objects[data['armature']]
    bones = [arm.pose.bones[name] for name in data['bone_names']]
    rest = [Matrix(rows) for rows in data['rest']]
    offsets = [Vector(value) for value in data['offsets']]
    to_arm = arm.matrix_world.inverted() @ Matrix(data['basis']).inverted()
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, len(rotations)
    scene.render.fps = max(1, round(config['fps']))
    scene.render.fps_base = scene.render.fps / config['fps']
    for frame, local_rotations in enumerate(rotations, 1):
        global_rotations, positions, targets = [], [], []
        for j, bone in enumerate(bones):
            parent = data['parents'][j]
            local = Matrix(local_rotations[j].tolist())
            global_rotations.append(local if parent < 0 else global_rotations[parent] @ local)
            positions.append(rest[0].translation.copy() if parent < 0 else positions[parent] + global_rotations[parent] @ offsets[j])
            world = (global_rotations[j] @ rest[j].to_3x3()).to_4x4()
            world.translation = positions[j]
            target = to_arm @ world
            targets.append(target)
            kwargs = {} if parent < 0 else {'parent_matrix': targets[parent], 'parent_matrix_local': bone.parent.bone.matrix_local}
            bone.rotation_mode = 'QUATERNION'
            bone.matrix_basis = bone.bone.convert_local_to_pose(target, bone.bone.matrix_local, invert=True, **kwargs)
            for channel in ('location', 'rotation_quaternion', 'scale'):
                bone.keyframe_insert(channel, frame=frame, group=bone.name)
    scene.frame_set(1)
    # Export only the character. Lights, reference cameras and source animations are absent.
    bpy.ops.object.select_all(action='DESELECT')
    arm.select_set(True)
    for obj in scene.objects:
        if obj.type == 'MESH' and any(mod.type == 'ARMATURE' and mod.object == arm for mod in obj.modifiers):
            obj.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.export_scene.gltf(filepath=str(directory / 'animated.glb'), export_format='GLB', use_selection=True,
                              export_animations=True, export_frame_range=True, export_force_sampling=True,
                              export_animation_mode='SCENE', export_anim_slide_to_zero=True)


if __name__ == '__main__':
    if bpy.app.version < (4, 2, 0):
        raise RuntimeError('Custom rig preparation requires Blender 4.2 or newer.')
    job = Path(sys.argv[sys.argv.index('--') + 1]).resolve()
    config = json.loads(job.read_text(encoding='utf8'))
    if config['mode'] == 'prepare':
        bpy.ops.wm.read_factory_settings(use_empty=True)
        prepare(job.parent, config)
    else:
        animate(job.parent, config)

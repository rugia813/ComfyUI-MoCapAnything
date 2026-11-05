"""Create an original skinned test character with Blender, never a downloaded asset."""
from pathlib import Path
import sys

import bpy
from mathutils import Matrix


out = Path(sys.argv[sys.argv.index('--') + 1]).resolve()
out.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.object.armature_add()
arm = bpy.context.object
arm.name = 'TestCharacter'
bpy.ops.object.mode_set(mode='EDIT')
root = arm.data.edit_bones[0]
root.name = 'Hips'
root.head, root.tail = (0, 0, 0), (0, 0, 1)
tip = arm.data.edit_bones.new('Head')
tip.head, tip.tail = (0, 0, 1), (0, 0, 2)
tip.parent = root
tip.use_connect = True
tip.roll = 0.4  # Exercise nontrivial bind axes, not only identity transforms.
bpy.ops.object.mode_set(mode='OBJECT')
for z, bone, color in [(0.5, 'Hips', (0.1, 0.4, 0.8, 1)), (1.5, 'Head', (0.9, 0.4, 0.1, 1))]:
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, z))
    mesh = bpy.context.object
    mesh.scale = (0.3, 0.22, 0.48)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    group = mesh.vertex_groups.new(name=bone)
    group.add(list(range(len(mesh.data.vertices))), 1.0, 'REPLACE')
    mod = mesh.modifiers.new('Skin', 'ARMATURE')
    mod.object = arm
    mesh.parent = arm
    mat = bpy.data.materials.new(bone)
    mat.diffuse_color = color
    mesh.data.materials.append(mat)
# Nonidentity object transform must survive import, rest rendering and animation.
arm.matrix_world = Matrix.Translation((1, -0.5, 0.2)) @ Matrix.Rotation(0.2, 4, 'Z')
bpy.ops.export_scene.gltf(filepath=str(out / 'character.glb'), export_format='GLB', export_animations=False)
bpy.ops.export_scene.gltf(filepath=str(out / 'character.gltf'), export_format='GLTF_SEPARATE', export_animations=False)
bpy.ops.export_scene.fbx(filepath=str(out / 'character.fbx'), add_leaf_bones=False, bake_anim=False)
head = arm.pose.bones['Head']
head.rotation_mode = 'QUATERNION'
head.rotation_quaternion = (1, 0, 0, 0)
head.keyframe_insert('rotation_quaternion', frame=1)
head.rotation_quaternion = (0.9238795, 0, 0, 0.3826834)
head.keyframe_insert('rotation_quaternion', frame=10)
bpy.context.scene.frame_start, bpy.context.scene.frame_end = 1, 10
bpy.ops.export_scene.fbx(filepath=str(out / 'animated_character.fbx'), add_leaf_bones=False, bake_anim=True)
bpy.ops.export_scene.gltf(filepath=str(out / 'animated_character.glb'), export_format='GLB', export_animations=True)
print('RIG_FIXTURES_CREATED')

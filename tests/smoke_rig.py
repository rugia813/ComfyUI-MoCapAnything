"""Real Blender + encoders + motion model integration; run after create_rig_fixture.py."""
import json
from pathlib import Path
import sys

import bootstrap
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import comfy.clip_vision
import comfy.sd
from comfy_api.latest import Types

from mocap_extension.rig import prepare_rig, animate_rig
from mocap_extension.preview import skeleton_glb
from mocap_extension.motion import Motion
from mocap_extension.pipeline import load_model, capture
from test_preview import read_glb, accessor


torch.set_grad_enabled(False)
artifacts = bootstrap.ROOT / '.reference/artifacts'
out = bootstrap.ROOT / 'test_outputs/rig_fixtures'
clip = comfy.sd.load_clip([str(artifacts / 't5-base.safetensors')], clip_type=comfy.sd.CLIPType.STABLE_AUDIO,
                          model_options={'dtype': torch.float32})
vision = comfy.clip_vision.load(str(artifacts / 'dinov2-large.safetensors'))
model = load_model(str(artifacts / 'mocapanything_v2.safetensors'))
for suffix in ('glb', 'gltf', 'fbx'):
    ref, rig, image = prepare_rig(Types.File3D(str(out / f'character.{suffix}')), clip, vision, '+Z')
    assert ref.names == ['Hips', 'Head'], ref.names
    assert ref.tensors['joint_t5embed'].shape == (2, 768)
    assert ref.tensors['ref_image_embed'].shape == (257, 1024)
    from PIL import Image
    Image.fromarray((image[0].numpy() * 255).astype(np.uint8)).save(out / f'{suffix}_reference.png')
    rest = np.broadcast_to(np.eye(3), (2, 2, 3, 3)).copy()
    rest[1] = Rotation.from_euler('Z', [[30], [50]], degrees=True).as_matrix()
    motion = Motion(ref, np.zeros((2, 2, 3)), rest[:, :, :2].reshape(2, 2, 6), 30)
    animated = animate_rig(motion, rig)
    animated.save_to(str(out / f'{suffix}_animated.glb'))
    (out / f'{suffix}_metadata.json').write_text(json.dumps(rig.metadata), encoding='utf8')
    doc, binary = read_glb(animated.get_bytes())
    assert doc['skins'] and doc['meshes'] and doc['animations']
    assert max(accessor(doc, binary, sampler['input']).max() for sampler in doc['animations'][0]['samplers']) > 0
    actual = capture(model, vision, ref, image.repeat(3, 1, 1, 1), 30, 3, 64)
    assert actual.rotations.shape == (3, 2, 6) and np.isfinite(actual.rotations).all()
    skeleton_glb(actual).save_to(str(out / f'{suffix}_captured_skeleton.glb'))
    animate_rig(actual, rig).save_to(str(out / f'{suffix}_captured_mesh.glb'))
    print(f'RIG_END_TO_END_PASSED {suffix}', flush=True)

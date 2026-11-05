import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import bootstrap
import numpy as np
from scipy.spatial.transform import Rotation
from comfy_extras.nodes_load_3d import Preview3DAdvanced

from mocap_extension.motion import Motion, load_reference
from mocap_extension.preview import skeleton_glb
from mocap_extension.rig import topology, clean_joint_name, animate_rig, PreparedRig
from test_nodes import reference_fixture


def read_glb(blob):
    magic, version, length = struct.unpack_from('<4sII', blob)
    assert magic == b'glTF' and version == 2 and length == len(blob)
    count, kind = struct.unpack_from('<II', blob, 12)
    assert kind == 0x4E4F534A
    doc = json.loads(blob[20:20+count])
    size, kind = struct.unpack_from('<II', blob, 20+count)
    assert kind == 0x004E4942
    return doc, blob[28+count:28+count+size]


def accessor(doc, buffer, index):
    info = doc['accessors'][index]
    view = doc['bufferViews'][info['bufferView']]
    width = {'SCALAR': 1, 'VEC3': 3, 'VEC4': 4}[info['type']]
    return np.frombuffer(buffer, dtype='<f4', count=info['count']*width,
                         offset=view.get('byteOffset', 0) + info.get('byteOffset', 0)).reshape(-1, width)


class PreviewTests(unittest.TestCase):
    def test_glb_animation_matches_bvh_rotations_and_native_viewer_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'ref.npz'
            reference_fixture(path)
            ref = load_reference(path)
            matrices = Rotation.from_euler('Z', [[0], [90], [180]], degrees=True).as_matrix()
            rot = np.repeat(matrices[:, None, :2, :].reshape(3, 1, 6), 2, axis=1)
            motion = Motion(ref, np.zeros((3, 2, 3)), rot, 23.976)
            metadata = {"prompt": {"6": {"class_type": "MoCapAnything_Capture"}},
                        "workflow": {"nodes": [{"id": 6}]}}
            model = skeleton_glb(motion, metadata)
            doc, data = read_glb(model.get_bytes())
            self.assertEqual(doc["asset"]["extras"]["comfyui"], metadata)
            self.assertEqual(doc['nodes'][1]['translation'], [0, 2, 0])
            for sampler in doc['animations'][0]['samplers']:
                np.testing.assert_allclose(accessor(doc, data, sampler['input']).ravel(), np.arange(3)/motion.fps, atol=1e-7)
                quats = accessor(doc, data, sampler['output'])
                np.testing.assert_allclose(Rotation.from_quat(quats).as_matrix(), matrices, atol=1e-6)
            self.assertTrue(doc['meshes'])
            with patch('folder_paths.get_temp_directory', return_value=temp):
                preview = Preview3DAdvanced.execute(model, {}, 512, 512)
            self.assertEqual(preview.ui.as_dict()['3d'][0]['type'], 'temp')

    def test_topology_and_names(self):
        hop, edge = topology([-1, 0, 0, 1])
        np.testing.assert_array_equal(hop, [[0, 1, 1, 2], [1, 0, 2, 1], [1, 2, 0, 3], [2, 1, 3, 0]])
        self.assertEqual(edge[1, 2], 3)
        self.assertEqual(edge[3, 3], 5)
        self.assertEqual(edge[1, 3], 2)
        self.assertEqual(edge[3, 1], 1)
        self.assertEqual(clean_joint_name('mixamorig:LeftUpLeg'), 'Left Up Leg')

    def test_reject_cross_rig_binding(self):
        rig = PreparedRig(b'', {}, object(), 'unused')
        with self.assertRaisesRegex(ValueError, 'same prepared rig'):
            animate_rig(Motion(None, None, None, 30), rig)


if __name__ == '__main__':
    unittest.main()

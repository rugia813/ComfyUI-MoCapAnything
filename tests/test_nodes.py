import asyncio
import importlib.util
import json
import io as string_io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import bootstrap
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from mocap_extension.motion import load_reference, Motion, rotation_matrices, write_bvh, read_bvh_skeleton
from mocap_extension.nodes import LoadMoCapAnything, LoadMoCapReference, CaptureMoCapAnything, SaveMoCapMotion, reference_path
from mocap_extension.pipeline import prepare_frames, make_batch


def reference_fixture(path):
    np.savez(path, names=np.array(["root", "rig:tip"]), parents=np.array([-1, 0]),
             offsets=np.array([[0, 0, 0], [0, 2, 0]], dtype=np.float32), global_scale=2.0,
             ref_position=np.array([[0, 0, 0], [0, 1, 0]], dtype=np.float32),
             ref_rot6d_a=np.tile([1, 0, 0, 0, 1, 0], (2, 1)).astype(np.float32),
             ref_image_embed=np.zeros((257, 1024), dtype=np.float32),
             joint_t5embed=np.zeros((2, 768), dtype=np.float32),
             graph_hop=np.array([[0, 1], [1, 0]]), graph_edge=np.array([[0, 1], [2, 0]]),
             static_pos_joint_mask=np.array([True, False]), static_rot_joint_mask=np.array([False, True]))


class NodeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.input = self.root / "input"
        self.output = self.root / "output"
        (self.input / "mocapanything").mkdir(parents=True)
        self.output.mkdir()
        self.path = self.input / "mocapanything/test.npz"
        reference_fixture(self.path)
        self.ref = load_reference(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_v3_registration_and_schemas(self):
        spec = importlib.util.spec_from_file_location("mocap_extension", bootstrap.ROOT / "__init__.py", submodule_search_locations=[str(bootstrap.ROOT)])
        extension = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extension)
        entry = asyncio.run(extension.comfy_entrypoint())
        classes = asyncio.run(entry.get_node_list())
        self.assertEqual(len(classes), 4)
        for cls in classes:
            schema = cls.define_schema()
            self.assertTrue(schema.node_id.startswith("MoCapAnything_"))
            self.assertIsInstance(cls.INPUT_TYPES(), dict)

    def test_frontend_workflow_matches_api_links(self):
        workflow = json.loads((bootstrap.ROOT / "example_workflows/video_to_bvh.json").read_text())
        api = json.loads((bootstrap.ROOT / "example_workflows/video_to_bvh.api.json").read_text())
        nodes = {node["id"]: node for node in workflow["nodes"]}
        for link_id, source, output, destination, input_slot, socket_type in workflow["links"]:
            input = nodes[destination]["inputs"][input_slot]
            self.assertEqual(input["link"], link_id)
            self.assertEqual(api[str(destination)]["inputs"][input["name"]], [str(source), output])
            self.assertEqual(nodes[source]["outputs"][output]["type"], socket_type)

    def test_reference_containment_and_cache(self):
        with patch("folder_paths.get_input_directory", return_value=str(self.input)):
            self.assertEqual(reference_path("test.npz"), self.path)
            for name in ("../escape.npz", "../../outside.npz", str(self.root / "escape.npz"), "test.pkl"):
                with self.assertRaises(ValueError):
                    reference_path(name)
            fingerprint = LoadMoCapReference.fingerprint_inputs("test.npz")
            with self.path.open("ab") as file:
                file.write(b"changed")
            self.assertNotEqual(fingerprint, LoadMoCapReference.fingerprint_inputs("test.npz"))

    def test_reject_pickled_reference(self):
        np.savez(self.path, names=np.array([{"unsafe": True}], dtype=object))
        with self.assertRaises(ValueError):
            load_reference(self.path)

    def test_reject_cycle_before_building_ancestor_mask(self):
        with np.load(self.path) as data:
            values = dict(data)
        values["parents"] = np.array([-1, 1])
        np.savez(self.path, **values)
        with self.assertRaisesRegex(ValueError, "parents"):
            load_reference(self.path)

    def test_batch_units_and_shapes(self):
        batch = make_batch(self.ref, torch.zeros(3, 257, 1024), "cpu")
        self.assertEqual(batch["image_embed"].shape, (1, 3, 257, 1024))
        self.assertEqual(batch["offset_a"][0, 1, 1], 1)
        np.testing.assert_equal(batch["ancestor_mask"].numpy(), [[[True, False], [True, True]]])
        self.assertEqual(batch["graph_hop"].dtype, torch.int64)

    def test_foreground_mask_crop_and_broadcast(self):
        images = torch.zeros(2, 20, 30, 3)
        masks = torch.zeros(1, 20, 30)
        masks[:, 5:15, 10:20] = 1
        frames = list(prepare_frames(images, masks))
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0].size, (12, 12))
        self.assertEqual(frames[0].getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(frames[0].getpixel((6, 6)), (0, 0, 0))
        self.assertEqual(len(list(prepare_frames(images, torch.zeros_like(masks)))), 2)

    def test_bvh_rotation_convention_and_fps(self):
        matrices = Rotation.from_euler("ZYX", [[30, 20, 10], [-40, 15, 60]], degrees=True).as_matrix()
        rotations = matrices[:, :2].reshape(1, 2, 6)
        np.testing.assert_allclose(rotation_matrices(rotations), matrices[None], atol=1e-8)
        motion = Motion(self.ref, np.zeros((1, 2, 3)), rotations, 23.976)
        buffer = string_io.StringIO()
        write_bvh(buffer, motion)
        lines = buffer.getvalue().splitlines()
        self.assertIn("Frames: 1", lines)
        time_line = next(line for line in lines if line.startswith("Frame Time:"))
        self.assertAlmostEqual(float(time_line.split(":")[1]), 1/23.976, places=9)
        angles = np.array([float(v) for v in lines[-1].split()][3:]).reshape(2, 3)
        np.testing.assert_allclose(Rotation.from_euler("ZYX", angles, degrees=True).as_matrix(), matrices, atol=1e-8)
        path = self.root / "test.bvh"
        path.write_text(buffer.getvalue())
        names, parents, offsets = read_bvh_skeleton(path)
        self.assertEqual(names, self.ref.names)
        np.testing.assert_equal(parents, self.ref.parents)
        np.testing.assert_equal(offsets, self.ref.offsets)

    def test_output_containment_dotted_prefix_and_collision(self):
        motion = Motion(self.ref, np.zeros((2, 2, 3)), np.tile(self.ref.tensors["ref_rot6d_a"], (2, 1, 1)), 30)
        with patch("folder_paths.get_output_directory", return_value=str(self.output)):
            with self.assertRaises(ValueError):
                SaveMoCapMotion.execute(motion, "../outside/motion")
            self.assertFalse((self.root / "outside").exists())
            first = SaveMoCapMotion.execute(motion, "nested/my.motion").result
            second = SaveMoCapMotion.execute(motion, "nested/my.motion").result
            self.assertNotEqual(first, second)
            for result in (first, second):
                self.assertTrue(Path(result[0]).is_file())
                with np.load(result[1], allow_pickle=False) as data:
                    self.assertEqual(data["rot6d"].shape, (2, 2, 6))
                    self.assertEqual(data["fps"], 30)


if __name__ == "__main__":
    unittest.main()

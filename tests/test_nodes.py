import asyncio
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import bootstrap
import numpy as np
import torch
from safetensors.torch import load_file
from scipy.spatial.transform import Rotation

from mocap_extension.motion import load_reference, rotation_matrices
from mocap_extension.nodes import LoadMoCapAnything
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
        self.assertEqual(len(classes), 5)
        for cls in classes:
            schema = cls.define_schema()
            self.assertTrue(schema.node_id.startswith("MoCapAnything_"))
            self.assertIsInstance(cls.INPUT_TYPES(), dict)

    def test_frontend_workflow_links(self):
        for path in (bootstrap.ROOT / 'example_workflows').glob('*.json'):
            workflow = json.loads(path.read_text(encoding="utf8"))
            nodes = {node["id"]: node for node in workflow["nodes"]}
            for link_id, source, output, destination, input_slot, socket_type in workflow["links"]:
                input = nodes[destination]["inputs"][input_slot]
                self.assertEqual(input["link"], link_id)
                self.assertIn(link_id, nodes[source]["outputs"][output]["links"])
                self.assertEqual(nodes[source]["outputs"][output]["type"], socket_type)

    def test_model_loader_converts_original_checkpoint_once(self):
        models = self.root / "models"
        models.mkdir()
        source = models / "video2pos2rot_epoch60.pt"
        torch.save({"model_state": {"module.video2pos.weight": torch.ones(1),
                                    "module.pos2rot.weight": torch.ones(1)}}, source)
        converted = source.with_suffix(".safetensors")
        with patch("folder_paths.get_full_path_or_raise", return_value=str(source)), \
             patch("folder_paths.get_folder_paths", return_value=[str(models)]), \
             patch("mocap_extension.nodes.load_model", return_value="loaded") as load:
            self.assertEqual(LoadMoCapAnything.execute(source.name).result[0], "loaded")
            self.assertEqual(load.call_args.args[0], str(converted))
            self.assertEqual(set(load_file(str(converted))), {"video2pos.weight", "pos2rot.weight"})
            with patch("mocap_extension.nodes.convert_checkpoint", side_effect=AssertionError("converted twice")):
                LoadMoCapAnything.execute(source.name)

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

    def test_foreground_mask_crop_and_unmasked_frames(self):
        images = torch.zeros(2, 20, 30, 3)
        masks = torch.zeros(1, 20, 30)
        masks[:, 5:15, 10:20] = 1
        frames = list(prepare_frames(images, masks))
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0].size, (12, 12))
        self.assertEqual(frames[0].getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(frames[0].getpixel((6, 6)), (0, 0, 0))
        self.assertEqual(len(list(prepare_frames(images, torch.zeros_like(masks)))), 2)
        plain = list(prepare_frames(images))
        self.assertEqual(len(plain), 2)
        self.assertEqual(plain[0].getpixel((15, 15)), (0, 0, 0))

    def test_rotation_convention(self):
        matrices = Rotation.from_euler("ZYX", [[30, 20, 10], [-40, 15, 60]], degrees=True).as_matrix()
        rotations = matrices[:, :2].reshape(1, 2, 6)
        np.testing.assert_allclose(rotation_matrices(rotations), matrices[None], atol=1e-8)


if __name__ == "__main__":
    unittest.main()

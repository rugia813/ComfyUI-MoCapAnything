"""Compare joint animation channels in two GLBs by bone name and sample time."""
import json
import struct
import sys

import numpy as np


def load(path):
    with open(path, "rb") as file:
        header = file.read(12)
        assert header[:4] == b"glTF"
        json_size, _ = struct.unpack("<I4s", file.read(8))
        data = json.loads(file.read(json_size))
        bin_size, _ = struct.unpack("<I4s", file.read(8))
        blob = file.read(bin_size)
    def accessor(index):
        a = data["accessors"][index]
        view = data["bufferViews"][a["bufferView"]]
        assert a["componentType"] == 5126
        shape = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[a["type"]]
        count = a["count"]
        offset = view.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = view.get("byteStride", shape * 4)
        return np.array([np.frombuffer(blob, dtype="<f4", count=shape, offset=offset+i*stride)
                         for i in range(count)])
    channels = {}
    for animation in data.get("animations", []):
        for channel in animation["channels"]:
            sampler = animation["samplers"][channel["sampler"]]
            target = channel["target"]
            key = (data["nodes"][target["node"]].get("name"), target["path"])
            channels[key] = accessor(sampler["input"]), accessor(sampler["output"])
    return data, channels


left_data, left = load(sys.argv[1])
right_data, right = load(sys.argv[2])
print("left", sys.argv[1], "right", sys.argv[2])
print("skins", len(left_data.get("skins", [])), len(right_data.get("skins", [])))
print("channels", len(left), len(right), "matching", len(left.keys() & right.keys()))
by_bone = []
all_angles = []
for key in sorted(left.keys() & right.keys()):
    times_a, values_a = left[key]
    times_b, values_b = right[key]
    if len(times_a) != len(times_b) or np.max(np.abs(times_a-times_b)) > 1e-4:
        print("time mismatch", key, len(times_a), len(times_b))
        continue
    if key[1] == "rotation":
        a = values_a / np.linalg.norm(values_a, axis=1, keepdims=True)
        b = values_b / np.linalg.norm(values_b, axis=1, keepdims=True)
        angles = 2 * np.arccos(np.clip(np.abs(np.sum(a*b, axis=1)), 0, 1)) * 180 / np.pi
        by_bone.append((float(np.mean(angles)), float(np.max(angles)), key[0]))
        all_angles.extend(angles.tolist())
    elif key[1] == "translation":
        diffs = np.linalg.norm(values_a-values_b, axis=1)
        if np.max(diffs) > 0.001:
            print("translation difference", key[0], float(np.mean(diffs)), float(np.max(diffs)))
print("rotation mean across bones", float(np.mean([x[0] for x in by_bone])))
print("rotation max", max(by_bone))
print("rotation samples", len(all_angles), "median", float(np.median(all_angles)),
      "p95", float(np.percentile(all_angles, 95)), "max", float(np.max(all_angles)),
      "above 5 deg", int(np.sum(np.array(all_angles) > 5)),
      "above 10 deg", int(np.sum(np.array(all_angles) > 10)))
print("largest bone differences", sorted(by_bone, reverse=True)[:10])

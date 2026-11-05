"""Store ComfyUI execution metadata in this extension's numeric and 3D outputs."""
import json
import struct

from comfy.cli_args import args


def execution_metadata(hidden):
    if args.disable_metadata or hidden is None:
        return {}
    metadata = {}
    if hidden.prompt is not None:
        metadata["prompt"] = hidden.prompt
    if hidden.extra_pnginfo:
        metadata.update(hidden.extra_pnginfo)
    return metadata


def embed_glb_metadata(blob, metadata):
    if not metadata:
        return blob
    if len(blob) < 20 or blob[:4] != b"glTF":
        raise ValueError("Expected a binary glTF file for workflow metadata.")
    magic, version, length = struct.unpack_from("<4sII", blob)
    json_length, json_type = struct.unpack_from("<I4s", blob, 12)
    if version != 2 or length != len(blob) or json_type != b"JSON":
        raise ValueError("Invalid binary glTF header.")
    gltf = json.loads(blob[20:20 + json_length])
    gltf.setdefault("asset", {}).setdefault("extras", {})["comfyui"] = metadata
    encoded = json.dumps(gltf, ensure_ascii=False, separators=(",", ":")).encode("utf8")
    encoded += b" " * (-len(encoded) % 4)
    tail = blob[20 + json_length:]
    return struct.pack("<4sII", magic, version, 20 + len(encoded) + len(tail)) + struct.pack("<I4s", len(encoded), b"JSON") + encoded + tail

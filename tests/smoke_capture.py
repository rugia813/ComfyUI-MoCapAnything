"""Run a local video through the actual node methods and native DINOv2 loader."""
import argparse
from pathlib import Path
import json
import time

import bootstrap
import numpy as np
import torch
import folder_paths
import comfy.clip_vision
from comfy_api.latest import InputImpl

from mocap_extension.nodes import LoadMoCapAnything, CaptureMoCapAnything
from mocap_extension.motion import load_reference


def run(args):
    torch.set_grad_enabled(False)
    args.output.mkdir(parents=True, exist_ok=True)
    folder_paths.add_model_folder_path("diffusion_models", str(args.checkpoint.parent.resolve()))
    components = InputImpl.VideoFromFile(str(args.video)).get_components()
    model = LoadMoCapAnything.execute(args.checkpoint.name).result[0]
    clip = comfy.clip_vision.load(str(args.encoder))
    reference = load_reference(args.reference)
    start = time.perf_counter()
    motion = CaptureMoCapAnything.execute(model, clip, reference, components.images, float(components.frame_rate), 8, 64).result[0]
    elapsed = time.perf_counter() - start
    assert len(motion.positions) == len(components.images)
    assert np.isfinite(motion.positions).all() and np.isfinite(motion.rotations).all()
    report = {"frames": len(motion.positions), "joints": len(reference.names), "fps": motion.fps,
              "capture_seconds": elapsed, "device": str(model.load_device),
              "encoder_dtype": str(clip.dtype), "peak_cuda_mib": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None}
    (args.output / "smoke_result.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2), flush=True)
    print("VIDEO_CAPTURE_PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("test_outputs"))
    run(parser.parse_args())

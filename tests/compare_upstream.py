"""Real-checkpoint numerical regression; arguments are paths to local test assets."""
import argparse
from pathlib import Path
import sys
import time

import bootstrap
import numpy as np
import torch
from safetensors.torch import load_file

from mocap_extension.motion import load_reference
from mocap_extension.pipeline import make_batch
from mocap_extension.model.network import MoCapAnything


def run(args):
    sys.path.insert(0, str(args.upstream))
    from models.v2.video2pose.model import RefGuidedVideo2PoseModel
    from models.v2.pose2rot.model import Pose2RotMemoryRestModel

    torch.set_grad_enabled(False)
    state = load_file(str(args.checkpoint))
    upstream_pose = RefGuidedVideo2PoseModel(num_layers=8, ref_layers=4, use_joint_embed=True, use_graph_ref_inner=True, use_graph_temporal_inner=True).eval().to(args.device)
    upstream_rot = Pose2RotMemoryRestModel(rest_layers=4, pose_layers=4, memory_layers=4, decoder_layers=8, decoder_use_cross_layers=6).eval().to(args.device)
    upstream_pose.load_state_dict({key.removeprefix("video2pos."): value for key, value in state.items() if key.startswith("video2pos.")})
    upstream_rot.load_state_dict({key.removeprefix("pos2rot."): value for key, value in state.items() if key.startswith("pos2rot.")})
    model = MoCapAnything().eval().to(args.device)
    model.load_state_dict(state)
    reference = load_reference(args.reference)
    generator = torch.Generator().manual_seed(42)
    reports = []
    for frames in args.frames:
        features = torch.from_numpy(reference.tensors["ref_image_embed"])[None].repeat(frames, 1, 1)
        features += torch.randn(features.shape, generator=generator) * 0.1
        batch = make_batch(reference, features, args.device)
        kwargs = dict(seq_len=frames, selfatt_temporal_layer_flag=[], selfatt_temporal=True,
                      crossatt2_temporal=True, selfatt_slidwindow=5, crossatt2_slidwindow=5)
        if args.device == "cuda":
            torch.cuda.synchronize()
        before = time.perf_counter()
        pose = upstream_pose(batch, attention_kwargs=kwargs)
        rot = upstream_rot(batch, pose_override=pose)["pred_rot6d"]
        if args.device == "cuda":
            torch.cuda.synchronize()
        reference_time = time.perf_counter() - before
        before = time.perf_counter()
        actual_pose, actual_rot = model(batch)
        if args.device == "cuda":
            torch.cuda.synchronize()
        port_time = time.perf_counter() - before
        np.testing.assert_allclose(actual_pose.cpu(), pose.cpu(), atol=5e-5, rtol=5e-4)
        np.testing.assert_allclose(actual_rot.cpu(), rot.cpu(), atol=5e-5, rtol=5e-4)
        report = {"frames": frames, "pose_max_error": (pose-actual_pose).abs().max().item(),
                  "rotation_max_error": (rot-actual_rot).abs().max().item(),
                  "upstream_seconds": reference_time, "port_seconds": port_time}
        if frames > 80:
            pieces = []
            for start in range(0, frames, 32):
                end = min(start + 32, frames)
                left, right = max(0, start - 40), min(frames, end + 40)
                _, piece = model(make_batch(reference, features[left:right], args.device))
                pieces.append(piece[:, start-left:end-left].clone())
            chunked = torch.cat(pieces, dim=1)
            np.testing.assert_allclose(chunked.cpu(), actual_rot.cpu(), atol=5e-5, rtol=5e-4)
            report["chunk_max_error"] = (chunked-actual_rot).abs().max().item()
        reports.append(report)
        print(report, flush=True)
    print("UPSTREAM_PARITY_PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--frames", nargs="+", type=int, default=[1, 8, 97])
    run(parser.parse_args())

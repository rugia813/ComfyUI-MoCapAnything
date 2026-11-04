"""Pack one upstream reference into an NPZ that is safe to load in a workflow."""
import argparse
from pathlib import Path
import pickle
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion import read_bvh_skeleton, load_reference


def convert(base, sequence, destination, frame=0):
    species = sequence.split("/")[0].split("#")[0]
    names, parents, offsets = read_bvh_skeleton(base / "bvh" / f"{sequence}.bvh")
    info = np.load(base / "species_info_dict.npy", allow_pickle=True).item()[species]
    with (base / "cache/__mesh2pose1002_species_scale_cache.pkl").open("rb") as file:
        scale = float(pickle.load(file)[species]["global_scale"])
    with np.load(base / "bvh_pose" / f"{sequence}.npz", allow_pickle=False) as pose:
        position = pose["position"][frame].astype(np.float32)
        rotation = pose["rot6d"][frame].astype(np.float32)
    with np.load(base / "npz_train_image_only" / f"{sequence}.npz", allow_pickle=False) as features:
        image = features["image_embed"][frame].astype(np.float32)
    static_pos, static_rot = np.zeros(len(names), dtype=bool), np.zeros(len(names), dtype=bool)
    static_pos[info["static_joints"]] = True
    static_rot[info["static_rot_joints"]] = True
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, names=np.asarray(names), parents=parents, offsets=offsets, global_scale=scale,
                        ref_position=(position - position[:1]) / scale, ref_rot6d_a=rotation, ref_image_embed=image,
                        graph_hop=info["joints_distance"], graph_edge=info["joint_relation"], joint_t5embed=info["t5_embedding"],
                        static_pos_joint_mask=static_pos, static_rot_joint_mask=static_rot)
    load_reference(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path, help="Upstream dataset root, e.g. demo/data/zoo1030")
    parser.add_argument("sequence", help="Reference sequence, e.g. Hamster#Hamster-RollAttack/y0")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--trust-upstream-pickle", action="store_true", help="Allow reading trusted upstream species/scale pickle metadata")
    args = parser.parse_args()
    if not args.trust_upstream_pickle:
        parser.error("Upstream metadata uses pickle. Inspect its source, then pass --trust-upstream-pickle for trusted local data.")
    convert(args.base, args.sequence, args.destination, args.frame)

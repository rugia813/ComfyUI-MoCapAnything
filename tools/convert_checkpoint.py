"""Convert the released V2 checkpoint without importing or running upstream code."""
import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def convert(source, destination):
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    for key in ("model_state", "model", "model_state_dict", "state_dict"):
        if key in checkpoint:
            checkpoint = checkpoint[key]
            break
    state = {key.removeprefix("module."): value.contiguous() for key, value in checkpoint.items()}
    if not any(key.startswith("video2pos.") for key in state) or not any(key.startswith("pos2rot.") for key in state):
        raise ValueError("Expected the released MoCapAnything V2 video2pos/pos2rot checkpoint.")
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(destination), metadata={"format": "pt", "model": "MoCapAnythingV2"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    convert(args.source, args.destination)

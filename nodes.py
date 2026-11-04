from pathlib import Path
import hashlib
import os
import tempfile

import numpy as np
import folder_paths
from comfy_api.latest import io, ui

from .motion import load_reference, write_bvh
from .pipeline import load_model, capture
from .preview import skeleton_glb
from .rig import prepare_rig, animate_rig
from .tools.convert_checkpoint import convert as convert_checkpoint


MoCapModel = io.Custom("MOCAPANYTHING_MODEL")
MoCapReference = io.Custom("MOCAPANYTHING_REFERENCE")
MoCapMotion = io.Custom("MOCAPANYTHING_MOTION")
MoCapRig = io.Custom("MOCAPANYTHING_RIG")


def reference_path(name):
    root = Path(folder_paths.get_input_directory()) / "mocapanything"
    path = root / name
    if path.suffix.lower() != ".npz" or not folder_paths.is_within_directory(str(root), str(path)):
        raise ValueError("Select a reference NPZ inside ComfyUI/input/mocapanything.")
    if not path.is_file():
        raise FileNotFoundError(f"Reference not found: {path}")
    return path


class LoadMoCapAnything(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MoCapAnything_LoadModel", display_name="Load MoCapAnything V2", category="MoCapAnything",
            description="Load a V2 .pt or .safetensors checkpoint. A .pt file is converted once beside the original.",
            inputs=[io.Combo.Input("model_name", options=[name for name in folder_paths.get_filename_list("diffusion_models") if name.lower().endswith((".pt", ".safetensors"))])],
            outputs=[MoCapModel.Output("model")],
        )

    @classmethod
    def execute(cls, model_name):
        path = cls.model_path(model_name)
        if path.suffix.lower() == ".pt":
            converted = path.with_suffix(".safetensors")
            if not converted.exists() or converted.stat().st_mtime_ns < path.stat().st_mtime_ns:
                fd, temporary = tempfile.mkstemp(suffix=".safetensors", dir=path.parent)
                os.close(fd)
                try:
                    convert_checkpoint(path, temporary)
                    os.replace(temporary, converted)
                finally:
                    Path(temporary).unlink(missing_ok=True)
            path = converted
        return io.NodeOutput(load_model(str(path)))

    @classmethod
    def model_path(cls, model_name):
        path = Path(folder_paths.get_full_path_or_raise("diffusion_models", model_name))
        if path.suffix.lower() not in (".pt", ".safetensors") or not any(
            folder_paths.is_within_directory(root, str(path)) for root in folder_paths.get_folder_paths("diffusion_models")
        ):
            raise ValueError("Select a .pt or .safetensors checkpoint inside a configured diffusion_models folder.")
        return path

    @classmethod
    def fingerprint_inputs(cls, model_name):
        path = cls.model_path(model_name)
        stat = path.stat()
        return (stat.st_size, stat.st_mtime_ns)


class LoadMoCapReference(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        root = Path(folder_paths.get_input_directory()) / "mocapanything"
        return io.Schema(
            node_id="MoCapAnything_LoadReference", display_name="Load MoCapAnything Reference", category="MoCapAnything",
            description="A converted reference skeleton with its matching pose, rotations and DINOv2/joint features.",
            inputs=[io.Combo.Input("reference_name", options=sorted(p.relative_to(root).as_posix() for p in root.rglob("*.npz") if p.is_file()))],
            outputs=[MoCapReference.Output("reference")],
        )

    @classmethod
    def execute(cls, reference_name):
        return io.NodeOutput(load_reference(reference_path(reference_name)))

    @classmethod
    def fingerprint_inputs(cls, reference_name):
        with reference_path(reference_name).open("rb") as file:
            return hashlib.file_digest(file, "sha256").hexdigest()


class CaptureMoCapAnything(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MoCapAnything_Capture", display_name="MoCapAnything Capture", category="MoCapAnything",
            description="Capture root-relative joint motion from an IMAGE frame batch. Use DINOv2-Large in CLIP Vision Loader. Optional foreground masks use white=subject.",
            inputs=[MoCapModel.Input("model"), io.ClipVision.Input("clip_vision"), MoCapReference.Input("reference"),
                    io.Image.Input("images"), io.Float.Input("fps", default=30.0, min=0.01, max=1000.0, step=0.01),
                    io.Int.Input("batch_size", default=8, min=1, max=4096, advanced=True),
                    io.Int.Input("chunk_size", default=64, min=1, max=100000, advanced=True),
                    io.Mask.Input("foreground_mask", optional=True)],
            outputs=[MoCapMotion.Output("motion")],
        )

    @classmethod
    def execute(cls, model, clip_vision, reference, images, fps=30.0, batch_size=8, chunk_size=64, foreground_mask=None):
        return io.NodeOutput(capture(model, clip_vision, reference, images, fps, batch_size, chunk_size, foreground_mask))


class SaveMoCapMotion(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MoCapAnything_SaveMotion", display_name="Save MoCapAnything Motion", category="MoCapAnything",
            inputs=[MoCapMotion.Input("motion"), io.String.Input("filename_prefix", default="mocapanything/motion")],
            outputs=[io.String.Output("bvh_path"), io.String.Output("npz_path")], is_output_node=True,
            description="Save BVH animation and numeric NPZ motion inside ComfyUI/output, preserving frame rate and skeleton units.",
        )

    @classmethod
    def execute(cls, motion, filename_prefix="mocapanything/motion"):
        output = folder_paths.get_output_directory()
        # Check before the native helper creates directories, including symlinks.
        if not folder_paths.is_within_directory(output, os.path.join(output, filename_prefix)):
            raise ValueError("filename_prefix must stay inside ComfyUI/output.")
        directory, filename, counter, _, _ = folder_paths.get_save_image_path(filename_prefix, output)
        while True:
            base = Path(directory) / f"{filename}_{counter:05d}"
            bvh_path, npz_path = Path(str(base) + ".bvh"), Path(str(base) + ".npz")
            if not bvh_path.exists() and not npz_path.exists():
                break
            counter += 1
        # Exclusive creation prevents overwriting existing output files.
        with bvh_path.open("x", encoding="utf8", newline="\n") as file:
            write_bvh(file, motion)
        with npz_path.open("xb") as file:
            np.savez_compressed(file, positions=motion.positions, rot6d=motion.rotations,
                                parents=motion.reference.parents, offsets=motion.reference.offsets,
                                names=np.asarray(motion.reference.names), fps=motion.fps)
        return io.NodeOutput(str(bvh_path), str(npz_path), ui=ui.PreviewText(str(bvh_path)))


class PrepareMoCapRig(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MoCapAnything_PrepareRig", display_name="Prepare MoCapAnything Rig Reference", category="MoCapAnything",
            is_experimental=True,
            description="Build a reference from a rigged FBX/GLB/glTF rest pose. Requires Blender 4.2+, native T5-base CLIP and DINOv2-Large. Single-rest-pose scale/static-joint estimation differs from upstream's animation-dataset preparation.",
            inputs=[io.File3DAny.Input("model_3d"), io.Clip.Input("clip"), io.ClipVision.Input("clip_vision"),
                    io.Combo.Input("forward_axis", options=["+Z", "-Z", "+X", "-X"], default="+Z",
                                   tooltip="Character's facing direction in the native 3D viewer's Y-up coordinates.")],
            outputs=[MoCapReference.Output("reference"), MoCapRig.Output("rig"), io.Image.Output("reference_image")],
        )

    @classmethod
    def execute(cls, model_3d, clip, clip_vision, forward_axis="+Z"):
        return io.NodeOutput(*prepare_rig(model_3d, clip, clip_vision, forward_axis))


class AnimateMoCapRig(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MoCapAnything_AnimateRig", display_name="Animate MoCapAnything Rig", category="MoCapAnything",
            description="Apply captured rotations to the same prepared rig, preserving its mesh and skin weights. Connect model_3d to Preview 3D (Advanced) or Save 3D.",
            inputs=[MoCapMotion.Input("motion"), MoCapRig.Input("rig")],
            outputs=[io.File3DGLB.Output("model_3d")],
        )

    @classmethod
    def execute(cls, motion, rig):
        return io.NodeOutput(animate_rig(motion, rig))


class MoCapSkeleton3D(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MoCapAnything_Skeleton3D", display_name="MoCapAnything Motion to 3D Skeleton", category="MoCapAnything",
            description="Animated GLB bone geometry using the same rotations and offsets as BVH. Connect to Preview 3D (Advanced). No Blender required.",
            inputs=[MoCapMotion.Input("motion")], outputs=[io.File3DGLB.Output("model_3d")],
        )

    @classmethod
    def execute(cls, motion):
        return io.NodeOutput(skeleton_glb(motion))

from pathlib import Path
import os
import tempfile

import folder_paths
from comfy_api.latest import io, ui

from .pipeline import load_model, capture
from .preview import skeleton_glb
from .rig import prepare_rig, animate_rig
from .metadata import execution_metadata
from .tools.convert_checkpoint import convert as convert_checkpoint


MoCapModel = io.Custom("MOCAPANYTHING_MODEL")
MoCapReference = io.Custom("MOCAPANYTHING_REFERENCE")
MoCapMotion = io.Custom("MOCAPANYTHING_MOTION")
MoCapRig = io.Custom("MOCAPANYTHING_RIG")


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


class CaptureMoCapAnything(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MoCapAnything_Capture", display_name="MoCapAnything Capture", category="MoCapAnything",
            description="Capture motion from video frames. Connect a ComfyUI foreground mask when subject isolation is needed.",
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


class PrepareMoCapRig(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MoCapAnything_PrepareRig", display_name="Prepare MoCapAnything Rig Reference", category="MoCapAnything",
            is_experimental=True,
            description="Build a reference from a skinned character. Embedded skeletal animation improves the reference; a static rig uses its rest pose.",
            inputs=[io.File3DAny.Input("model_3d"), io.Clip.Input("clip"), io.ClipVision.Input("clip_vision"),
                    io.Combo.Input("forward_axis", options=["+Z", "-Z", "+X", "-X"], default="+Z",
                                   tooltip="Character's facing direction in the native 3D viewer's Y-up coordinates.")],
            outputs=[MoCapReference.Output("reference"), MoCapRig.Output("rig"), io.Image.Output("reference_image")],
        )

    @classmethod
    def execute(cls, model_3d, clip, clip_vision, forward_axis="+Z"):
        reference, rig, image = prepare_rig(model_3d, clip, clip_vision, forward_axis)
        if not rig.metadata['has_motion']:
            return io.NodeOutput(reference, rig, image, ui=ui.PreviewText(
                'Static rig: using its rest pose as the reference. Embed skeletal animation in the character file for a motion-based reference.'))
        return io.NodeOutput(reference, rig, image)


class AnimateMoCapRig(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MoCapAnything_AnimateRig", display_name="Animate MoCapAnything Rig", category="MoCapAnything",
            description="Apply captured rotations to the same prepared rig, preserving its mesh and skin weights.",
            inputs=[MoCapMotion.Input("motion"), MoCapRig.Input("rig")],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            outputs=[io.File3DGLB.Output("model_3d")],
        )

    @classmethod
    def execute(cls, motion, rig):
        return io.NodeOutput(animate_rig(motion, rig, execution_metadata(cls.hidden)))


class MoCapSkeleton3D(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MoCapAnything_Skeleton3D", display_name="MoCapAnything Motion to 3D Skeleton", category="MoCapAnything",
            description="Animated GLB bone geometry using the captured rotations and skeleton offsets. Connect to Preview 3D (Advanced). No Blender required.",
            inputs=[MoCapMotion.Input("motion")], outputs=[io.File3DGLB.Output("model_3d")],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def execute(cls, motion):
        return io.NodeOutput(skeleton_glb(motion, execution_metadata(cls.hidden)))

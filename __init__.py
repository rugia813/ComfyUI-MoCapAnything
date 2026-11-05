from comfy_api.latest import ComfyExtension

from .nodes import LoadMoCapAnything, CaptureMoCapAnything, PrepareMoCapRig, AnimateMoCapRig, MoCapSkeleton3D


class MoCapAnythingExtension(ComfyExtension):
    async def get_node_list(self):
        return [LoadMoCapAnything, CaptureMoCapAnything, PrepareMoCapRig, AnimateMoCapRig, MoCapSkeleton3D]


async def comfy_entrypoint():
    return MoCapAnythingExtension()

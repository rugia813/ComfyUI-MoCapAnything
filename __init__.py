from comfy_api.latest import ComfyExtension

from .nodes import LoadMoCapAnything, LoadMoCapReference, CaptureMoCapAnything, SaveMoCapMotion


class MoCapAnythingExtension(ComfyExtension):
    async def get_node_list(self):
        return [LoadMoCapAnything, LoadMoCapReference, CaptureMoCapAnything, SaveMoCapMotion]


async def comfy_entrypoint():
    return MoCapAnythingExtension()

"""Validate and execute the example with ComfyUI's real graph executor, without a web server."""
import asyncio
import json
from pathlib import Path
import sys

import bootstrap
import folder_paths
import nodes
import execution
from comfy_extras import nodes_video


class EventSink:
    client_id = None
    last_node_id = None

    def send_sync(self, event, data, sid=None):
        if event == "execution_error":
            print(json.dumps(data, indent=2), flush=True)


async def main():
    artifacts = bootstrap.ROOT / ".reference/artifacts"
    inputs = bootstrap.ROOT / ".reference/input"
    folder_paths.set_input_directory(str(inputs))
    folder_paths.set_output_directory(str(bootstrap.ROOT / "test_outputs"))
    folder_paths.add_model_folder_path("diffusion_models", str(artifacts))
    folder_paths.add_model_folder_path("clip_vision", str(artifacts))
    assert await nodes.load_custom_node(str(bootstrap.ROOT))
    for cls in await (await nodes_video.comfy_entrypoint()).get_node_list():
        nodes.NODE_CLASS_MAPPINGS[cls.GET_SCHEMA().node_id] = cls
    prompt = json.loads((bootstrap.ROOT / "example_workflows/video_to_bvh.api.json").read_text())
    prompt["4"]["inputs"]["model_name"] = "mocapanything_v2.safetensors"
    valid, error, outputs, node_errors = await execution.validate_prompt("mocap-test", prompt, None)
    assert valid, (error, node_errors)
    executor = execution.PromptExecutor(EventSink(), cache_type=execution.CacheType.CLASSIC, cache_args={"ram": 0, "ram_inactive": 0})
    await executor.execute_async(prompt, "mocap-test", execute_outputs=outputs)
    assert executor.success, executor.status_messages
    print("COMFY_WORKFLOW_EXECUTION_PASSED", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

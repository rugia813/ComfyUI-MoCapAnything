"""Validate and execute the example with ComfyUI's real graph executor, without a web server."""
import asyncio
import json
from pathlib import Path
import sys
import shutil

import bootstrap
import folder_paths
import nodes
import execution
from comfy_extras import nodes_video, nodes_load_3d


def prompt_from_workflow(path):
    workflow = json.loads(path.read_text())
    links = {link[0]: link for link in workflow['links']}
    prompt = {}
    for node in workflow['nodes']:
        if node['type'] == 'Note':
            continue
        values = iter(node.get('widgets_values', []))
        inputs = {}
        for socket in node.get('inputs', []):
            if 'widget' in socket:
                inputs[socket['name']] = next(values)
            if socket.get('link') is not None:
                link = links[socket['link']]
                inputs[socket['name']] = [str(link[1]), link[2]]
        prompt[str(node['id'])] = {'class_type': node['type'], 'inputs': inputs}
    return prompt


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
    folder_paths.set_temp_directory(str(bootstrap.ROOT / "test_outputs/temp"))
    Path(folder_paths.get_temp_directory()).mkdir(exist_ok=True)
    folder_paths.add_model_folder_path("diffusion_models", str(artifacts))
    folder_paths.add_model_folder_path("clip_vision", str(artifacts))
    folder_paths.add_model_folder_path("text_encoders", str(artifacts))
    assert await nodes.load_custom_node(str(bootstrap.ROOT))
    for module in (nodes_video, nodes_load_3d):
        for cls in await (await module.comfy_entrypoint()).get_node_list():
            nodes.NODE_CLASS_MAPPINGS[cls.GET_SCHEMA().node_id] = cls
    prompt = prompt_from_workflow(bootstrap.ROOT / 'example_workflows/custom_rig_to_3d.json')
    (inputs / '3d').mkdir(exist_ok=True)
    shutil.copy2(bootstrap.ROOT / 'test_outputs/rig_fixtures/character.glb', inputs / '3d/character.glb')
    prompt['11']['inputs']['model_file'] = '3d/character.glb'
    prompt["4"]["inputs"]["model_name"] = "video2pos2rot_epoch60.pt"
    valid, error, outputs, node_errors = await execution.validate_prompt("mocap-test", prompt, None)
    assert valid, (error, node_errors)
    executor = execution.PromptExecutor(EventSink(), cache_type=execution.CacheType.CLASSIC, cache_args={"ram": 0, "ram_inactive": 0})
    await executor.execute_async(prompt, "mocap-test", execute_outputs=outputs)
    assert executor.success, executor.status_messages
    print("COMFY_WORKFLOW_EXECUTION_PASSED", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

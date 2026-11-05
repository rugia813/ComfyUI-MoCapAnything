"""Validate and execute the example with ComfyUI's real graph executor, without a web server."""
import asyncio
import argparse
import json
from pathlib import Path
import sys
import shutil

import bootstrap
import folder_paths
import nodes
import execution
from comfy_extras import nodes_video, nodes_load_3d, nodes_save_3d, nodes_bg_removal


def prompt_from_workflow(path):
    workflow = json.loads(path.read_text(encoding="utf8"))
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--character', type=Path, help='Use a real rigged GLB instead of the generated fixture')
    parser.add_argument('--video', type=Path, help='Use a real input video instead of the small fixture')
    parser.add_argument('--forward-axis', default='+Z', choices=['+Z', '-Z', '+X', '-X'])
    args = parser.parse_args()
    artifacts = bootstrap.ROOT / ".reference/artifacts"
    inputs = bootstrap.ROOT / ".reference/input"
    inputs.mkdir(parents=True, exist_ok=True)
    folder_paths.set_input_directory(str(inputs))
    folder_paths.set_output_directory(str(bootstrap.ROOT / "test_outputs"))
    folder_paths.set_temp_directory(str(bootstrap.ROOT / "test_outputs/temp"))
    Path(folder_paths.get_temp_directory()).mkdir(exist_ok=True)
    folder_paths.add_model_folder_path("diffusion_models", str(artifacts))
    folder_paths.add_model_folder_path("clip_vision", str(artifacts))
    folder_paths.add_model_folder_path("text_encoders", str(artifacts))
    assert await nodes.load_custom_node(str(bootstrap.ROOT))
    for module in (nodes_video, nodes_load_3d, nodes_save_3d, nodes_bg_removal):
        for cls in await (await module.comfy_entrypoint()).get_node_list():
            nodes.NODE_CLASS_MAPPINGS[cls.GET_SCHEMA().node_id] = cls
    prompt = prompt_from_workflow(bootstrap.ROOT / 'example_workflows/custom_rig_to_3d.json')
    (inputs / '3d').mkdir(exist_ok=True)
    video = args.video or bootstrap.ROOT / 'example_workflows/input/hamster.mp4'
    shutil.copy2(video, inputs / video.name)
    prompt['1']['inputs']['file'] = video.name
    character = args.character or bootstrap.ROOT / 'test_outputs/rig_fixtures/animated_character.glb'
    shutil.copy2(character, inputs / '3d/character.glb')
    prompt['11']['inputs']['model_file'] = '3d/character.glb'
    prompt['5']['inputs']['forward_axis'] = args.forward_axis
    prompt["4"]["inputs"]["model_name"] = "video2pos2rot_epoch60.pt"
    valid, error, outputs, node_errors = await execution.validate_prompt("mocap-test", prompt, None)
    assert valid, (error, node_errors)
    executor = execution.PromptExecutor(EventSink(), cache_type=execution.CacheType.CLASSIC, cache_args={"ram": 0, "ram_inactive": 0})
    await executor.execute_async(prompt, "mocap-test", execute_outputs=outputs)
    assert executor.success, executor.status_messages
    print("COMFY_WORKFLOW_EXECUTION_PASSED", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

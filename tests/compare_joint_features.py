"""Check native T5-base pooled features against Hugging Face's upstream encoder."""
import json

import bootstrap
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import T5Config, T5EncoderModel
import comfy.sd

from mocap_extension.rig import joint_features, clean_joint_name


torch.set_grad_enabled(False)
path = bootstrap.ROOT / '.reference/artifacts/t5-base.safetensors'
clip = comfy.sd.load_clip([str(path)], clip_type=comfy.sd.CLIPType.STABLE_AUDIO, model_options={'dtype': torch.float32})
names = ['Hips', 'mixamorig:LeftUpLeg', 'Tail_01', 'Right(Hand)']
got = joint_features(clip, names)
config = json.loads((bootstrap.COMFY / 'comfy/text_encoders/t5_config_base.json').read_text())
expected_model = T5EncoderModel(T5Config(**config)).eval()
state = load_file(str(path))
state = {key: val for key, val in state.items() if key.startswith('encoder.') or key == 'shared.weight'}
state['encoder.embed_tokens.weight'] = state['shared.weight']
expected_model.load_state_dict(state, strict=True)
tokenizer = clip.tokenizer.clip_t5base.tokenizer
tokens = tokenizer([clean_joint_name(name) for name in names], padding=True, return_tensors='pt')
mask = tokens['attention_mask'][..., None]
hidden = expected_model(**tokens).last_hidden_state
expected = ((hidden * mask).sum(1) / mask.sum(1)).numpy()
np.testing.assert_allclose(got, expected, atol=3e-5, rtol=3e-5)
print('T5_FEATURE_PARITY_PASSED', np.max(np.abs(got - expected)))
del clip

import sys
sys.path.append("../src")

import os

import torch
from safetensors import safe_open
from transformers import MixtralForCausalLM, AutoTokenizer, pipeline
from slicegpt import layernorm_fusion
from slicegpt.adapters.mixtral_adapter import MixtralModelAdapter
from slicegpt.modules import RMSN


MODEL = "/srv/shared-data/huggingface/mixtral-rotated"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = MixtralForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
# use unknown token id on mixtral
# https://medium.com/@parikshitsaikia1619/mistral-mastery-fine-tuning-fast-inference-guide-62e163198b06
model.config.pad_token_id = tokenizer.unk_token_id
model_adapter = MixtralModelAdapter(model)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

layernorm_fusion.replace_layers(model_adapter)
layernorm_fusion.replace_modules(
    model_adapter.model,
    model_adapter.original_layer_norm_type,
    lambda _: RMSN(model_adapter.hidden_size),
    replace_layers=False,
)

for layer_adapter in model_adapter.get_layers():
    layer_adapter.layer.mlp_shortcut_Q = torch.zeros(model_adapter.hidden_size, model_adapter.hidden_size).to(
        dtype=torch.bfloat16
    )
    layer_adapter.layer.attn_shortcut_Q = torch.zeros(model_adapter.hidden_size, model_adapter.hidden_size).to(
        dtype=torch.bfloat16
    )

with safe_open(os.path.join(MODEL, "model-rotation.safetensors"), framework="pt", device="cpu") as f:
    weights = {k: f.get_tensor(k) for k in f.keys()}

model.load_state_dict(weights, strict=False)

output, = pipe("<s>[INST]Hi, I'm noah. Are you ?[/INST]", max_new_tokens=30)
print(output["generated_text"])

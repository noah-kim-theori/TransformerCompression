import sys
sys.path.append("../src")

import argparse
import json
import os
import subprocess

import torch
import torch.nn as nn
from bitsandbytes.nn import Linear4bit
from safetensors import safe_open
from transformers import MixtralForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
from slicegpt.adapters.mixtral_adapter import MixtralModelAdapter
from slicegpt.layernorm_fusion import replace_modules, replace_layers
from slicegpt.modules import RMSN
from slicegpt.rotate import slice_rotated_model
from slicegpt.slicing_scheduler import ConstSlicingScheduler


parser = argparse.ArgumentParser()
parser.add_argument("--sparsity", default=0, type=int)
parser.add_argument("--round-interval", default=64, type=int)
parser.add_argument("--quantize", default=False, action="store_true")
parser.add_argument("--prompts", default="./prompts.json")
parser.add_argument("--model", default="./models/mixtral-rotated")
args = parser.parse_args()

MODEL, PROMPTS = args.model,args.prompts
SPARSITY, ROUND_INTERVAL = args.sparsity / 100, args.round_interval
QUANTIZE = args.quantize

print("LOAD MODEL")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = MixtralForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
model_adapter = MixtralModelAdapter(model)

replace_layers(model_adapter)
replace_modules(
    model_adapter.model,
    model_adapter.original_layer_norm_type,
    lambda _: RMSN(model_adapter.hidden_size),
    replace_layers=False,
)

_pivot = next(model.parameters())
for layer_adapter in model_adapter.get_layers():
    layer_adapter.layer.mlp_shortcut_Q = _pivot.new_zeros(model_adapter.hidden_size, model_adapter.hidden_size)
    layer_adapter.layer.attn_shortcut_Q = _pivot.new_zeros(model_adapter.hidden_size, model_adapter.hidden_size)

with safe_open(os.path.join(MODEL, "model-rotation.safetensors"), framework="pt", device="cpu") as f:
    weights = {k: f.get_tensor(k) for k in f.keys()}

model.load_state_dict(weights, strict=False)

dim = int((1 - SPARSITY) * model_adapter.hidden_size)
dim -= dim % ROUND_INTERVAL
print("ROTATE:", dim)

scheduler = ConstSlicingScheduler(dim, do_slice_head=True)
slice_rotated_model(model_adapter, scheduler)

if QUANTIZE:
    print("QUANTIZE")
    def quantize(module: nn.Linear) -> Linear4bit:
        quant = Linear4bit(module.in_features, module.out_features, bias=module.bias is not None)
        quant.load_state_dict(module.state_dict())
        return quant

    for layer in tqdm(model.model.layers):
        replace_modules(
            layer,
            nn.Linear,
            quantize,
            replace_layers=False,
        )

print("MOVE TO GPU VRAM")
model.to("cuda:0")

def pipe(input_ids: str | list[dict[str, str]], **kwargs):
    outputs = model.generate(
        **{
            k: v.cuda()
            for k, v in tokenizer(input_ids, return_tensors="pt").items()
        },
        pad_token_id=tokenizer.eos_token_id,
        **kwargs)
    output, = tokenizer.batch_decode(outputs)
    return output

print("GENERATE")
simple = pipe("<s>[INST]Hi, I'm noah. Are you ?[/INST]", max_new_tokens=30)

with open(PROMPTS) as f:
    prompts = json.load(f)

PATH = f"./models/response/perf_{int(SPARSITY * 100)}{'_quant' if QUANTIZE else ''}.json"
os.makedirs(os.path.dirname(PATH))

def gpuview():
    _COMMAND = "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv"
    output = subprocess.check_output(_COMMAND.split()).decode("ascii")
    _, report = output.strip().split("\n")
    return report

outputs = {
    "__simple__": simple,
    "__metadata__": {
        "model": MODEL,
        "sparsity": SPARSITY,
        "round_interval": ROUND_INTERVAL,
        "quantize": QUANTIZE,
        "gpu": {
            "on-load": gpuview(),
            "post-inference": [],
        }
    }
}
for key, val in tqdm(prompts.items(), total=len(prompts)):
    result = pipe(val, max_new_tokens=2048)
    outputs[key] = result
    outputs["__metadata__"]["gpu"]["post-inference"].append(gpuview())
    with open(PATH, "w") as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)

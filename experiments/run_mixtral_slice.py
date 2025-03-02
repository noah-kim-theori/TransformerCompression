import sys
sys.path.append("../src")

from slicegpt import layernorm_fusion, rotate
from slicegpt.adapters.mixtral_adapter import MixtralModelAdapter
from slicegpt.slicing_scheduler import ConstSlicingScheduler

sys.path.pop()

import json
import os
import shutil

import torch
from datasets import Dataset
from safetensors.torch import save_file
from torch.utils.data import DataLoader, SubsetRandomSampler
from transformers import MixtralForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm

print("[*] LOAD MODEL")

MODEL = "./models/Mixtral-8x7B-Instruct-v0.1/"
PATH = "./models/mixtral-rotated"
DATASET = "./dataset.json"
# os.makedirs(PATH, exist_ok=True)
assert os.path.exists(PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = MixtralForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
# use unknown token id on mixtral
# https://medium.com/@parikshitsaikia1619/mistral-mastery-fine-tuning-fast-inference-guide-62e163198b06
model.config.pad_token_id = tokenizer.unk_token_id
model_adapter = MixtralModelAdapter(model)

print("[*] START FUSION")

layernorm_fusion.replace_layers(model_adapter)
layernorm_fusion.fuse_modules(model_adapter)

print("[*] PREPARE DATASET")

def prepare_dataloader(
    tokenizer,
    max_seqlen: int,
    batch_size: int,
    nsamples: int,
    seed: int = 0,
) -> DataLoader:
    with open(DATASET) as f:
        scripts = json.load(f)

    dataset = Dataset.from_list([{"text": script} for script in scripts.values()])

    def tokenize(datum: dict):
        return tokenizer(
            datum["text"],
            # padding=True,
            padding="longest",
            truncation=True,
            max_length=max_seqlen,
            return_tensors="pt"
        )

    dataset.set_transform(tokenize)
    generator = torch.Generator().manual_seed(seed)
    sampler = SubsetRandomSampler(
        torch.randperm(len(dataset), generator=generator)[:nsamples])
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

tokenizer.pad_token_id = tokenizer.unk_token_id
train_loader = prepare_dataloader(
    tokenizer=tokenizer,
    max_seqlen=4096,
    batch_size=1,
    nsamples=128,
)

print("[*] ROTATE")

# do not slice
scheduler = ConstSlicingScheduler(model_adapter.hidden_size)
rotate.rotate_and_slice(
    model_adapter,
    train_loader,
    scheduler,
    ignore_tokens=[tokenizer.pad_token_id])

print("[*] WRITE")

with open(os.path.join(PATH, "slice_config.json"), "w") as f:
    f.write(model_adapter.slicing_conf.to_json_string())

with open(os.path.join(MODEL, "model.safetensors.index.json")) as f:
    loaded = json.load(f)
    weight_map = loaded["weight_map"]

ckpt = model.state_dict()
files, dropped = {}, []
for k, v in weight_map.items():
    if v not in files:
        files[v] = {}
    if k not in ckpt:
        dropped.append(k)
        continue

    files[v][k] = ckpt[k]

keys = set(ckpt) - set(
    key
    for keys in files.values()
    for key in keys)

with open(os.path.join(PATH, "model.safetensors.log"), "w") as f:
    json.dump({"dropped": dropped, "rotation": list(keys)}, f, indent=4)

files["model-rotation.safetensors"] = {k: ckpt[k].contiguous() for k in keys}

for k, tensors in tqdm(files.items(), total=len(files)):
    save_file(tensors, os.path.join(PATH, k), {"format": "pt"})

with open(os.path.join(PATH, "model.safetensors.index.json"), "w") as f:
    loaded["weight_map"] = {
        key: filename
        for filename, keys in files.items()
        for key in keys
    }
    json.dump(loaded, f, indent=4)

CONFIGS = [
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
]

for config in CONFIGS:
    shutil.copy(os.path.join(MODEL, config), os.path.join(PATH, config))

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
    loaded = json.load(f)["weight_map"]

files, ckpt = {}, model.state_dict()
for k, v in loaded.items():
    if v not in files:
        files[v] = {}
    if k not in ckpt:
        MODEL_LN = "model.norm.weight"
        PRE_ATTN_LN = "input_layernorm.weight"
        PRE_MLP_LN = "post_attention_layernorm.weight"
        assert any(
            k.endswith(postfix) for postfix in [MODEL_LN, PRE_ATTN_LN, PRE_MLP_LN]
        ), f"unintended key: {k}"
        if k.endswith(PRE_ATTN_LN):
             k = k.replace(PRE_ATTN_LN, "attn_shortcut_Q")
        if k.endswith(PRE_MLP_LN):
             k = k.replace(PRE_MLP_LN, "mlp_shortcut_Q")

    if k not in ckpt:
        print(k)
        continue

    files[v][k] = ckpt[k]

for k, tensors in tqdm(files.items(), total=len(files)):
    save_file(tensors, os.path.join(PATH, k), {"format": "pt"})

CONFIGS = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
]

for config in CONFIGS:
    shutil.copy(os.path.join(MODEL, config), os.path.join(PATH, config))

# Dataset cho FireboltVL: cùng format CCDataset (JSONL/JSON, conversations, image).
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoProcessor
from PIL import Image
from functools import partial
import random
import numpy as np


def load_jsonl(json_path):
    with open(json_path, "r") as f:
        return [json.loads(line) for line in f]


def load_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


class CCDataset(Dataset):
    def __init__(self, image_path, json_path, tokenizer, processor, image_token: str = "<image>"):
        self.image_path = image_path
        self.json_path = json_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.image_token = image_token
        if json_path.endswith(".jsonl"):
            self.data = load_jsonl(json_path)
        elif json_path.endswith(".json"):
            self.data = load_json(json_path)
        else:
            raise ValueError(f"Unsupported file extension: {json_path}")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<|pad|>"

    def __len__(self):
        return len(self.data)

    def _format_prompt(self, conversation, has_image: bool):
        conv = []
        for turn in conversation:
            role = turn.get("from")
            if role == "human":
                text = turn.get("value") or ""
                if has_image and self.image_token not in text:
                    text = f"{self.image_token} {text}"
                conv.append({"role": "user", "content": "<image> " + text if text else "<image>"})
            elif role == "assistant":
                conv.append({"role": "assistant", "content": turn["value"]})
            else:
                conv.append({"role": "user", "content": "<image> " + (turn.get("value") or "")})
        encoded = self.tokenizer.apply_chat_template(
            conv, add_generation_prompt=False, tokenize=True, return_tensors="pt",
        )
        if isinstance(encoded, torch.Tensor):
            input_ids = encoded
            attention_mask = torch.ones_like(encoded, dtype=torch.long)
        else:
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask") or torch.ones_like(input_ids, dtype=torch.long)
        return input_ids, attention_mask

    def __getitem__(self, idx):
        item = self.data[idx]
        has_image = "image" in item
        if has_image:
            img_path = os.path.join(self.image_path, item["image"])
            try:
                image = Image.open(img_path).convert("RGB")
                pixel_values = self.processor(images=[image], return_tensors="pt")["pixel_values"][0]
            except FileNotFoundError:
                return None
        else:
            pixel_values = torch.zeros(3, 384, 384, dtype=torch.float32)
        input_ids, attention_mask = self._format_prompt(item["conversations"], has_image=has_image)
        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "pixel_values": pixel_values,
        }


def collate_with_pad(batch, pad_token_id: int):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    pixel_values = torch.stack([ex["pixel_values"] for ex in batch], dim=0)
    seqs = [ex["input_ids"] for ex in batch]
    masks = [ex["attention_mask"] for ex in batch]
    max_len = max(x.size(0) for x in seqs)
    padded_ids, padded_mask = [], []
    for ids, m in zip(seqs, masks):
        pad_len = max_len - ids.size(0)
        if pad_len > 0:
            ids = torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)], dim=0)
            m = torch.cat([m, torch.zeros(pad_len, dtype=torch.long)], dim=0)
        padded_ids.append(ids)
        padded_mask.append(m)
    input_ids = torch.stack(padded_ids, dim=0)
    attention_mask = torch.stack(padded_mask, dim=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "pixel_values": pixel_values}


def worker_init_fn(worker_id):
    base_seed = torch.initial_seed() % (2 ** 31)
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


def create_dataloader(
    image_path,
    json_path,
    tokenizer,
    processor,
    batch_size=32,
    num_workers=4,
    ddp=False,
    rank=0,
    world_size=1,
    seed: int = 42,
    drop_last: bool = False,
    val_batch_size: int = 4,
    train_val_split=(0.95, 0.05),
):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    dataset = CCDataset(image_path=image_path, json_path=json_path, tokenizer=tokenizer, processor=processor)
    lengths = [int(len(dataset) * train_val_split[0]), len(dataset) - int(len(dataset) * train_val_split[0])]
    if lengths[1] <= 0:
        lengths[0], lengths[1] = len(dataset) - 1, 1
    g = torch.Generator().manual_seed(seed)
    train_set, val_set = torch.utils.data.random_split(dataset, lengths, generator=g)
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, drop_last=drop_last) if ddp else None
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False) if ddp else None
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    collate = partial(collate_with_pad, pad_token_id=pad_id)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler,
        collate_fn=collate, num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0), worker_init_fn=worker_init_fn if num_workers > 0 else None,
        drop_last=drop_last, prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_set, batch_size=min(val_batch_size, batch_size), shuffle=False, sampler=val_sampler,
        collate_fn=collate, num_workers=0, pin_memory=True,
    )
    return {
        "train_dataloader": train_loader,
        "val_dataloader": val_loader,
        "train_sampler": train_sampler,
        "val_sampler": val_sampler,
    }

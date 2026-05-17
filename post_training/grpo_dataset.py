import os
import random
from functools import partial
from typing import List, Optional

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class GRPODataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        image_base_path: str,
        tokenizer,
        processor,
        max_prompt_length: int = 256,
        image_token: str = "<image>",
        split: str = "train",
        filter_correct_only: bool = False,
    ):
        self.image_base_path = image_base_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_prompt_length = max_prompt_length
        self.image_token = image_token

        ds = load_dataset(dataset_name, split=split)
        if filter_correct_only:
            ds = ds.filter(lambda x: x["correct"])
        self.data = ds

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        ground_truth = item["gt"]
        image_rel_path = item["image"]

        image_path = os.path.join(self.image_base_path, image_rel_path)
        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, OSError):
            return None

        try:
            pixel_values = self.processor(images=[image], return_tensors="pt")["pixel_values"]
        finally:
            image.close()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. When answering questions, "
                    "first think step by step inside <think>...</think> tags, "
                    "then provide your final answer inside <answer>...</answer> tags."
                ),
            },
            {"role": "user", "content": f"{self.image_token}\n{question}"},
        ]
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        if isinstance(tokenized, torch.Tensor):
            prompt_input_ids = tokenized.squeeze(0)
        else:
            prompt_input_ids = tokenized["input_ids"].squeeze(0)

        if prompt_input_ids.size(0) > self.max_prompt_length:
            prompt_input_ids = prompt_input_ids[: self.max_prompt_length]

        return {
            "prompt_input_ids": prompt_input_ids,
            "pixel_values": pixel_values,
            "image_counts": torch.tensor(1, dtype=torch.long),
            "ground_truth": ground_truth,
        }


def collate_grpo_prompts(batch, pad_token_id: int):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    prompt_ids_list = [ex["prompt_input_ids"] for ex in batch]
    max_len = max(ids.size(0) for ids in prompt_ids_list)

    padded_ids = []
    padded_masks = []
    for ids in prompt_ids_list:
        pad_len = max_len - ids.size(0)
        if pad_len > 0:
            pad = torch.full((pad_len,), pad_token_id, dtype=torch.long)
            ids = torch.cat([pad, ids], dim=0)
            mask = torch.cat([torch.zeros(pad_len, dtype=torch.long), torch.ones(ids.size(0) - pad_len, dtype=torch.long)], dim=0)
        else:
            mask = torch.ones(ids.size(0), dtype=torch.long)
        padded_ids.append(ids)
        padded_masks.append(mask)

    pixel_values_list = [ex["pixel_values"] for ex in batch]
    max_images = max(pv.size(0) for pv in pixel_values_list)
    padded_pvs = []
    for pv in pixel_values_list:
        pad_len = max_images - pv.size(0)
        if pad_len > 0:
            pad = torch.zeros((pad_len, *pv.shape[1:]), dtype=pv.dtype)
            pv = torch.cat([pv, pad], dim=0)
        padded_pvs.append(pv)

    return {
        "prompt_input_ids": torch.stack(padded_ids, dim=0),
        "prompt_attention_mask": torch.stack(padded_masks, dim=0),
        "pixel_values": torch.stack(padded_pvs, dim=0),
        "image_counts": torch.stack([ex["image_counts"] for ex in batch], dim=0),
        "ground_truths": [ex["ground_truth"] for ex in batch],
    }


def _worker_init_fn(worker_id):
    base_seed = torch.initial_seed() % (2**31)
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


def create_grpo_dataloader(
    dataset_name: str,
    image_base_path: str,
    tokenizer,
    processor,
    batch_size: int = 1,
    num_workers: int = 4,
    ddp: bool = False,
    rank: int = 0,
    world_size: int = 1,
    max_prompt_length: int = 256,
    filter_correct_only: bool = False,
    seed: int = 42,
):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    dataset = GRPODataset(
        dataset_name=dataset_name,
        image_base_path=image_base_path,
        tokenizer=tokenizer,
        processor=processor,
        max_prompt_length=max_prompt_length,
        filter_correct_only=filter_correct_only,
    )

    sampler = None
    if ddp:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True,
        )

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    collate = partial(collate_grpo_prompts, pad_token_id=pad_id)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        drop_last=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return {"dataloader": dataloader, "sampler": sampler}

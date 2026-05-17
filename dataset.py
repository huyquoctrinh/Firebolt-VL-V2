# Dataset cho FireboltVL: cùng format CCDataset (JSONL/JSON, conversations, image).
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoProcessor
from PIL import Image
from functools import partial
import random
import numpy as np
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

IGNORE_INDEX = -100
SOURCE_ALIASES = {
    "llava_cot",
    "llava_inst",
    "llava_mix_vsft",
    "llaval_mix_vsft",
    "llava_next",
    "llava_sharegpt",
    "mmpr",
    "mmpr-v1.2",
    "MMPR-v1.2",
}


def load_jsonl(json_path):
    with open(json_path, "r") as f:
        return [json.loads(line) for line in f]


def load_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def iter_json_records(json_path: Path):
    if json_path.suffix == ".jsonl":
        with json_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    if isinstance(record, dict):
                        yield record
        return

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        for record in data:
            if isinstance(record, dict):
                yield record
    elif isinstance(data, dict):
        for key in ("data", "annotations", "items"):
            values = data.get(key)
            if isinstance(values, list):
                for record in values:
                    if isinstance(record, dict):
                        yield record
                return
        yield data


def as_path_list(value) -> List[Any]:
    if isinstance(value, (str, Path)):
        return [value]
    try:
        return list(value)
    except TypeError:
        pass
    return [value]


def clean_image_value(value: Any):
    if isinstance(value, list):
        values = []
        for item in value:
            cleaned = clean_image_value(item)
            if cleaned:
                values.append(cleaned)
        return values
    if not isinstance(value, str):
        return ""
    value = value.strip()
    if value.startswith("./"):
        value = value[2:]
    return value.replace("\\", "/")


def source_key(value: Any) -> str:
    return str(value).strip().rstrip("/").split("/")[-1]


def discover_json_specs(json_path, dataset_root: Path) -> List[Tuple[Path, List[str]]]:
    specs: List[Tuple[Path, List[str]]] = []
    for raw_entry in as_path_list(json_path):
        entry = str(raw_entry)
        key = source_key(entry)
        if key in SOURCE_ALIASES:
            specs.extend(discover_source_specs(key, dataset_root))
            continue

        path = Path(entry)
        if not path.is_absolute():
            path = dataset_root / path
        if path.is_dir():
            inferred = source_key(path)
            if inferred in SOURCE_ALIASES:
                specs.extend(discover_source_specs(inferred, dataset_root))
            else:
                specs.extend((p, infer_image_roots(p, dataset_root)) for p in sorted(path.rglob("*.json*")))
        elif path.is_file():
            specs.append((path, infer_image_roots(path, dataset_root)))
        else:
            raise FileNotFoundError(f"Annotation source not found: {entry}")
    return specs


def discover_source_specs(source: str, dataset_root: Path) -> List[Tuple[Path, List[str]]]:
    normalized = source.lower()
    specs: List[Tuple[Path, List[str]]] = []

    if normalized == "llava_cot":
        for name in ("train.jsonl", "train_llava.json"):
            path = dataset_root / "llava_cot" / name
            if path.is_file():
                specs.append((path, ["llava_cot"]))
        return specs

    if normalized == "llava_inst":
        root = dataset_root / "llava_inst"
        image_roots = [
            "llava_inst_image",
            "llava_cot/coco/train2017",
            "llava_next/llava_next_raw_format/coco/train2017",
        ]
        return [(path, image_roots) for path in sorted(root.glob("*.json"))]

    if normalized in {"llava_mix_vsft", "llaval_mix_vsft"}:
        root = dataset_root / "llaval_mix_vsft"
        preferred = root / "llava_vfst_converted.json"
        paths = [preferred] if preferred.is_file() else sorted(root.glob("*.json"))
        return [(path, ["llaval_mix_vsft/images", "llaval_mix_vsft"]) for path in paths]

    if normalized == "llava_next":
        path = dataset_root / "llava_next/llava_next_raw_format/llava_next_raw_format_processed.json"
        return [(path, ["llava_next/llava_next_raw_format"])] if path.is_file() else []

    if normalized == "llava_sharegpt":
        processed = dataset_root / "llava_sharegpt/llava_sharegpt_processed.jsonl"
        if processed.is_file():
            specs.append((processed, [""]))
        sft_root = dataset_root / "llava_sharegpt/sft"
        specs.extend((path, ["llava_sharegpt/image_data"]) for path in sorted(sft_root.rglob("*.jsonl")))
        return specs

    if normalized in {"mmpr", "mmpr-v1.2"}:
        return discover_mmpr_specs(dataset_root)

    return specs


def discover_mmpr_specs(dataset_root: Path) -> List[Tuple[Path, List[str]]]:
    meta_path = dataset_root / "MMPR-v1.2/meta.json"
    if not meta_path.is_file():
        annotations = dataset_root / "MMPR-v1.2/annotations"
        return [(path, ["MMPR-v1.2/images"]) for path in sorted(annotations.glob("*.jsonl"))]

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    specs = []
    seen = set()
    for item in meta.values():
        annotation = item.get("annotation")
        image_root = item.get("root", "MMPR-v1.2/images")
        if not annotation:
            continue
        key = (annotation, image_root)
        if key in seen:
            continue
        seen.add(key)
        path = dataset_root / annotation
        if path.is_file():
            specs.append((path, [image_root]))
    return specs


def infer_image_roots(path: Path, dataset_root: Path) -> List[str]:
    try:
        rel = path.resolve().relative_to(dataset_root.resolve())
    except ValueError:
        return [""]
    parts = rel.parts
    if not parts:
        return [""]
    if parts[0] == "llava_cot":
        return ["llava_cot"]
    if parts[0] == "llava_inst":
        return ["llava_inst_image", "llava_cot/coco/train2017", "llava_next/llava_next_raw_format/coco/train2017"]
    if parts[0] == "llaval_mix_vsft":
        return ["llaval_mix_vsft/images", "llaval_mix_vsft"]
    if parts[0] == "llava_next":
        return ["llava_next/llava_next_raw_format"]
    if parts[0] == "llava_sharegpt":
        if path.name == "llava_sharegpt_processed.jsonl":
            return [""]
        if len(parts) > 2 and parts[1] == "image_data":
            parent = "/".join(parts[:-1])
            return [f"{parent}/png", f"{parent}/documents", parent, "llava_sharegpt/image_data"]
        return ["llava_sharegpt/image_data"]
    if parts[0] == "MMPR-v1.2":
        for spec_path, roots in discover_mmpr_specs(dataset_root):
            if spec_path == path:
                return roots
        return ["MMPR-v1.2/images"]
    return [""]


def first_text(record: Dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        text = value if isinstance(value, str) else str(value)
        text = text.strip()
        if text:
            return text
    return ""


def mask_image_patches(pixel_values, mask_ratio, patch_size=14):
    """Zero out random patches at pixel level. Returns (masked_tensor, True)."""
    _, H, W = pixel_values.shape
    grid_h, grid_w = H // patch_size, W // patch_size
    num_patches = grid_h * grid_w
    num_mask = int(num_patches * mask_ratio)
    if num_mask == 0:
        return pixel_values, False
    indices = torch.randperm(num_patches)[:num_mask]
    masked = pixel_values.clone()
    for idx in indices:
        r, c = idx // grid_w, idx % grid_w
        masked[:, r * patch_size:(r + 1) * patch_size, c * patch_size:(c + 1) * patch_size] = 0.0
    return masked, True


class CCDataset(Dataset):
    def __init__(
        self,
        image_path,
        json_path,
        tokenizer,
        processor,
        image_token: str = "<image>",
        max_length: Optional[int] = None,
        vjepa_processor=None,
        dino_processor=None,
    ):
        self.image_path = image_path
        self.dataset_root = Path(image_path)
        self.json_path = json_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.image_token = image_token
        self.max_length = max_length
        self.vjepa_processor = vjepa_processor
        self.dino_processor = dino_processor
        self.data = self._load_mixed_records(json_path)
        if not self.data:
            raise ValueError(f"No usable training records found in {json_path}")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<|pad|>"

    def __len__(self):
        return len(self.data)

    def _load_mixed_records(self, json_path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        specs = discover_json_specs(json_path, self.dataset_root)
        for path, image_roots in specs:
            for record in iter_json_records(path):
                item = self._normalize_record(record, image_roots)
                if item is not None:
                    records.append(item)
        return records

    def _normalize_record(self, record: Dict[str, Any], image_roots: List[str]) -> Optional[Dict[str, Any]]:
        conversations = record.get("conversations")
        if not isinstance(conversations, list):
            question = first_text(record, ["question", "prompt", "instruction", "query"])
            answer = first_text(record, ["chosen", "response", "output", "answer", "answer_gt", "gt_answer", "label"])
            if question and answer:
                conversations = [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": answer},
                ]
            else:
                return None

        normalized_conversations = []
        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            value = turn.get("value")
            if value is None:
                value = turn.get("content")
            if value is None:
                continue
            normalized_conversations.append(
                {
                    "from": turn.get("from") or turn.get("role"),
                    "value": value if isinstance(value, str) else str(value),
                }
            )

        if not normalized_conversations:
            return None

        image = record.get("image")
        if image is None:
            image = first_text(record, ["imgname", "image_path", "filename", "file_name"])

        return {
            "id": record.get("id"),
            "image": clean_image_value(image),
            "conversations": normalized_conversations,
            "_image_roots": image_roots,
        }

    def _normalize_role(self, role):
        role = (role or "").lower()
        if role in {"assistant", "gpt"}:
            return "assistant"
        if role == "system":
            return "system"
        return "user"

    def _build_messages(self, conversation, has_image: bool, image_count: int = 0):
        messages = []
        for turn in conversation:
            role = self._normalize_role(turn.get("from") or turn.get("role"))
            messages.append({"role": role, "content": turn.get("value") or turn.get("content") or ""})

        if has_image and messages:
            required_image_tokens = max(1, image_count)
            existing_image_tokens = sum(msg["content"].count(self.image_token) for msg in messages)
            missing_image_tokens = max(0, required_image_tokens - existing_image_tokens)
            if missing_image_tokens:
                prefix = " ".join([self.image_token] * missing_image_tokens)
                first_user_idx = next((i for i, msg in enumerate(messages) if msg["role"] == "user"), None)
                if first_user_idx is None:
                    messages.insert(0, {"role": "user", "content": prefix})
                else:
                    content = messages[first_user_idx]["content"]
                    messages[first_user_idx]["content"] = f"{prefix} {content}" if content else prefix
        return messages

    def _tokenize_piece(self, text: str):
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _format_prompt(self, conversation, has_image: bool, image_count: int = 0):
        messages = self._build_messages(conversation, has_image=has_image, image_count=image_count)
        input_ids = []
        labels = []

        bos = self.tokenizer.bos_token or ""
        if bos:
            bos_ids = self._tokenize_piece(bos)
            input_ids.extend(bos_ids)
            labels.extend([IGNORE_INDEX] * len(bos_ids))

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            header_ids = self._tokenize_piece(f"<|im_start|>{role}\n")
            content_footer_ids = self._tokenize_piece(f"{content}<|im_end|>\n")
            input_ids.extend(header_ids)
            labels.extend([IGNORE_INDEX] * len(header_ids))
            input_ids.extend(content_footer_ids)
            if role == "assistant":
                labels.extend(content_footer_ids)
            else:
                labels.extend([IGNORE_INDEX] * len(content_footer_ids))

        if self.max_length is not None:
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]

        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        labels = torch.tensor(labels, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        return input_ids, attention_mask, labels

    def _candidate_image_paths(self, image_value: str, image_roots: List[str]) -> List[Path]:
        image_value = clean_image_value(image_value)
        if not image_value:
            return []
        image_path = Path(image_value)
        if image_path.is_absolute():
            return [image_path]

        candidates = [self.dataset_root / image_value]
        for root in image_roots:
            if not root:
                continue
            root = root.strip("/")
            if image_value == root or image_value.startswith(root + "/"):
                candidates.append(self.dataset_root / image_value)
            else:
                candidates.append(self.dataset_root / root / image_value)
        return candidates

    def _get_image_paths(self, item):
        image_value = item.get("image")
        image_roots = item.get("_image_roots", [""])
        if isinstance(image_value, str):
            image_values = [image_value]
        elif isinstance(image_value, list):
            image_values = [x for x in image_value if isinstance(x, str) and x]
        else:
            image_values = []
        resolved = []
        for value in image_values:
            for candidate in self._candidate_image_paths(value, image_roots):
                if candidate.is_file():
                    resolved.append(str(candidate))
                    break
        return resolved

    def __getitem__(self, idx):
        item = self.data[idx]
        image_paths = self._get_image_paths(item)
        has_image = len(image_paths) > 0
        pixel_values_vjepa = None
        pixel_values_dino = None
        if has_image:
            images = []
            for image_path in image_paths:
                try:
                    images.append(Image.open(image_path).convert("RGB"))
                except FileNotFoundError:
                    continue
            if not images:
                return None
            try:
                pixel_values = self.processor(images=images, return_tensors="pt")["pixel_values"]
                if self.vjepa_processor is not None:
                    vjepa_out = self.vjepa_processor(videos=[[image] for image in images], return_tensors="pt")
                    key = "pixel_values_videos" if "pixel_values_videos" in vjepa_out else "pixel_values"
                    pixel_values_vjepa = vjepa_out[key]
                    if pixel_values_vjepa.dim() == 5 and pixel_values_vjepa.size(1) == 1:
                        pixel_values_vjepa = pixel_values_vjepa[:, 0]
                if self.dino_processor is not None:
                    pixel_values_dino = self.dino_processor(images=images, return_tensors="pt")["pixel_values"]
            finally:
                for image in images:
                    image.close()
        else:
            pixel_values = torch.zeros(1, 3, 384, 384, dtype=torch.float32)
        image_count = pixel_values.size(0)
        input_ids, attention_mask, labels = self._format_prompt(
            item["conversations"],
            has_image=has_image,
            image_count=image_count,
        )
        result = {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
            "pixel_values": pixel_values,
            "image_counts": torch.tensor(image_count, dtype=torch.long),
        }
        if pixel_values_vjepa is not None:
            result["pixel_values_vjepa"] = pixel_values_vjepa
        if pixel_values_dino is not None:
            result["pixel_values_dino"] = pixel_values_dino
        return result


def collate_with_pad(batch, pad_token_id: int, mask_ratio: float = 0.0, mask_prob: float = 0.0, patch_size: int = 14):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    def pad_image_stack(values):
        max_images = max(x.size(0) for x in values)
        padded = []
        for x in values:
            pad_len = max_images - x.size(0)
            if pad_len > 0:
                pad = torch.zeros((pad_len, *x.shape[1:]), dtype=x.dtype)
                x = torch.cat([x, pad], dim=0)
            padded.append(x)
        return torch.stack(padded, dim=0)

    pixel_values_list = [ex["pixel_values"] for ex in batch]
    image_counts = torch.stack([ex["image_counts"] for ex in batch], dim=0)

    has_vjepa = all("pixel_values_vjepa" in ex for ex in batch)
    if has_vjepa:
        pixel_values_vjepa = pad_image_stack([ex["pixel_values_vjepa"] for ex in batch])
    else:
        pixel_values_vjepa = None

    has_dino = all("pixel_values_dino" in ex for ex in batch)
    if has_dino:
        pixel_values_dino = pad_image_stack([ex["pixel_values_dino"] for ex in batch])
    else:
        pixel_values_dino = None

    if mask_ratio > 0 and mask_prob > 0:
        masked_pixels = []
        mask_flags_list = []
        for pv in pixel_values_list:
            if random.random() < mask_prob:
                masked_images = []
                flags = []
                for image_pv in pv:
                    mp, flag = mask_image_patches(image_pv, mask_ratio, patch_size)
                    masked_images.append(mp)
                    flags.append(flag)
                mp = torch.stack(masked_images, dim=0)
                masked_pixels.append(mp)
                mask_flags_list.append(any(flags))
            else:
                masked_pixels.append(pv)
                mask_flags_list.append(False)
        pixel_values = pad_image_stack(masked_pixels)
        mask_flags = torch.tensor(mask_flags_list, dtype=torch.bool)
    else:
        pixel_values = pad_image_stack(pixel_values_list)
        mask_flags = torch.zeros(len(batch), dtype=torch.bool)

    seqs = [ex["input_ids"] for ex in batch]
    masks = [ex["attention_mask"] for ex in batch]
    label_seqs = [ex["labels"] for ex in batch]
    max_len = max(x.size(0) for x in seqs)
    padded_ids, padded_mask, padded_labels = [], [], []
    for ids, m, labels in zip(seqs, masks, label_seqs):
        pad_len = max_len - ids.size(0)
        if pad_len > 0:
            ids = torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)], dim=0)
            m = torch.cat([m, torch.zeros(pad_len, dtype=torch.long)], dim=0)
            labels = torch.cat([labels, torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long)], dim=0)
        padded_ids.append(ids)
        padded_mask.append(m)
        padded_labels.append(labels)
    input_ids = torch.stack(padded_ids, dim=0)
    attention_mask = torch.stack(padded_mask, dim=0)
    labels = torch.stack(padded_labels, dim=0)
    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_counts": image_counts,
        "mask_flags": mask_flags,
    }
    if pixel_values_vjepa is not None:
        result["pixel_values_vjepa"] = pixel_values_vjepa
    if pixel_values_dino is not None:
        result["pixel_values_dino"] = pixel_values_dino
    return result


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
    max_length: Optional[int] = None,
    mask_ratio: float = 0.0,
    mask_prob: float = 0.0,
    patch_size: int = 14,
    vjepa_processor=None,
    dino_processor=None,
):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    dataset = CCDataset(
        image_path=image_path,
        json_path=json_path,
        tokenizer=tokenizer,
        processor=processor,
        max_length=max_length,
        vjepa_processor=vjepa_processor,
        dino_processor=dino_processor,
    )
    lengths = [int(len(dataset) * train_val_split[0]), len(dataset) - int(len(dataset) * train_val_split[0])]
    if lengths[1] <= 0:
        lengths[0], lengths[1] = len(dataset) - 1, 1
    g = torch.Generator().manual_seed(seed)
    train_set, val_set = torch.utils.data.random_split(dataset, lengths, generator=g)
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, drop_last=drop_last) if ddp else None
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False) if ddp else None
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    collate = partial(collate_with_pad, pad_token_id=pad_id, mask_ratio=mask_ratio, mask_prob=mask_prob, patch_size=patch_size)
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

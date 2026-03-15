"""
Convert llava_vfst.json to LLaVA-compatible format:
  - Map "user" -> "human", "assistant" -> "gpt"
  - Prepend <image>\n to the first human message for samples with images
  - Strip directory prefix from image paths (keep only filename)
"""

import json
import os

INPUT_PATH = "/home/mamba/ML_project/Testing/Huy/joint_vlm/dataset/llaval_mix_vsft/llava_vfst.json"
OUTPUT_PATH = "/home/mamba/ML_project/Testing/Huy/joint_vlm/dataset/llaval_mix_vsft/llava_vfst_converted.json"

ROLE_MAP = {"user": "human", "assistant": "gpt"}


def convert_sample(sample):
    has_image = "image" in sample

    new_conversations = []
    first_human = True
    for turn in sample["conversations"]:
        role = ROLE_MAP.get(turn["from"], turn["from"])
        value = turn["value"] if turn["value"] is not None else ""

        # Add <image> token to the first human message
        if has_image and role == "human" and first_human:
            value = "<image>\n" + value
            first_human = False

        new_conversations.append({"from": role, "value": value})

    new_sample = {"id": sample["id"], "conversations": new_conversations}

    if has_image:
        # Strip directory prefix, keep only filename
        new_sample["image"] = os.path.basename(sample["image"])

    return new_sample


def main():
    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples from {INPUT_PATH}")

    converted = [convert_sample(s) for s in data]

    with open(OUTPUT_PATH, "w") as f:
        json.dump(converted, f, indent=2)

    print(f"Saved converted data to {OUTPUT_PATH}")

    # Print a sample for verification
    print("\nSample entry:")
    print(json.dumps(converted[0], indent=2))


if __name__ == "__main__":
    main()

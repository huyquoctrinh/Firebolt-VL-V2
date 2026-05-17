import re
import string
from typing import Dict, List, Optional

import torch


def extract_answer(response: str) -> Optional[str]:
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    articles = {"a", "an", "the"}
    words = text.split()
    words = [w for w in words if w not in articles]
    return " ".join(words).strip()


def word_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def accuracy_reward(response: str, ground_truth: str) -> float:
    norm_gt = normalize_answer(ground_truth)
    if not norm_gt:
        return 0.0

    # Best case: answer is in <answer> tags
    extracted = extract_answer(response)
    if extracted is not None:
        norm_extracted = normalize_answer(extracted)
        if not norm_extracted:
            return 0.1  # Used tags but empty answer
        if norm_extracted == norm_gt:
            return 1.0
        if norm_gt in norm_extracted or norm_extracted in norm_gt:
            return 0.8
        return max(0.1, word_f1(extracted, ground_truth))

    # Fallback: check full response for GT content
    norm_response = normalize_answer(response)
    if norm_gt in norm_response:
        return 0.4

    # Soft fallback: word overlap between response and GT
    f1 = word_f1(response, ground_truth)
    return f1 * 0.3


def format_reward(response: str) -> float:
    reward = 0.0
    # Full marks for target format
    if re.search(r"<think>.*?</think>", response, re.DOTALL):
        reward += 0.5
    if re.search(r"<answer>.*?</answer>", response, re.DOTALL):
        reward += 0.5

    # Partial credit for any structured reasoning (model's existing format)
    if reward == 0.0:
        has_reasoning = bool(
            re.search(r"<REASONING>|<CONCLUSION>|<SUMMARY>", response)
        )
        if has_reasoning:
            reward = 0.1

    return reward


def compute_rewards(
    responses: List[str],
    ground_truths: List[str],
    reward_weights: Dict[str, float],
) -> torch.Tensor:
    w_acc = reward_weights.get("accuracy", 2.0)
    w_fmt = reward_weights.get("format", 1.0)
    rewards = []
    for resp, gt in zip(responses, ground_truths):
        r = w_acc * accuracy_reward(resp, gt) + w_fmt * format_reward(resp)
        rewards.append(r)
    return torch.tensor(rewards, dtype=torch.float32)

import math
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from omegaconf import DictConfig
from transformers import GenerationConfig

from .rewards import compute_rewards


def _raw_model(model):
    return model.module if isinstance(model, DDP) else model


class GRPOTrainer:
    def __init__(
        self,
        cfg: DictConfig,
        policy_model: nn.Module,
        ref_model: nn.Module,
        tokenizer,
        processor,
        optimizer,
        scheduler,
        device: torch.device,
        rank: Optional[int] = None,
    ):
        self.cfg = cfg
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.processor = processor
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.rank = rank

        grpo_cfg = cfg.grpo
        self.group_size = int(grpo_cfg.group_size)
        self.clip_epsilon = float(grpo_cfg.clip_epsilon)
        self.kl_coeff = float(grpo_cfg.kl_coeff)
        self.max_new_tokens = int(grpo_cfg.max_new_tokens)
        self.temperature = float(grpo_cfg.temperature)
        self.top_p = float(grpo_cfg.top_p)
        self.reward_weights = dict(grpo_cfg.reward_weights)

        stage_cfg = getattr(cfg.training, "stage3", None)
        self.grad_accum_steps = int(
            getattr(stage_cfg, "gradient_accumulation_steps", None)
            or getattr(cfg.training, "gradient_accumulation_steps", 1)
        )
        self.amp_dtype = (
            torch.bfloat16
            if getattr(cfg.training, "amp_dtype", "bf16") == "bf16"
            else torch.float16
        )

    def _is_main(self) -> bool:
        return self.rank is None or self.rank == 0

    @torch.no_grad()
    def generate_completions(
        self, prompt_batch: Dict,
    ) -> Dict:
        raw_policy = _raw_model(self.policy_model)
        raw_policy.eval()

        prompt_ids = prompt_batch["prompt_input_ids"].to(self.device)
        prompt_mask = prompt_batch["prompt_attention_mask"].to(self.device)
        pixel_values = prompt_batch["pixel_values"].to(self.device, dtype=self.amp_dtype)
        image_counts = prompt_batch["image_counts"].to(self.device)
        ground_truths = prompt_batch["ground_truths"]
        batch_size = prompt_ids.size(0)
        prompt_len = prompt_ids.size(1)

        G = self.group_size
        expanded_ids = prompt_ids.repeat_interleave(G, dim=0)
        expanded_mask = prompt_mask.repeat_interleave(G, dim=0)
        expanded_pv = pixel_values.repeat_interleave(G, dim=0)
        expanded_ic = image_counts.repeat_interleave(G, dim=0)

        gen_cfg = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=False,
        )

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.device.type == "cuda",
        ):
            gen_ids = raw_policy.generate(
                expanded_ids,
                attention_mask=expanded_mask,
                pixel_values=expanded_pv,
                image_counts=expanded_ic,
                generation_config=gen_cfg,
            )

        response_ids = gen_ids[:, prompt_len:]
        response_texts = self.tokenizer.batch_decode(
            response_ids, skip_special_tokens=True,
        )

        expanded_gts = []
        for gt in ground_truths:
            expanded_gts.extend([gt] * G)

        raw_policy.train()

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "response_ids": response_ids,
            "full_ids": gen_ids,
            "pixel_values": pixel_values,
            "image_counts": image_counts,
            "response_texts": response_texts,
            "ground_truths": expanded_gts,
            "batch_size": batch_size,
            "prompt_len": prompt_len,
        }

    def _compute_token_log_probs(
        self,
        model: nn.Module,
        full_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_counts: torch.Tensor,
        prompt_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = _raw_model(model)
        outputs = raw(
            input_ids=full_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_counts=image_counts,
            labels=None,
        )
        logits = outputs.logits

        shift_logits = logits[:, :-1, :]
        shift_labels = full_ids[:, 1:]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            -1, shift_labels.unsqueeze(-1),
        ).squeeze(-1)

        response_log_probs = token_log_probs[:, prompt_len - 1 :]

        response_ids = full_ids[:, prompt_len:]
        pad_id = self.tokenizer.pad_token_id or 0
        response_mask = (response_ids != pad_id).float()

        return response_log_probs, response_mask

    def compute_grpo_loss(
        self, prompt_batch: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        gen_data = self.generate_completions(prompt_batch)

        response_texts = gen_data["response_texts"]
        ground_truths = gen_data["ground_truths"]
        batch_size = gen_data["batch_size"]
        prompt_len = gen_data["prompt_len"]
        G = self.group_size

        rewards = compute_rewards(
            response_texts, ground_truths, self.reward_weights,
        ).to(self.device)

        rewards_grouped = rewards.view(batch_size, G)
        group_mean = rewards_grouped.mean(dim=1, keepdim=True)
        group_std = rewards_grouped.std(dim=1, keepdim=True)
        advantages = torch.where(
            group_std < 1e-8,
            torch.zeros_like(rewards_grouped),
            (rewards_grouped - group_mean) / (group_std + 1e-8),
        )
        advantages = advantages.flatten()

        full_ids = gen_data["full_ids"].to(self.device)
        full_mask = torch.ones_like(full_ids, dtype=torch.long)
        pad_id = self.tokenizer.pad_token_id or 0
        full_mask[full_ids == pad_id] = 0

        pixel_values = gen_data["pixel_values"].to(
            self.device, dtype=self.amp_dtype,
        )
        image_counts = gen_data["image_counts"].to(self.device)
        expanded_pv = pixel_values.repeat_interleave(G, dim=0)
        expanded_ic = image_counts.repeat_interleave(G, dim=0)

        use_amp = self.device.type == "cuda"

        with torch.no_grad():
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=use_amp,
            ):
                ref_log_probs, response_mask = self._compute_token_log_probs(
                    self.ref_model,
                    full_ids,
                    full_mask,
                    expanded_pv,
                    expanded_ic,
                    prompt_len,
                )

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=use_amp,
        ):
            current_log_probs, response_mask = self._compute_token_log_probs(
                self.policy_model,
                full_ids,
                full_mask,
                expanded_pv,
                expanded_ic,
                prompt_len,
            )

        old_log_probs = current_log_probs.detach()

        log_ratio = current_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)

        advantages_expanded = advantages.unsqueeze(-1).expand_as(ratio)
        surr1 = ratio * advantages_expanded
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
            * advantages_expanded
        )
        policy_loss_per_token = -torch.min(surr1, surr2)

        mask_sum = response_mask.sum().clamp(min=1.0)
        policy_loss = (policy_loss_per_token * response_mask).sum() / mask_sum

        kl_per_token = current_log_probs - ref_log_probs
        kl_loss = (kl_per_token * response_mask).sum() / mask_sum

        total_loss = policy_loss + self.kl_coeff * kl_loss

        acc_rewards = []
        for resp, gt in zip(response_texts, ground_truths):
            from .rewards import accuracy_reward as _acc_r
            acc_rewards.append(_acc_r(resp, gt))

        metrics = {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "mean_reward": rewards.mean().item(),
            "mean_advantage": advantages.mean().item(),
            "accuracy_rate": sum(acc_rewards) / max(len(acc_rewards), 1),
            "mean_response_len": response_mask.sum(dim=1).mean().item(),
        }

        return total_loss, metrics

    def train_epoch(
        self,
        dataloader,
        epoch: int,
        num_epochs: int,
        train_sampler=None,
    ) -> Dict:
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        self.policy_model.train()
        trainable_params = [
            p for p in self.policy_model.parameters() if p.requires_grad
        ]

        accum_metrics = {
            "total_loss": 0.0,
            "policy_loss": 0.0,
            "kl_loss": 0.0,
            "mean_reward": 0.0,
            "accuracy_rate": 0.0,
        }
        n_batches = 0
        pending = 0

        self.optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(
            dataloader,
            desc=f"GRPO Epoch {epoch + 1}/{num_epochs}",
            disable=not self._is_main(),
        )

        for batch in pbar:
            if batch is None:
                continue

            loss, metrics = self.compute_grpo_loss(batch)
            (loss / self.grad_accum_steps).backward()

            for k in accum_metrics:
                if k in metrics:
                    accum_metrics[k] += metrics[k]
            n_batches += 1
            pending += 1

            if pending == self.grad_accum_steps:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                pending = 0

            if self._is_main():
                pbar.set_postfix(
                    loss=accum_metrics["total_loss"] / max(n_batches, 1),
                    reward=accum_metrics["mean_reward"] / max(n_batches, 1),
                    acc=accum_metrics["accuracy_rate"] / max(n_batches, 1),
                    accum=f"{pending}/{self.grad_accum_steps}",
                )

        if pending > 0:
            if pending != self.grad_accum_steps:
                scale = self.grad_accum_steps / pending
                for param in trainable_params:
                    if param.grad is not None:
                        param.grad.mul_(scale)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        self.scheduler.step()

        avg_metrics = {
            k: v / max(n_batches, 1) for k, v in accum_metrics.items()
        }

        if dist.is_initialized():
            for k, v in avg_metrics.items():
                t = torch.tensor(v, device=self.device)
                dist.all_reduce(t, op=dist.ReduceOp.AVG)
                avg_metrics[k] = t.item()

        return avg_metrics

    def train(
        self,
        dataloader,
        num_epochs: int,
        train_sampler=None,
        tokenizer=None,
        save_fn=None,
    ):
        for epoch in range(num_epochs):
            metrics = self.train_epoch(
                dataloader, epoch, num_epochs, train_sampler,
            )

            if self._is_main():
                msg = (
                    f"GRPO Epoch [{epoch + 1}/{num_epochs}] "
                    f"Loss: {metrics['total_loss']:.4f} "
                    f"Policy: {metrics['policy_loss']:.4f} "
                    f"KL: {metrics['kl_loss']:.4f} "
                    f"Reward: {metrics['mean_reward']:.4f} "
                    f"Acc: {metrics['accuracy_rate']:.4f}"
                )
                print(msg)
                logging.info(msg)

            if save_fn is not None:
                save_fn(epoch)

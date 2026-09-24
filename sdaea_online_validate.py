#!/usr/bin/env python3
"""
SDAEA 的单智能体、在线内稳态学习验证版。

这个文件刻意只验证讨论中最核心的假设：

1. Godot 返回的 reward 被完全丢弃，不参与任何参数更新。
2. 原始内驱信号只来自电量变化、低电量、存活状态和死亡：
   - 电量处境改善 -> pleasure teaching signal
   - 电量处境恶化 -> pain teaching signal
   - 活着的每一步 -> 很小的 tonic pleasure（使目标在数学上真的是延长生命）
   - 死亡 -> 很强的 pain teaching signal
3. pleasure/pain 网络学习预测未来的内稳态结果，而不是被 loss 无条件推大/推小。
4. 动作网络在智能体活着时每一步都更新。
5. 死亡不保存轨迹、不回放、不恢复旧世界；死亡误差通过 eligibility trace
   作用于仍带有“突触标记”的近期动作。当前 Godot 环境只有一个智能体，因此死亡后
   调用一次全局 reset 只用于复活这个唯一智能体。
6. 环境不会告诉模型红色、绿色、墙或食物的语义。视觉含义只能从
   “看见什么 -> 做了什么 -> 后续电量/死亡怎样变化”中学到。

这不是完整的 AGI 方案，也不是种群进化层。它是一个可证伪的第一阶段实验：
与随机/冻结策略比较，观察在线学习后的寿命分布是否上升。

典型用法（使用 Godot 编辑器）：

    # 先运行本文件；看到提示后，在 Godot 编辑器中按 Play
    python sdaea_online_validate.py --max-steps 100000

    # 不连接 Godot，只检查更新方向、内驱信号和网络梯度
    python sdaea_online_validate.py --self-test

    # 随机且不学习的对照组
    python sdaea_online_validate.py --no-learn --random-actions --max-steps 20000

    # 去掉“额外死亡痛苦”的消融组（电量归零本身仍会产生生理痛苦）
    python sdaea_online_validate.py --death-pain 0 --max-steps 100000

    # 分离“世界表征学习”和“内稳态行为学习”的两个对照
    python sdaea_online_validate.py --freeze-behavior --max-steps 20000
    python sdaea_online_validate.py --freeze-world --max-steps 20000

    # 从某个快照做新的冻结测试；这是 warm-start，不会恢复旧 Godot 世界
    python sdaea_online_validate.py --warm-start CHECKPOINT --no-learn --deterministic

注意：
Godot RL 0.8.x 当前把同一个 done 同时作为 terminated 和 truncated 返回。本验证环境
没有时间截断，所以这里把 done 解释为死亡。如果以后加入时间上限，必须在 Godot 端
把真正死亡与时间截断分开。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import signal
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


@dataclass(frozen=True)
class ActionSpec:
    """One discrete Godot action head, in the exact order expected by Godot."""

    name: str
    size: int
    learned: bool = True
    fixed_value: int = 0


@dataclass
class WorldPrediction:
    next_latent: torch.Tensor
    next_visual_logits: torch.Tensor
    hp_delta: torch.Tensor
    death_logit: torch.Tensor


@dataclass(frozen=True)
class IntrinsicOutcome:
    pleasure: float
    pain: float
    drive_before: float
    drive_after: float
    low_hp_urgency: float


@dataclass
class UpdateStats:
    trace_norm: float = 0.0
    direction_norm: float = 0.0
    applied_update_norm: float = 0.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("--device mps was requested, but MPS is unavailable.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _unwrap_singleton(value):
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return value


def decode_eye(
    value,
    expected_height: int,
    expected_width: int,
    key: str,
) -> np.ndarray:
    """Decode a Godot eye observation to float32 HWC RGB in [0, 1]."""

    value = _unwrap_singleton(value)
    if isinstance(value, bytes):
        raw = value
        array = np.frombuffer(raw, dtype=np.uint8)
    elif isinstance(value, str):
        text = value[2:] if value.startswith("0x") else value
        try:
            raw = bytes.fromhex(text)
        except ValueError as exc:
            raise ValueError(f"Observation {key!r} is not a valid hexadecimal image.") from exc
        array = np.frombuffer(raw, dtype=np.uint8)
    else:
        array = np.asarray(value)

    if array.ndim == 1:
        expected = expected_height * expected_width * 3
        if array.size != expected:
            raise ValueError(
                f"Observation {key!r} has {array.size} values; expected "
                f"{expected_height}*{expected_width}*3={expected}. "
                "Adjust --eye-height/--eye-width."
            )
        array = array.reshape(expected_height, expected_width, 3)
    elif array.ndim == 4 and array.shape[0] == 1:
        array = array[0]

    if array.ndim != 3:
        raise ValueError(f"Observation {key!r} must be HWC/CHW RGB; got shape {array.shape}.")
    if array.shape[-1] == 3:
        pass
    elif array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    else:
        raise ValueError(f"Observation {key!r} has no RGB channel dimension: {array.shape}.")

    array = np.asarray(array, dtype=np.float32)
    if array.size and float(np.nanmax(array)) > 1.0:
        array = array / 255.0
    return np.ascontiguousarray(np.clip(array, 0.0, 1.0))


def observation_to_tensor(
    observation: Dict,
    left_eye_key: str,
    right_eye_key: str,
    expected_height: int,
    expected_width: int,
    image_size: int,
    device: torch.device,
) -> torch.Tensor:
    missing = [k for k in (left_eye_key, right_eye_key) if k not in observation]
    if missing:
        raise KeyError(
            f"Missing eye observation(s) {missing}; available keys: {list(observation.keys())}"
        )
    left = decode_eye(observation[left_eye_key], expected_height, expected_width, left_eye_key)
    right = decode_eye(observation[right_eye_key], expected_height, expected_width, right_eye_key)
    left_t = torch.from_numpy(left).permute(2, 0, 1)
    right_t = torch.from_numpy(right).permute(2, 0, 1)
    eyes = torch.cat((left_t, right_t), dim=0).unsqueeze(0).to(device=device)
    if eyes.shape[-2:] != (image_size, image_size):
        eyes = F.interpolate(
            eyes,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
    return eyes.contiguous()


def extract_hp(observation: Dict, hp_key: str) -> float:
    if hp_key not in observation:
        raise KeyError(f"Missing HP key {hp_key!r}; available keys: {list(observation.keys())}")
    values = np.asarray(_unwrap_singleton(observation[hp_key]), dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError(f"Observation {hp_key!r} is empty.")
    hp = float(values[0])
    if not math.isfinite(hp):
        raise ValueError(f"Observation {hp_key!r} is not finite: {hp}")
    return hp


class VisualEncoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projection(self.conv(image))


class ActorHead(nn.Module):
    def __init__(self, hidden_dim: int, specs: Sequence[ActionSpec]) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, spec.size) for spec in specs])
        # An initially uniform policy makes the first experiment easier to interpret.
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, hidden: torch.Tensor) -> List[torch.Tensor]:
        feature = self.trunk(hidden)
        return [head(feature) for head in self.heads]

    def learning_parameters(self, specs: Sequence[ActionSpec]) -> List[nn.Parameter]:
        params: List[nn.Parameter] = list(self.trunk.parameters())
        for spec, head in zip(specs, self.heads):
            if spec.learned:
                params.extend(head.parameters())
        return params


class AffectHead(nn.Module):
    """Predict discounted future pleasure and pain from the current internal state."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.output = nn.Linear(hidden_dim, 2)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.01)
        initial_value = 0.1
        inverse_softplus = math.log(math.expm1(initial_value))
        nn.init.constant_(self.output.bias, inverse_softplus)

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = self.output(self.trunk(hidden))
        positive = F.softplus(raw)
        return positive[:, 0], positive[:, 1]


class OnlineHomeostaticAgent(nn.Module):
    """
    Four deliberately separated parameter groups:

    - encoder/core/dynamics: world model, trained by sensory and body prediction
    - actor: actions, trained by online TD eligibility traces
    - affect: predicted pleasure/pain, trained by online TD eligibility traces
    - target modules: slow copies used only for stable bootstrap targets
    """

    def __init__(
        self,
        specs: Sequence[ActionSpec],
        latent_dim: int = 128,
        hidden_dim: int = 192,
        visual_prediction_size: int = 8,
    ) -> None:
        super().__init__()
        self.specs = list(specs)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.visual_prediction_size = visual_prediction_size
        self.action_vector_dim = sum(spec.size for spec in specs)
        self.visual_prediction_dim = 6 * visual_prediction_size * visual_prediction_size
        self.feedback_dim = (
            hidden_dim
            + self.action_vector_dim
            + 2
            + latent_dim
            + self.visual_prediction_dim
            + 2
        )

        self.encoder = VisualEncoder(latent_dim)
        recurrent_input = latent_dim + 1 + self.action_vector_dim + self.feedback_dim
        self.core = nn.GRUCell(recurrent_input, hidden_dim)
        self.actor = ActorHead(hidden_dim, specs)
        self.affect = AffectHead(hidden_dim)

        dynamics_input = hidden_dim + self.action_vector_dim
        self.dynamics_trunk = nn.Sequential(
            nn.Linear(dynamics_input, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.next_latent_head = nn.Linear(hidden_dim, latent_dim)
        self.next_visual_head = nn.Linear(hidden_dim, self.visual_prediction_dim)
        self.hp_delta_head = nn.Linear(hidden_dim, 1)
        self.death_head = nn.Linear(hidden_dim, 1)

        # Targets are initialized exactly, frozen, then moved by EMA only.
        import copy

        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_affect = copy.deepcopy(self.affect)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.target_affect.parameters():
            parameter.requires_grad_(False)

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def initial_feedback(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.feedback_dim, device=device)

    def actions_to_onehot(
        self,
        actions: Sequence[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if len(actions) != len(self.specs):
            raise ValueError(f"Expected {len(self.specs)} actions, got {len(actions)}.")
        pieces = []
        for action, spec in zip(actions, self.specs):
            if action < 0 or action >= spec.size:
                raise ValueError(f"Action {action} is outside {spec.name}[0,{spec.size}).")
            index = torch.tensor([action], device=device)
            pieces.append(F.one_hot(index, num_classes=spec.size).to(dtype=dtype))
        return torch.cat(pieces, dim=-1)

    def encode_state(
        self,
        image: torch.Tensor,
        hp_normalized: float,
        previous_actions: Sequence[int],
        previous_hidden: torch.Tensor,
        previous_feedback: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(image)
        action_vector = self.actions_to_onehot(
            previous_actions,
            device=image.device,
            dtype=latent.dtype,
        )
        # Preserve values above nominal full charge while keeping recurrent input bounded.
        hp_signal = torch.tensor(
            [[math.tanh(hp_normalized)]],
            device=image.device,
            dtype=latent.dtype,
        )
        recurrent_input = torch.cat(
            (latent, hp_signal, action_vector, previous_feedback),
            dim=-1,
        )
        hidden = self.core(recurrent_input, previous_hidden)
        return latent, hidden

    def heads(
        self, hidden: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        # Numeric recurrent state remains available, while detach separates plasticity rules.
        detached_hidden = hidden.detach()
        logits = self.actor(detached_hidden)
        pleasure, pain = self.affect(detached_hidden)
        return logits, pleasure, pain

    def predict_world(
        self,
        hidden: torch.Tensor,
        actions: Sequence[int],
    ) -> WorldPrediction:
        action_vector = self.actions_to_onehot(
            actions,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        feature = self.dynamics_trunk(torch.cat((hidden, action_vector), dim=-1))
        return WorldPrediction(
            next_latent=self.next_latent_head(feature),
            next_visual_logits=self.next_visual_head(feature),
            hp_delta=self.hp_delta_head(feature).squeeze(-1),
            death_logit=self.death_head(feature).squeeze(-1),
        )

    def make_feedback(
        self,
        hidden: torch.Tensor,
        logits: Sequence[torch.Tensor],
        pleasure: torch.Tensor,
        pain: torch.Tensor,
        prediction: WorldPrediction,
    ) -> torch.Tensor:
        """
        Feed a bounded compressed form of every model output into the next step.

        This provides numerical recurrent memory. The prediction objectives determine
        whether that recurrent channel learns useful content; recurrence alone does not.
        """

        values = [
            hidden,
            *logits,
            pleasure.unsqueeze(-1),
            pain.unsqueeze(-1),
            prediction.next_latent,
            torch.sigmoid(prediction.next_visual_logits),
            prediction.hp_delta.unsqueeze(-1),
            torch.sigmoid(prediction.death_logit).unsqueeze(-1),
        ]
        feedback = torch.tanh(torch.cat(values, dim=-1))
        if feedback.shape[-1] != self.feedback_dim:
            raise RuntimeError(
                f"Internal feedback shape {feedback.shape[-1]} != {self.feedback_dim}."
            )
        return feedback

    def world_parameters(self) -> List[nn.Parameter]:
        modules: Sequence[nn.Module] = (
            self.encoder,
            self.core,
            self.dynamics_trunk,
            self.next_latent_head,
            self.next_visual_head,
            self.hp_delta_head,
            self.death_head,
        )
        return [parameter for module in modules for parameter in module.parameters()]

    def actor_parameters(self) -> List[nn.Parameter]:
        return self.actor.learning_parameters(self.specs)

    def affect_parameters(self) -> List[nn.Parameter]:
        return list(self.affect.parameters())

    @torch.no_grad()
    def update_targets(self, decay: float) -> None:
        for target, source in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            target.mul_(decay).add_(source, alpha=1.0 - decay)
        for target, source in zip(self.target_affect.parameters(), self.affect.parameters()):
            target.mul_(decay).add_(source, alpha=1.0 - decay)


class MultiTimescaleEligibility:
    """
    A decaying parameter-local tag, not a trajectory buffer.

    Each entry has exactly the shape of one parameter. It stores no observations,
    frames, actions, world states or autograd graphs.
    """

    def __init__(
        self,
        parameters: Sequence[nn.Parameter],
        decays_and_weights: Sequence[Tuple[float, float]],
        max_norm: float,
    ) -> None:
        self.parameters = list(parameters)
        self.decays_and_weights = list(decays_and_weights)
        self.max_norm = max_norm
        self.traces: List[List[torch.Tensor]] = [
            [torch.zeros_like(parameter, dtype=torch.float32) for parameter in self.parameters]
            for _ in self.decays_and_weights
        ]

    @torch.no_grad()
    def accumulate(self, gradients: Sequence[Optional[torch.Tensor]]) -> None:
        if len(gradients) != len(self.parameters):
            raise ValueError("Gradient/parameter length mismatch in eligibility trace.")
        for (decay, _weight), trace_set in zip(self.decays_and_weights, self.traces):
            for trace, gradient in zip(trace_set, gradients):
                trace.mul_(decay)
                if gradient is not None:
                    if not bool(torch.isfinite(gradient).all().item()):
                        raise FloatingPointError("Non-finite gradient entered an eligibility trace.")
                    trace.add_(gradient.detach().to(dtype=torch.float32))
            self._clip_trace_set(trace_set)

    @torch.no_grad()
    def _clip_trace_set(self, trace_set: Sequence[torch.Tensor]) -> None:
        norm = tensor_list_norm(trace_set)
        if not math.isfinite(norm):
            raise FloatingPointError("Eligibility trace norm became non-finite.")
        if norm > self.max_norm:
            scale = self.max_norm / (norm + 1e-12)
            for trace in trace_set:
                trace.mul_(scale)

    @torch.no_grad()
    def combined(self) -> List[torch.Tensor]:
        result = [torch.zeros_like(parameter, dtype=torch.float32) for parameter in self.parameters]
        for (_decay, weight), trace_set in zip(self.decays_and_weights, self.traces):
            for output, trace in zip(result, trace_set):
                output.add_(trace, alpha=weight)
        return result

    @torch.no_grad()
    def norm(self) -> float:
        return tensor_list_norm(self.combined())

    @torch.no_grad()
    def clear(self) -> None:
        for trace_set in self.traces:
            for trace in trace_set:
                trace.zero_()


def tensor_list_norm(tensors: Iterable[Optional[torch.Tensor]]) -> float:
    total = 0.0
    for tensor in tensors:
        if tensor is not None:
            total += float(torch.sum(tensor.detach().float() ** 2).item())
    return math.sqrt(total)


@torch.no_grad()
def apply_manual_ascent(
    parameters: Sequence[nn.Parameter],
    direction: Sequence[torch.Tensor],
    learning_rate: float,
    max_direction_norm: float,
    trace_norm: float,
) -> UpdateStats:
    raw_norm = tensor_list_norm(direction)
    if not math.isfinite(raw_norm):
        raise FloatingPointError("Manual update direction became non-finite.")
    scale = min(1.0, max_direction_norm / (raw_norm + 1e-12))
    for parameter, update in zip(parameters, direction):
        parameter.add_(
            update.to(device=parameter.device, dtype=parameter.dtype),
            alpha=learning_rate * scale,
        )
    return UpdateStats(
        trace_norm=trace_norm,
        direction_norm=raw_norm,
        applied_update_norm=learning_rate * raw_norm * scale,
    )


def _stable_softplus(value: float) -> float:
    return float(np.logaddexp(0.0, value))


def homeostatic_drive(hp_normalized: float, safe_hp: float, temperature: float) -> float:
    return temperature * _stable_softplus((safe_hp - hp_normalized) / temperature)


def intrinsic_outcome(
    hp_before: float,
    hp_after: float,
    terminal: bool,
    safe_hp: float,
    drive_temperature: float,
    low_hp_threshold: float,
    low_hp_weight: float,
    alive_pleasure: float,
    death_pain: float,
) -> IntrinsicOutcome:
    """
    Primitive internal teaching signal. It never inspects colors, collisions or env reward.
    HP values are normalized by --hp-scale before reaching this function.
    """

    before = homeostatic_drive(hp_before, safe_hp, drive_temperature)
    after = homeostatic_drive(hp_after, safe_hp, drive_temperature)
    relief = max(before - after, 0.0)
    worsening = max(after - before, 0.0)
    if low_hp_threshold > 0:
        low_fraction = max(low_hp_threshold - hp_after, 0.0) / low_hp_threshold
    else:
        low_fraction = 0.0
    urgency = low_hp_weight * low_fraction * low_fraction
    pleasure = 0.0 if terminal else relief + alive_pleasure
    pain = worsening + urgency + (death_pain if terminal else 0.0)
    return IntrinsicOutcome(
        pleasure=pleasure,
        pain=pain,
        drive_before=before,
        drive_after=after,
        low_hp_urgency=urgency,
    )


def choose_actions(
    logits: Sequence[torch.Tensor],
    specs: Sequence[ActionSpec],
    deterministic: bool,
    random_actions: bool,
) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
    actions: List[int] = []
    log_prob_terms: List[torch.Tensor] = []
    entropy_terms: List[torch.Tensor] = []
    for head_logits, spec in zip(logits, specs):
        distribution = Categorical(logits=head_logits)
        if not spec.learned:
            action = spec.fixed_value
        elif random_actions:
            action = random.randrange(spec.size)
        elif deterministic:
            action = int(torch.argmax(head_logits, dim=-1).item())
        else:
            action = int(distribution.sample().item())
        actions.append(action)
        if spec.learned:
            action_tensor = torch.tensor([action], device=head_logits.device)
            log_prob_terms.append(distribution.log_prob(action_tensor))
            entropy_terms.append(distribution.entropy())
    if not log_prob_terms:
        raise RuntimeError("At least one action head must be learned in this validation.")
    return (
        actions,
        torch.stack(log_prob_terms, dim=0).sum(),
        torch.stack(entropy_terms, dim=0).sum(),
    )


def selected_log_prob(
    logits: Sequence[torch.Tensor],
    specs: Sequence[ActionSpec],
    actions: Sequence[int],
) -> torch.Tensor:
    terms = []
    for head_logits, spec, action in zip(logits, specs, actions):
        if spec.learned:
            action_tensor = torch.tensor([action], device=head_logits.device)
            terms.append(Categorical(logits=head_logits).log_prob(action_tensor))
    if not terms:
        raise RuntimeError("At least one action head must be learned.")
    return torch.stack(terms, dim=0).sum()


def detached_gradients(
    output: torch.Tensor,
    parameters: Sequence[nn.Parameter],
    retain_graph: bool,
) -> List[Optional[torch.Tensor]]:
    gradients = torch.autograd.grad(
        output,
        parameters,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )
    return [None if gradient is None else gradient.detach() for gradient in gradients]


def world_losses(
    prediction: WorldPrediction,
    target_next_latent: Optional[torch.Tensor],
    target_next_visual: Optional[torch.Tensor],
    target_hp_delta: float,
    terminal: bool,
    latent_weight: float,
    visual_weight: float,
    hp_weight: float,
    death_weight: float,
    death_positive_weight: float,
    death_focal_gamma: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    zero = prediction.hp_delta.sum() * 0.0
    if terminal:
        latent_loss = zero
        visual_loss = zero
    else:
        if target_next_latent is None or target_next_visual is None:
            raise ValueError("Non-terminal world loss requires next-state targets.")
        latent_loss = F.smooth_l1_loss(prediction.next_latent, target_next_latent)
        predicted_visual = torch.sigmoid(prediction.next_visual_logits).view_as(target_next_visual)
        visual_loss = F.smooth_l1_loss(predicted_visual, target_next_visual)
    hp_target = torch.tensor(
        [target_hp_delta],
        device=prediction.hp_delta.device,
        dtype=prediction.hp_delta.dtype,
    )
    hp_loss = F.smooth_l1_loss(prediction.hp_delta, hp_target)
    death_target = torch.tensor(
        [float(terminal)],
        device=prediction.death_logit.device,
        dtype=prediction.death_logit.dtype,
    )
    positive_weight = torch.tensor(
        [death_positive_weight],
        device=prediction.death_logit.device,
        dtype=prediction.death_logit.dtype,
    )
    death_bce = F.binary_cross_entropy_with_logits(
        prediction.death_logit,
        death_target,
        pos_weight=positive_weight,
        reduction="none",
    )
    death_probability = torch.sigmoid(prediction.death_logit)
    correct_class_probability = (
        death_target * death_probability
        + (1.0 - death_target) * (1.0 - death_probability)
    )
    death_loss = (
        (1.0 - correct_class_probability).pow(death_focal_gamma) * death_bce
    ).mean()
    total = (
        latent_weight * latent_loss
        + visual_weight * visual_loss
        + hp_weight * hp_loss
        + death_weight * death_loss
    )
    parts = {
        "world_loss": float(total.detach().item()),
        "latent_loss": float(latent_loss.detach().item()),
        "visual_loss": float(visual_loss.detach().item()),
        "hp_loss": float(hp_loss.detach().item()),
        "death_loss": float(death_loss.detach().item()),
    }
    return total, parts


class CsvMetricLogger:
    FIELDS = [
        "step",
        "life",
        "life_step",
        "hp",
        "hp_normalized",
        "next_hp",
        "next_hp_reported",
        "next_hp_normalized",
        "hp_delta",
        "action",
        "pleasure_signal",
        "pain_signal",
        "drive_before",
        "drive_after",
        "low_hp_urgency",
        "alive_pleasure_component",
        "predicted_pleasure",
        "predicted_pain",
        "next_predicted_pleasure",
        "next_predicted_pain",
        "delta_pleasure_raw",
        "delta_pain_raw",
        "delta_pleasure_used",
        "delta_pain_used",
        "policy_delta",
        "td_clip_hit",
        "entropy",
        "selected_log_prob",
        "selected_log_prob_change",
        "current_action_sign_ok",
        "actor_trace_norm",
        "actor_direction_norm",
        "actor_update_norm",
        "actor_direction_clip_hit",
        "pleasure_trace_norm",
        "pain_trace_norm",
        "affect_direction_norm",
        "affect_update_norm",
        "affect_direction_clip_hit",
        "world_loss",
        "latent_loss",
        "visual_loss",
        "hp_loss",
        "death_loss",
        "world_grad_norm",
        "world_grad_clip_hit",
        "predicted_hp_delta",
        "predicted_death_probability",
        "terminal",
        "terminal_source",
        "done_with_positive_hp",
        "deaths",
        "env_reward_ignored",
    ]

    def __init__(self, path: Path, append: bool = False, flush_every: int = 50) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        already_has_data = append and path.exists() and path.stat().st_size > 0
        self.file = path.open("a" if append else "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDS)
        self.flush_every = max(1, flush_every)
        self.rows_since_flush = 0
        if not already_has_data:
            self.writer.writeheader()
            self.file.flush()

    def write(self, row: Dict, force_flush: bool = False) -> None:
        self.writer.writerow({field: row.get(field, "") for field in self.FIELDS})
        self.rows_since_flush += 1
        if force_flush or self.rows_since_flush >= self.flush_every:
            self.file.flush()
            self.rows_since_flush = 0

    def close(self) -> None:
        self.file.close()


def extract_action_specs(env, learn_shoot: bool) -> List[ActionSpec]:
    if env.num_envs != 1:
        raise RuntimeError(
            "This validation file intentionally supports exactly one Godot agent. "
            f"The connected environment reported n_agents={env.num_envs}. "
            "A multi-agent version needs per-agent respawn instead of global env.reset()."
        )
    action_space = env.action_spaces[0]
    specs = []
    for name, space in action_space.spaces.items():
        size = getattr(space, "n", None)
        if size is None:
            raise TypeError(
                f"Action {name!r} is not discrete. This first validation supports only "
                "Godot Discrete action heads."
            )
        is_shoot = "shoot" in name.lower()
        specs.append(
            ActionSpec(
                name=name,
                size=int(size),
                learned=(learn_shoot or not is_shoot),
                fixed_value=0,
            )
        )
    return specs


def parameter_group_sanity(agent: OnlineHomeostaticAgent) -> None:
    groups = {
        "world": {id(parameter) for parameter in agent.world_parameters()},
        "actor": {id(parameter) for parameter in agent.actor_parameters()},
        "affect": {id(parameter) for parameter in agent.affect_parameters()},
        "target": {
            id(parameter)
            for module in (agent.target_encoder, agent.target_affect)
            for parameter in module.parameters()
        },
    }
    names = list(groups)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = groups[left] & groups[right]
            if overlap:
                raise RuntimeError(f"Parameter groups {left}/{right} overlap ({len(overlap)} tensors).")
    for name, identifiers in groups.items():
        if not identifiers:
            raise RuntimeError(f"Parameter group {name} is empty.")


def safe_env_reward(rewards) -> float:
    try:
        return float(np.asarray(rewards, dtype=np.float32).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return float("nan")


def atomic_torch_save(payload: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_checkpoint(
    run_dir: Path,
    agent: OnlineHomeostaticAgent,
    optimizer: torch.optim.Optimizer,
    specs: Sequence[ActionSpec],
    args: argparse.Namespace,
    step: int,
    life: int,
    life_step: int,
    deaths: int,
    lifetimes: Sequence[int],
    numbered: bool,
    extra_name: Optional[str] = None,
) -> Path:
    payload = {
        "format_version": 1,
        "model": agent.state_dict(),
        "world_optimizer": optimizer.state_dict(),
        "action_specs": [asdict(spec) for spec in specs],
        "args": vars(args),
        "step": step,
        "life": life,
        "life_step": life_step,
        "deaths": deaths,
        "lifetimes": list(lifetimes),
    }
    checkpoint_dir = run_dir / "checkpoints"
    latest = checkpoint_dir / "latest.pt"
    atomic_torch_save(payload, latest)
    if numbered:
        atomic_torch_save(payload, checkpoint_dir / f"step_{step:09d}.pt")
    if extra_name is not None:
        atomic_torch_save(payload, checkpoint_dir / extra_name)
    return latest


def load_checkpoint(
    path: Path,
    agent: OnlineHomeostaticAgent,
    optimizer: torch.optim.Optimizer,
    specs: Sequence[ActionSpec],
    device: torch.device,
    load_optimizer: bool,
) -> Dict:
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("format_version") != 1:
        raise RuntimeError(
            f"Unsupported checkpoint format_version={checkpoint.get('format_version')!r}."
        )
    saved_specs = checkpoint.get("action_specs")
    current_specs = [asdict(spec) for spec in specs]
    if saved_specs != current_specs:
        raise RuntimeError(
            "Checkpoint action heads do not match the connected Godot environment.\n"
            f"saved={saved_specs}\ncurrent={current_specs}"
        )
    agent.load_state_dict(checkpoint["model"])
    if load_optimizer:
        optimizer.load_state_dict(checkpoint["world_optimizer"])
    return checkpoint


def build_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        return Path(args.run_dir).expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (Path.cwd() / "runs" / f"sdaea_online_{stamp}").resolve()


def make_target_visual(next_image: torch.Tensor, size: int) -> torch.Tensor:
    return F.adaptive_avg_pool2d(next_image, (size, size)).flatten(start_dim=1).detach()


def train(args: argparse.Namespace) -> None:
    behavior_learning = not args.no_learn and not args.freeze_behavior
    world_learning = not args.no_learn and not args.freeze_world
    if args.random_actions and behavior_learning:
        raise ValueError(
            "--random-actions cannot be used while actor/affect learning is enabled."
        )
    if args.deterministic and behavior_learning:
        raise ValueError(
            "--deterministic uses argmax and is evaluation-only. "
            "Use it with --no-learn or --freeze-behavior."
        )
    if args.warm_start_optimizer and not args.warm_start:
        raise ValueError("--warm-start-optimizer requires --warm-start CHECKPOINT.")
    if args.hp_scale <= 0:
        raise ValueError("--hp-scale must be positive.")
    if not 0.0 <= args.target_ema < 1.0:
        raise ValueError("--target-ema must be in [0, 1).")
    if not 0.0 <= args.gamma <= 1.0:
        raise ValueError("--gamma must be in [0, 1].")
    for name in ("actor_lambda_fast", "actor_lambda_slow", "affect_lambda"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1].")
    if not 0.0 <= args.actor_trace_fast_weight <= 1.0:
        raise ValueError("--actor-trace-fast-weight must be in [0, 1].")
    if args.drive_temperature <= 0:
        raise ValueError("--drive-temperature must be positive.")
    if args.td_clip <= 0:
        raise ValueError("--td-clip must be positive.")
    if args.death_positive_weight <= 0:
        raise ValueError("--death-positive-weight must be positive.")
    if args.death_focal_gamma < 0:
        raise ValueError("--death-focal-gamma cannot be negative.")
    if min(
        args.eye_height,
        args.eye_width,
        args.image_size,
        args.visual_prediction_size,
        args.latent_dim,
        args.hidden_dim,
    ) <= 0:
        raise ValueError("Image and model dimensions must all be positive.")

    # Lazy import lets --self-test run without opening a Godot socket.
    from godot_rl.core.godot_env import GodotEnv

    seed_everything(args.seed)
    device = choose_device(args.device)
    run_dir = build_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as config_file:
        json.dump(vars(args), config_file, indent=2, ensure_ascii=False, default=str)

    env = None
    logger = None
    interrupted = False
    try:
        print(f"[SDAEA] device={device}; run_dir={run_dir}")
        env = GodotEnv(
            env_path=args.env_path,
            port=args.port,
            show_window=args.show_window,
            seed=args.seed,
            framerate=args.framerate,
            action_repeat=args.action_repeat,
            speedup=args.speedup,
        )
        specs = extract_action_specs(env, args.learn_shoot)
        print("[SDAEA] exact Godot action order:")
        for index, spec in enumerate(specs):
            mode = "learn" if spec.learned else f"fixed={spec.fixed_value}"
            print(f"  {index}: {spec.name} (Discrete({spec.size}), {mode})")
        print(
            "[SDAEA] Godot reward is ignored; actor/affect reinforcement comes from "
            "HP/death only, while the world model uses self-supervised sensory prediction."
        )
        print(
            f"[SDAEA] entropy coefficient={args.entropy_coefficient} is an exploration "
            "regularizer, not an environment reward (set it to 0 for an ablation)."
        )
        print(
            "[SDAEA] limitation: the current one-agent Godot API uses a global reset "
            "after death. It starts a new physical world; it does not replay or rewind "
            "that world for gradient learning."
        )
        if args.env_path is None:
            print(
                "[SDAEA] editor mode: --seed fixes Python/PyTorch, but the current "
                "Godot connector cannot seed an already-running editor scene."
            )

        agent = OnlineHomeostaticAgent(
            specs=specs,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            visual_prediction_size=args.visual_prediction_size,
        ).to(device)
        parameter_group_sanity(agent)
        world_parameters = agent.world_parameters()
        actor_parameters = agent.actor_parameters()
        affect_parameters = agent.affect_parameters()
        world_optimizer = torch.optim.Adam(
            world_parameters,
            lr=args.world_lr,
            weight_decay=args.world_weight_decay,
        )

        life = 1
        life_step = 0
        deaths = 0
        lifetimes: List[int] = []
        if args.warm_start:
            checkpoint = load_checkpoint(
                Path(args.warm_start).expanduser().resolve(),
                agent,
                world_optimizer,
                specs,
                device,
                load_optimizer=args.warm_start_optimizer,
            )
            if args.warm_start_optimizer:
                # The command line is the resolved configuration for this new run.
                for parameter_group in world_optimizer.param_groups:
                    parameter_group["lr"] = args.world_lr
                    parameter_group["weight_decay"] = args.world_weight_decay
            print(
                f"[SDAEA] warm-started parameters from source step="
                f"{int(checkpoint.get('step', 0))}. This is a new run: Godot state, "
                "recurrent memory, eligibility tags, RNG state and lifetime counters "
                "are intentionally not restored."
            )

        actor_trace = MultiTimescaleEligibility(
            actor_parameters,
            decays_and_weights=(
                (args.gamma * args.actor_lambda_fast, args.actor_trace_fast_weight),
                (args.gamma * args.actor_lambda_slow, 1.0 - args.actor_trace_fast_weight),
            ),
            max_norm=args.max_trace_norm,
        )
        pleasure_trace = MultiTimescaleEligibility(
            affect_parameters,
            decays_and_weights=((args.gamma * args.affect_lambda, 1.0),),
            max_norm=args.max_trace_norm,
        )
        pain_trace = MultiTimescaleEligibility(
            affect_parameters,
            decays_and_weights=((args.gamma * args.affect_lambda, 1.0),),
            max_norm=args.max_trace_norm,
        )

        logger = CsvMetricLogger(
            run_dir / "metrics.csv",
            append=False,
            flush_every=args.metrics_flush_every,
        )
        observations, _reset_info = env.reset()
        observation = observations[0]
        hp_raw = extract_hp(observation, args.hp_key)
        initial_hp_normalized = hp_raw / args.hp_scale
        print(
            f"[SDAEA] initial raw HP={hp_raw:.4f}; normalized="
            f"{initial_hp_normalized:.4f} using --hp-scale={args.hp_scale}."
        )
        if not 0.2 <= initial_hp_normalized <= 2.0:
            print(
                "[SDAEA] WARNING: initial normalized HP is unusual. Verify --hp-scale "
                "against the Godot max/base HP before interpreting affect magnitudes."
            )
        hidden = agent.initial_hidden(1, device)
        feedback = agent.initial_feedback(1, device)
        previous_actions = [0 for _ in specs]
        learning_enabled = behavior_learning or world_learning
        agent.train(learning_enabled)
        save_checkpoint(
            run_dir,
            agent,
            world_optimizer,
            specs,
            args,
            step=0,
            life=life,
            life_step=0,
            deaths=0,
            lifetimes=lifetimes,
            numbered=True,
        )
        print(
            f"[SDAEA] step-0 checkpoint: "
            f"{run_dir / 'checkpoints' / 'step_000000000.pt'}"
        )

        stop_requested = False

        def request_stop(_signum, _frame) -> None:
            nonlocal stop_requested
            stop_requested = True

        previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
        final_step = 0
        try:
            for step in range(1, args.max_steps + 1):
                final_step = step
                if stop_requested:
                    interrupted = True
                    break
                life_step += 1
                hp_normalized = hp_raw / args.hp_scale
                image = observation_to_tensor(
                    observation,
                    args.left_eye_key,
                    args.right_eye_key,
                    args.eye_height,
                    args.eye_width,
                    args.image_size,
                    device,
                )

                with torch.set_grad_enabled(learning_enabled):
                    _latent, current_hidden = agent.encode_state(
                        image,
                        hp_normalized,
                        previous_actions,
                        hidden,
                        feedback,
                    )
                    logits, predicted_pleasure, predicted_pain = agent.heads(current_hidden)
                    actions, log_prob, entropy = choose_actions(
                        logits,
                        specs,
                        deterministic=args.deterministic,
                        random_actions=args.random_actions,
                    )
                    prediction = agent.predict_world(current_hidden, actions)
                    current_feedback = agent.make_feedback(
                        current_hidden,
                        logits,
                        predicted_pleasure,
                        predicted_pain,
                        prediction,
                    )

                if not behavior_learning:
                    score_gradients = entropy_gradients = None
                    pleasure_gradients = pain_gradients = None
                else:
                    # All local gradients are captured before any parameter mutation.
                    score_gradients = detached_gradients(
                        log_prob,
                        actor_parameters,
                        retain_graph=True,
                    )
                    entropy_gradients = detached_gradients(
                        entropy,
                        actor_parameters,
                        retain_graph=False,
                    )
                    pleasure_gradients = detached_gradients(
                        predicted_pleasure.sum(),
                        affect_parameters,
                        retain_graph=True,
                    )
                    pain_gradients = detached_gradients(
                        predicted_pain.sum(),
                        affect_parameters,
                        retain_graph=False,
                    )

                (
                    next_observations,
                    ignored_env_rewards,
                    terminated,
                    truncated,
                    _step_info,
                ) = env.step([actions], order_ij=True)
                next_observation = next_observations[0]
                next_hp_reported = extract_hp(next_observation, args.hp_key)
                done_signal = bool(terminated[0] or truncated[0])
                hp_zero = next_hp_reported <= 0.0
                terminal = bool(done_signal or hp_zero)
                if hp_zero:
                    terminal_source = "hp_zero"
                elif done_signal:
                    # In this project a positive-HP done can represent a human veto.
                    terminal_source = "done_or_human_veto"
                else:
                    terminal_source = "none"

                # If an environment auto-resets on done, do not misread reset HP as pleasure.
                next_hp_effective = 0.0 if terminal else next_hp_reported
                next_hp_normalized = next_hp_effective / args.hp_scale
                hp_delta_normalized = next_hp_normalized - hp_normalized

                target_next_latent: Optional[torch.Tensor] = None
                target_next_visual: Optional[torch.Tensor] = None
                next_predicted_pleasure = 0.0
                next_predicted_pain = 0.0
                if not terminal:
                    next_image = observation_to_tensor(
                        next_observation,
                        args.left_eye_key,
                        args.right_eye_key,
                        args.eye_height,
                        args.eye_width,
                        args.image_size,
                        device,
                    )
                    with torch.no_grad():
                        _next_latent_online, next_hidden = agent.encode_state(
                            next_image,
                            next_hp_normalized,
                            actions,
                            current_hidden.detach(),
                            current_feedback.detach(),
                        )
                        next_pleasure_tensor, next_pain_tensor = agent.target_affect(
                            next_hidden.detach()
                        )
                        next_predicted_pleasure = float(next_pleasure_tensor.item())
                        next_predicted_pain = float(next_pain_tensor.item())
                        target_next_latent = agent.target_encoder(next_image).detach()
                        target_next_visual = make_target_visual(
                            next_image,
                            args.visual_prediction_size,
                        )

                intrinsic = intrinsic_outcome(
                    hp_before=hp_normalized,
                    hp_after=next_hp_normalized,
                    terminal=terminal,
                    safe_hp=args.safe_hp,
                    drive_temperature=args.drive_temperature,
                    low_hp_threshold=args.low_hp_threshold,
                    low_hp_weight=args.low_hp_weight,
                    alive_pleasure=args.alive_pleasure,
                    death_pain=args.death_pain,
                )
                pleasure_value = float(predicted_pleasure.detach().item())
                pain_value = float(predicted_pain.detach().item())
                bootstrap = 0.0 if terminal else 1.0
                pleasure_target = (
                    intrinsic.pleasure
                    + args.gamma * bootstrap * next_predicted_pleasure
                )
                pain_target = intrinsic.pain + args.gamma * bootstrap * next_predicted_pain
                delta_pleasure_raw = pleasure_target - pleasure_value
                delta_pain_raw = pain_target - pain_value
                delta_pleasure = float(
                    np.clip(delta_pleasure_raw, -args.td_clip, args.td_clip)
                )
                delta_pain = float(np.clip(delta_pain_raw, -args.td_clip, args.td_clip))
                policy_delta = float(
                    np.clip(
                        delta_pleasure - args.pain_policy_weight * delta_pain,
                        -args.td_clip,
                        args.td_clip,
                    )
                )
                finite_scalars = {
                    "pleasure value": pleasure_value,
                    "pain value": pain_value,
                    "pleasure TD": delta_pleasure,
                    "pain TD": delta_pain,
                    "policy TD": policy_delta,
                    "intrinsic pleasure": intrinsic.pleasure,
                    "intrinsic pain": intrinsic.pain,
                }
                for scalar_name, scalar_value in finite_scalars.items():
                    if not math.isfinite(scalar_value):
                        raise FloatingPointError(
                            f"{scalar_name} became non-finite at step {step}: {scalar_value}"
                        )

                with torch.set_grad_enabled(world_learning):
                    total_world_loss, loss_parts = world_losses(
                        prediction,
                        target_next_latent,
                        target_next_visual,
                        target_hp_delta=float(
                            np.clip(
                                hp_delta_normalized,
                                -args.max_hp_delta_target,
                                args.max_hp_delta_target,
                            )
                        ),
                        terminal=terminal,
                        latent_weight=args.latent_loss_weight,
                        visual_weight=args.visual_loss_weight,
                        hp_weight=args.hp_loss_weight,
                        death_weight=args.death_loss_weight,
                        death_positive_weight=args.death_positive_weight,
                        death_focal_gamma=args.death_focal_gamma,
                    )
                if not bool(torch.isfinite(total_world_loss).all().item()):
                    raise FloatingPointError(
                        f"World loss became non-finite at step {step}: "
                        f"{loss_parts['world_loss']}"
                    )
                if terminal and deaths == 0:
                    save_checkpoint(
                        run_dir,
                        agent,
                        world_optimizer,
                        specs,
                        args,
                        step,
                        life,
                        life_step,
                        deaths,
                        lifetimes,
                        numbered=False,
                        extra_name="before_first_death_update.pt",
                    )

                actor_stats = UpdateStats()
                affect_stats = UpdateStats()
                pleasure_trace_norm = 0.0
                pain_trace_norm = 0.0
                world_grad_norm = 0.0
                selected_log_prob_change = 0.0
                current_action_sign_ok = ""

                if world_learning:
                    # Every local gradient was captured before this first parameter mutation.
                    world_optimizer.zero_grad(set_to_none=True)
                    total_world_loss.backward()
                    world_grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            world_parameters,
                            args.max_world_grad_norm,
                            error_if_nonfinite=True,
                        ).item()
                    )
                    world_optimizer.step()

                if behavior_learning:
                    assert score_gradients is not None
                    assert entropy_gradients is not None
                    assert pleasure_gradients is not None
                    assert pain_gradients is not None
                    actor_trace.accumulate(score_gradients)
                    actor_eligibility = actor_trace.combined()
                    actor_direction = []
                    for eligibility, entropy_gradient in zip(
                        actor_eligibility, entropy_gradients
                    ):
                        direction = eligibility * policy_delta
                        if entropy_gradient is not None:
                            direction = direction + args.entropy_coefficient * entropy_gradient
                        actor_direction.append(direction)
                    actor_stats = apply_manual_ascent(
                        actor_parameters,
                        actor_direction,
                        learning_rate=args.actor_lr,
                        max_direction_norm=args.max_actor_direction_norm,
                        trace_norm=actor_trace.norm(),
                    )

                    pleasure_trace.accumulate(pleasure_gradients)
                    pain_trace.accumulate(pain_gradients)
                    pleasure_eligibility = pleasure_trace.combined()
                    pain_eligibility = pain_trace.combined()
                    pleasure_trace_norm = tensor_list_norm(pleasure_eligibility)
                    pain_trace_norm = tensor_list_norm(pain_eligibility)
                    affect_direction = [
                        plus_trace * delta_pleasure + minus_trace * delta_pain
                        for plus_trace, minus_trace in zip(
                            pleasure_eligibility,
                            pain_eligibility,
                        )
                    ]
                    affect_stats = apply_manual_ascent(
                        affect_parameters,
                        affect_direction,
                        learning_rate=args.affect_lr,
                        max_direction_norm=args.max_affect_direction_norm,
                        trace_norm=math.sqrt(
                            pleasure_trace_norm**2 + pain_trace_norm**2
                        ),
                    )

                    with torch.no_grad():
                        post_logits = agent.actor(current_hidden.detach())
                        post_log_prob = float(
                            selected_log_prob(post_logits, specs, actions).item()
                        )
                        selected_log_prob_change = post_log_prob - float(log_prob.item())
                        if abs(policy_delta) > 1e-7:
                            # With eligibility memory this diagnostic can occasionally disagree
                            # for the current action; the exact local sign is tested by --self-test.
                            current_action_sign_ok = int(
                                selected_log_prob_change * policy_delta > 0.0
                            )
                if learning_enabled:
                    agent.update_targets(args.target_ema)
                if terminal and deaths == 0:
                    save_checkpoint(
                        run_dir,
                        agent,
                        world_optimizer,
                        specs,
                        args,
                        step,
                        life,
                        life_step,
                        deaths=1,
                        lifetimes=[*lifetimes, life_step],
                        numbered=True,
                        extra_name="after_first_death_update.pt",
                    )

                ignored_reward_value = safe_env_reward(ignored_env_rewards)
                row = {
                    "step": step,
                    "life": life,
                    "life_step": life_step,
                    "hp": hp_raw,
                    "hp_normalized": hp_normalized,
                    "next_hp": next_hp_effective,
                    "next_hp_reported": next_hp_reported,
                    "next_hp_normalized": next_hp_normalized,
                    "hp_delta": next_hp_effective - hp_raw,
                    "action": json.dumps(
                        {spec.name: action for spec, action in zip(specs, actions)},
                        ensure_ascii=False,
                    ),
                    "pleasure_signal": intrinsic.pleasure,
                    "pain_signal": intrinsic.pain,
                    "drive_before": intrinsic.drive_before,
                    "drive_after": intrinsic.drive_after,
                    "low_hp_urgency": intrinsic.low_hp_urgency,
                    "alive_pleasure_component": 0.0
                    if terminal
                    else args.alive_pleasure,
                    "predicted_pleasure": pleasure_value,
                    "predicted_pain": pain_value,
                    "next_predicted_pleasure": next_predicted_pleasure,
                    "next_predicted_pain": next_predicted_pain,
                    "delta_pleasure_raw": delta_pleasure_raw,
                    "delta_pain_raw": delta_pain_raw,
                    "delta_pleasure_used": delta_pleasure,
                    "delta_pain_used": delta_pain,
                    "policy_delta": policy_delta,
                    "td_clip_hit": int(
                        abs(delta_pleasure_raw) > args.td_clip
                        or abs(delta_pain_raw) > args.td_clip
                    ),
                    "entropy": float(entropy.detach().item()),
                    "selected_log_prob": float(log_prob.detach().item()),
                    "selected_log_prob_change": selected_log_prob_change,
                    "current_action_sign_ok": current_action_sign_ok,
                    "actor_trace_norm": actor_stats.trace_norm,
                    "actor_direction_norm": actor_stats.direction_norm,
                    "actor_update_norm": actor_stats.applied_update_norm,
                    "actor_direction_clip_hit": int(
                        actor_stats.direction_norm > args.max_actor_direction_norm
                    ),
                    "pleasure_trace_norm": pleasure_trace_norm,
                    "pain_trace_norm": pain_trace_norm,
                    "affect_direction_norm": affect_stats.direction_norm,
                    "affect_update_norm": affect_stats.applied_update_norm,
                    "affect_direction_clip_hit": int(
                        affect_stats.direction_norm > args.max_affect_direction_norm
                    ),
                    **loss_parts,
                    "world_grad_norm": world_grad_norm,
                    "world_grad_clip_hit": int(
                        world_grad_norm > args.max_world_grad_norm
                    ),
                    "predicted_hp_delta": float(prediction.hp_delta.detach().item()),
                    "predicted_death_probability": float(
                        torch.sigmoid(prediction.death_logit.detach()).item()
                    ),
                    "terminal": int(terminal),
                    "terminal_source": terminal_source,
                    "done_with_positive_hp": int(done_signal and not hp_zero),
                    "deaths": deaths + int(terminal),
                    "env_reward_ignored": ignored_reward_value,
                }
                logger.write(row, force_flush=terminal)

                if terminal:
                    deaths += 1
                    lifetimes.append(life_step)
                    recent = lifetimes[-10:]
                    print(
                        f"[death] step={step} life={life} lifetime={life_step} "
                        f"recent_mean={np.mean(recent):.1f} deaths={deaths}"
                    )
                    if deaths == 1:
                        print(
                            "[SDAEA] first-death snapshots: "
                            f"{run_dir / 'checkpoints' / 'before_first_death_update.pt'} "
                            "and "
                            f"{run_dir / 'checkpoints' / 'after_first_death_update.pt'}"
                        )
                    actor_trace.clear()
                    pleasure_trace.clear()
                    pain_trace.clear()
                    hidden = agent.initial_hidden(1, device)
                    feedback = agent.initial_feedback(1, device)
                    previous_actions = [0 for _ in specs]
                    life += 1
                    life_step = 0
                    observations, _reset_info = env.reset()
                    observation = observations[0]
                    hp_raw = extract_hp(observation, args.hp_key)
                else:
                    # detach keeps the numerical memory but prevents unbounded BPTT graphs.
                    hidden = current_hidden.detach()
                    feedback = current_feedback.detach()
                    previous_actions = actions
                    observation = next_observation
                    hp_raw = next_hp_reported

                if args.log_every > 0 and step % args.log_every == 0:
                    recent_mean = float(np.mean(lifetimes[-10:])) if lifetimes else float("nan")
                    print(
                        f"[step {step}] life_step={life_step} hp={hp_raw:.3f} "
                        f"r+={intrinsic.pleasure:.4f} r-={intrinsic.pain:.4f} "
                        f"td={policy_delta:+.4f} world={loss_parts['world_loss']:.4f} "
                        f"deaths={deaths} recent_life={recent_mean:.1f}"
                    )

                if args.save_every > 0 and step % args.save_every == 0:
                    latest = save_checkpoint(
                        run_dir,
                        agent,
                        world_optimizer,
                        specs,
                        args,
                        step,
                        life,
                        life_step,
                        deaths,
                        lifetimes,
                        numbered=True,
                    )
                    print(f"[SDAEA] checkpoint: {latest}")

                if args.max_deaths > 0 and deaths >= args.max_deaths:
                    break
        except KeyboardInterrupt:
            interrupted = True
            print("\n[SDAEA] Ctrl-C received; saving before exit.")
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)

        latest = save_checkpoint(
            run_dir,
            agent,
            world_optimizer,
            specs,
            args,
            final_step,
            life,
            life_step,
            deaths,
            lifetimes,
            numbered=False,
        )
        summary = {
            "steps": final_step,
            "deaths": deaths,
            "completed_lifetimes": lifetimes,
            "completed_mean_lifetime": float(np.mean(lifetimes)) if lifetimes else None,
            "completed_median_lifetime": float(np.median(lifetimes)) if lifetimes else None,
            # This unfinished life must not be silently discarded or treated as a death.
            "right_censored_current_lifetime": life_step if life_step > 0 else None,
            "learning_world": world_learning,
            "learning_actor_affect": behavior_learning,
            "external_reward_used_for_learning": False,
        }
        with (run_dir / "summary.json").open("w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, indent=2, ensure_ascii=False)
        if lifetimes:
            print(
                f"[SDAEA] finished: steps={final_step}, deaths={deaths}, "
                f"mean_lifetime={np.mean(lifetimes):.1f}, "
                f"last10={np.mean(lifetimes[-10:]):.1f}"
            )
        else:
            print(
                f"[SDAEA] finished: steps={final_step}, no completed lifetime yet, "
                f"current_lifetime={life_step}"
            )
        print(f"[SDAEA] metrics={run_dir / 'metrics.csv'}")
        print(f"[SDAEA] checkpoint={latest}")
    finally:
        if logger is not None:
            logger.close()
        if env is not None:
            try:
                env.close()
            except (BrokenPipeError, ConnectionError, OSError):
                pass
        if interrupted:
            print("[SDAEA] interrupted cleanly.")


def _self_test_policy_update_sign(device: torch.device) -> None:
    torch.manual_seed(7)
    actor = nn.Linear(4, 3, bias=True).to(device)
    feature = torch.tensor([[0.3, -0.5, 0.8, 0.2]], device=device)
    action = torch.tensor([1], device=device)
    initial_state = {name: value.detach().clone() for name, value in actor.state_dict().items()}

    def one_direction(sign: float) -> Tuple[float, float]:
        actor.load_state_dict(initial_state)
        logits = actor(feature)
        before = Categorical(logits=logits).log_prob(action)
        parameters = list(actor.parameters())
        gradients = torch.autograd.grad(before, parameters)
        with torch.no_grad():
            for parameter, gradient in zip(parameters, gradients):
                parameter.add_(gradient, alpha=0.05 * sign)
            after = Categorical(logits=actor(feature)).log_prob(action)
        return float(before.item()), float(after.item())

    positive_before, positive_after = one_direction(+1.0)
    negative_before, negative_after = one_direction(-1.0)
    assert positive_after > positive_before, (
        "Positive TD update did not increase selected-action log probability."
    )
    assert negative_after < negative_before, (
        "Negative TD update did not decrease selected-action log probability."
    )


def _self_test_delayed_eligibility(device: torch.device) -> None:
    """A delayed body signal must reach an old action without storing its trajectory."""

    torch.manual_seed(11)
    actor = nn.Linear(3, 2).to(device)
    state_at_action = torch.tensor([[0.4, -0.2, 0.7]], device=device)
    selected_action = torch.tensor([1], device=device)
    parameters = list(actor.parameters())
    original_log_prob = Categorical(logits=actor(state_at_action)).log_prob(selected_action)
    gradients = list(torch.autograd.grad(original_log_prob, parameters))
    decay = 0.95
    trace = MultiTimescaleEligibility(
        parameters,
        decays_and_weights=((decay, 1.0),),
        max_norm=1e6,
    )
    trace.accumulate(gradients)
    initial_norm = trace.norm()
    zeros = [torch.zeros_like(parameter) for parameter in parameters]
    delay_steps = 20
    for _ in range(delay_steps):
        trace.accumulate(zeros)
    expected_norm = initial_norm * decay**delay_steps
    assert math.isclose(trace.norm(), expected_norm, rel_tol=2e-5, abs_tol=1e-7)
    apply_manual_ascent(
        parameters,
        trace.combined(),
        learning_rate=0.05,
        max_direction_norm=1e6,
        trace_norm=trace.norm(),
    )
    with torch.no_grad():
        updated_log_prob = Categorical(logits=actor(state_at_action)).log_prob(selected_action)
    assert float(updated_log_prob.item()) > float(original_log_prob.item())


def _self_test_intrinsic_signs() -> None:
    common = dict(
        safe_hp=0.7,
        drive_temperature=0.15,
        low_hp_threshold=0.35,
        low_hp_weight=0.5,
        alive_pleasure=0.0,
        death_pain=5.0,
    )
    healed = intrinsic_outcome(0.4, 0.7, terminal=False, **common)
    damaged = intrinsic_outcome(0.7, 0.4, terminal=False, **common)
    died = intrinsic_outcome(0.4, 0.0, terminal=True, **common)
    assert healed.pleasure > healed.pain
    assert damaged.pain > damaged.pleasure
    assert died.pain >= common["death_pain"]
    assert died.pleasure == 0.0


def _self_test_agent(device: torch.device) -> None:
    specs = [
        ActionSpec("accelerate_forward", 3),
        ActionSpec("accelerate_sideways", 3),
        ActionSpec("shoot", 2, learned=False),
        ActionSpec("turn", 3),
    ]
    agent = OnlineHomeostaticAgent(
        specs,
        latent_dim=16,
        hidden_dim=24,
        visual_prediction_size=4,
    ).to(device)
    parameter_group_sanity(agent)
    image = torch.rand(1, 6, 32, 32, device=device)
    next_image = torch.rand(1, 6, 32, 32, device=device)
    hidden = agent.initial_hidden(1, device)
    feedback = agent.initial_feedback(1, device)
    previous_actions = [0, 0, 0, 0]

    _latent, current_hidden = agent.encode_state(
        image,
        hp_normalized=0.6,
        previous_actions=previous_actions,
        previous_hidden=hidden,
        previous_feedback=feedback,
    )
    logits, pleasure, pain = agent.heads(current_hidden)
    actions, log_prob, entropy = choose_actions(
        logits,
        specs,
        deterministic=False,
        random_actions=False,
    )
    prediction = agent.predict_world(current_hidden, actions)
    carried = agent.make_feedback(
        current_hidden,
        logits,
        pleasure,
        pain,
        prediction,
    ).detach()
    assert carried.shape == (1, agent.feedback_dim)
    assert not carried.requires_grad
    assert math.isfinite(float(log_prob.item()))
    assert math.isfinite(float(entropy.item()))

    actor_parameters = agent.actor_parameters()
    affect_parameters = agent.affect_parameters()
    # Reproduce the real step ordering: capture every local gradient first.
    score_gradients = detached_gradients(log_prob, actor_parameters, retain_graph=True)
    _entropy_gradients = detached_gradients(entropy, actor_parameters, retain_graph=False)
    pleasure_gradients = detached_gradients(
        pleasure.sum(),
        affect_parameters,
        retain_graph=True,
    )
    _pain_gradients = detached_gradients(
        pain.sum(),
        affect_parameters,
        retain_graph=False,
    )

    with torch.no_grad():
        target_latent = agent.target_encoder(next_image)
        target_visual = make_target_visual(next_image, 4)
    loss, parts = world_losses(
        prediction,
        target_latent,
        target_visual,
        target_hp_delta=-0.01,
        terminal=False,
        latent_weight=1.0,
        visual_weight=1.0,
        hp_weight=1.0,
        death_weight=1.0,
        death_positive_weight=10.0,
        death_focal_gamma=2.0,
    )
    loss.backward()
    assert math.isfinite(parts["world_loss"])
    assert any(parameter.grad is not None for parameter in agent.world_parameters())
    assert all(parameter.grad is None for parameter in agent.actor_parameters())
    assert all(parameter.grad is None for parameter in agent.affect_parameters())

    trace = MultiTimescaleEligibility(
        actor_parameters,
        decays_and_weights=((0.9, 0.7), (0.99, 0.3)),
        max_norm=100.0,
    )
    trace.accumulate(score_gradients)
    before_log_prob = float(log_prob.item())
    actor_direction = trace.combined()
    apply_manual_ascent(
        actor_parameters,
        actor_direction,
        learning_rate=0.01,
        max_direction_norm=100.0,
        trace_norm=trace.norm(),
    )
    with torch.no_grad():
        after_log_prob = float(
            selected_log_prob(agent.actor(current_hidden.detach()), specs, actions).item()
        )
    assert after_log_prob > before_log_prob
    assert trace.norm() > 0.0
    assert all(not tensor.requires_grad for trace_set in trace.traces for tensor in trace_set)

    affect_trace = MultiTimescaleEligibility(
        affect_parameters,
        decays_and_weights=((0.97, 1.0),),
        max_norm=100.0,
    )
    affect_trace.accumulate(pleasure_gradients)
    pleasure_before = float(pleasure.item())
    apply_manual_ascent(
        affect_parameters,
        affect_trace.combined(),
        learning_rate=0.01,
        max_direction_norm=100.0,
        trace_norm=affect_trace.norm(),
    )
    with torch.no_grad():
        pleasure_after, _pain_after = agent.affect(current_hidden.detach())
    assert float(pleasure_after.item()) > pleasure_before


def self_test(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = choose_device(args.device)
    _self_test_policy_update_sign(device)
    _self_test_delayed_eligibility(device)
    _self_test_intrinsic_signs()
    _self_test_agent(device)
    print(
        "[self-test] PASS: policy update signs, 20-step delayed eligibility credit, "
        "intrinsic affect signs, parameter separation, world gradients and detached memory."
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Online homeostatic SDAEA validation in the current Godot environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--self-test", action="store_true", help="Run local tests, no Godot.")
    parser.add_argument("--env-path", default=None, help="Exported Godot binary base path.")
    parser.add_argument("--port", type=int, default=11008)
    parser.add_argument(
        "--show-window",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep rendering enabled; visual learning normally requires this.",
    )
    parser.add_argument("--framerate", type=int, default=None)
    parser.add_argument("--action-repeat", type=int, default=None)
    parser.add_argument("--speedup", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--max-deaths", type=int, default=0, help="0 means unlimited.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
    )

    parser.add_argument("--left-eye-key", default="left_eye")
    parser.add_argument("--right-eye-key", default="right_eye")
    parser.add_argument("--hp-key", default="hp")
    parser.add_argument("--eye-height", type=int, default=300)
    parser.add_argument("--eye-width", type=int, default=320)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument(
        "--hp-scale",
        type=float,
        default=10.0,
        help="Raw HP corresponding to normalized HP=1; values above it are retained.",
    )

    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--visual-prediction-size", type=int, default=8)
    parser.add_argument("--learn-shoot", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-learn", action="store_true")
    parser.add_argument("--random-actions", action="store_true")
    parser.add_argument(
        "--freeze-world",
        action="store_true",
        help="Ablation: update actor/affect only; keep visual world representation fixed.",
    )
    parser.add_argument(
        "--freeze-behavior",
        action="store_true",
        help="Ablation: update world model only; keep actor/affect fixed.",
    )

    parser.add_argument("--gamma", type=float, default=0.998)
    parser.add_argument("--safe-hp", type=float, default=0.7)
    parser.add_argument("--drive-temperature", type=float, default=0.15)
    parser.add_argument("--low-hp-threshold", type=float, default=0.35)
    parser.add_argument("--low-hp-weight", type=float, default=0.01)
    parser.add_argument("--alive-pleasure", type=float, default=0.002)
    parser.add_argument("--death-pain", type=float, default=5.0)
    parser.add_argument("--pain-policy-weight", type=float, default=1.0)
    parser.add_argument("--td-clip", type=float, default=3.0)

    parser.add_argument("--actor-lr", type=float, default=3e-5)
    parser.add_argument("--affect-lr", type=float, default=1e-4)
    parser.add_argument("--world-lr", type=float, default=1e-4)
    parser.add_argument("--world-weight-decay", type=float, default=1e-6)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--actor-lambda-fast", type=float, default=0.90)
    parser.add_argument("--actor-lambda-slow", type=float, default=0.995)
    parser.add_argument("--actor-trace-fast-weight", type=float, default=0.7)
    parser.add_argument("--affect-lambda", type=float, default=0.97)
    parser.add_argument("--max-trace-norm", type=float, default=100.0)
    parser.add_argument("--max-actor-direction-norm", type=float, default=10.0)
    parser.add_argument("--max-affect-direction-norm", type=float, default=10.0)
    parser.add_argument("--max-world-grad-norm", type=float, default=10.0)
    parser.add_argument("--target-ema", type=float, default=0.995)

    parser.add_argument("--latent-loss-weight", type=float, default=0.25)
    parser.add_argument("--visual-loss-weight", type=float, default=1.0)
    parser.add_argument("--hp-loss-weight", type=float, default=2.0)
    parser.add_argument("--death-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--death-positive-weight",
        type=float,
        default=10.0,
        help="Compensate for rare terminal samples in the auxiliary death predictor.",
    )
    parser.add_argument(
        "--death-focal-gamma",
        type=float,
        default=2.0,
        help="Down-weight easy alive samples in the imbalanced death prediction loss.",
    )
    parser.add_argument("--max-hp-delta-target", type=float, default=5.0)

    parser.add_argument("--run-dir", default=None)
    parser.add_argument(
        "--warm-start",
        default=None,
        help=(
            "Load parameters from a checkpoint into a new run. This does not restore "
            "the Godot world, recurrent memory, eligibility tags, RNG, or lifetime."
        ),
    )
    parser.add_argument(
        "--warm-start-optimizer",
        action="store_true",
        help="Also load world Adam moments; current CLI learning rate still wins.",
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--metrics-flush-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=5000)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.self_test:
        self_test(args)
    else:
        train(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""SDAEA v2: streaming visual actor-critic driven only by the robot's body.

Run: python sdaea_streaming_v2.py --run-dir runs/v2_seed7 --seed 7
Then press Play in Godot. Dependencies and eye decoding are shared with v1.

No environment rewards, replay buffer, trajectory batches, or world snapshots.
One online update follows each motor decision (default: hold for 4 env.step calls).
Only the current transition, sensory EMA and parameter-sized eligibility traces
are retained by default. Optional gated recurrence retains a bounded observation
window for truncated temporal gradients. A single signed critic estimates future internal valence; positive
and negative body signals remain explicitly logged as pleasure/pain.

Update design is adapted from Stream AC / ObGD (Elsayed et al., 2024):
https://arxiv.org/abs/2410.14606 -- not a reproduction of their benchmarks.
HP shaping uses gamma*Phi(next)-Phi(now), with Phi(terminal)=0. For fixed
initial body state its discounted sum telescopes, so the base objective remains
discounted survival rather than cycling between damage and healing.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

from sdaea_online_validate import (
    atomic_torch_save, choose_device, extract_action_specs, extract_hp,
    observation_to_tensor, seed_everything,
)


@dataclass
class Config:
    seed: int = 7
    device: str = "cpu"  # Small streaming nets often run faster without GPU dispatch.
    threads: int = 2
    max_steps: int = 100_000  # Godot step calls, not decisions or rendered frames.
    hold_steps: int = 4
    gamma: float = 0.999
    trace_steps: float = 300.0  # e-fold time of lambda in env steps, before discount.
    optimizer: str = "bounded"  # Optional Adam comparison; no eligibility traces.
    adam_grad_clip: float = 1.0
    adam_td_clip: float = 5.0
    actor_lr: float = 0.1  # upper bound; ObGD sets the effective step size each update.
    critic_lr: float = 0.5
    actor_kappa: float = 3.0
    critic_kappa: float = 2.0
    entropy: float = 0.01  # scaled by |TD|, unlike the old constant pressure.
    exploration: float = 0.03  # mixture is part of the policy used for log-prob too.
    hp_unit: float = 5.0  # energy unit, NOT a guessed full-charge limit.
    alive: float = 0.01
    death_cost: float = 2.0
    shaping: float = 1.0
    energy_delta: float = 0.0  # Optional linear HP-change objective; body-only.
    memory_steps: float = 20.0
    recurrent_size: int = 0  # 0 preserves the original feedforward baseline.
    recurrent_tau: float = 200.0  # Initial decay time in physical steps.
    recurrent_layer: int = 0  # Zero-based hidden layer receiving feedback.
    cnn_extra: int = 0  # Additional stride-1 convolutions after the original CNN.
    bptt_steps: int = 8  # Decisions, including the current decision.
    depth: int = 2
    width: int = 96
    image_size: int = 48
    eye_height: int = 300
    eye_width: int = 320
    save_every: int = 5000
    log_every: int = 1000
    no_learn: bool = False
    random_actions: bool = False
    deterministic: bool = False


def validate(c: Config) -> None:
    for name in ("threads", "max_steps", "hold_steps", "hp_unit", "depth", "width",
                 "image_size", "eye_height", "eye_width", "actor_lr", "critic_lr",
                 "actor_kappa", "critic_kappa", "memory_steps"):
        if getattr(c, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if c.optimizer not in ("bounded", "adam"):
        raise ValueError("Unknown optimizer")
    if c.optimizer == "adam" and (c.trace_steps != 0 or not 0 < c.actor_lr <= .01
                                  or not 0 < c.critic_lr <= .01
                                  or c.adam_grad_clip <= 0 or c.adam_td_clip <= 0):
        raise ValueError("Adam requires trace_steps=0, explicit small rates, positive clipping")
    if c.recurrent_size < 0 or c.recurrent_tau <= 0 or c.bptt_steps < 1:
        raise ValueError("Require recurrent_size>=0, recurrent_tau>0, bptt_steps>=1")
    if not 0 <= c.recurrent_layer < c.depth or c.cnn_extra < 0:
        raise ValueError("Invalid recurrent layer or CNN depth")
    if not 0 < c.gamma < 1 or not 0 <= c.exploration < 1 or c.trace_steps < 0:
        raise ValueError("Require 0<gamma<1, 0<=exploration<1, trace_steps>=0")
    if min(c.alive, c.death_cost, c.shaping, c.entropy, c.energy_delta) < 0:
        raise ValueError("Internal signal and entropy coefficients must be nonnegative")
    if (c.deterministic or c.random_actions) and not c.no_learn:
        raise ValueError("--deterministic/--random-actions require --no-learn")


def body_signal(hp: float, next_hp: float, terminal: bool, c: Config) -> dict:
    if hp <= 0 or not math.isfinite(hp) or not math.isfinite(next_hp):
        raise ValueError("Learning requires a live, finite pre-action HP")
    phi = c.shaping * math.log1p(hp / c.hp_unit)
    next_phi = 0.0 if terminal else c.shaping * math.log1p(max(0.0, next_hp) / c.hp_unit)
    potential_change = c.gamma * next_phi - phi
    survival = 0.0 if terminal else c.alive
    death = c.death_cost if terminal else 0.0
    energy_change = c.energy_delta * ((0.0 if terminal else max(0.0, next_hp)) - hp) / c.hp_unit
    valence = survival - death + potential_change + energy_change
    return {
        "valence": valence,
        "pleasure": survival + max(potential_change, 0.0) + max(energy_change, 0.0),
        "pain": death + max(-potential_change, 0.0) + max(-energy_change, 0.0),
        "potential_change": potential_change,
        "energy_change": energy_change,
    }


def motor_table(specs) -> list[list[int]]:
    choices = [range(s.size) if s.learned else (s.fixed_value,) for s in specs]
    table = [list(a) for a in itertools.product(*choices)]
    if not table or len(table) > 128:
        raise ValueError("This version supports at most 128 joint discrete actions")
    return table


@dataclass
class State:
    image: torch.Tensor
    history: torch.Tensor
    body: torch.Tensor
    elapsed: int = 0  # Physical steps since the preceding observation.


class SensoryMemory:
    """Fixed-size low-pass sensory state. No learned long-memory claim or BPTT."""

    def __init__(self, c: Config, device: torch.device, n_actions: int):
        self.c, self.device, self.n_actions = c, device, n_actions
        self.ema = None
        self.last_hp = None

    def clear(self):
        self.ema = self.last_hp = None

    def state(self, observation, previous_action: int | None, elapsed: int) -> State:
        image = observation_to_tensor(
            observation, "left_eye", "right_eye", self.c.eye_height,
            self.c.eye_width, self.c.image_size, self.device,
        )
        hp = extract_hp(observation, "hp")
        small = F.adaptive_avg_pool2d(image, (6, 8))
        decay = math.exp(-elapsed / self.c.memory_steps)
        self.ema = small if self.ema is None else decay * self.ema + (1 - decay) * small
        delta = 0.0 if self.last_hp is None else (hp - self.last_hp) / self.c.hp_unit
        body = torch.zeros(1, 3 + self.n_actions, device=self.device)
        body[0, :3] = torch.tensor([
            math.log1p(max(0.0, hp) / self.c.hp_unit),
            1.0 / (1.0 + max(0.0, hp) / self.c.hp_unit),
            math.tanh(delta),
        ], device=self.device)
        if previous_action is not None:
            body[0, 3 + previous_action] = 1.0
        self.last_hp = hp
        return State(image.detach(), self.ema.detach().clone(), body, elapsed)


class VisualNetwork(nn.Module):
    """Owns its entire visual-to-output path; no detached learned world encoder."""

    def __init__(self, n_actions: int, outputs: int, width: int, depth: int = 2, cnn_extra: int = 0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, 16, 5, stride=2, padding=2), nn.LeakyReLU(0.1),
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((4, 6)), nn.Flatten(),
        )
        if cnn_extra:
            layers = list(self.conv.children())
            extra = []
            for _ in range(cnn_extra):
                extra.extend([nn.Conv2d(24, 24, 3, padding=1), nn.LeakyReLU(0.1)])
            self.conv = nn.Sequential(*(layers[:-2] + extra + layers[-2:]))
        # Raw spatial RGB average/max paths retain small bright objects while the
        # learned representation is immature. Every RGB channel is treated equally.
        inputs = 24 * 4 * 6 + 4 * 6 * 6 * 8 + 3 + n_actions
        layers = []
        for index in range(depth):
            layers.extend([nn.Linear(inputs if index == 0 else width, width),
                           nn.LayerNorm(width, elementwise_affine=False), nn.LeakyReLU(0.1)])
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(width, outputs)
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    fan_in = module.weight[0].numel()
                    bound = 1.0 / math.sqrt(fan_in)
                    module.weight.uniform_(-bound, bound)
                    module.weight.mul_(torch.rand_like(module.weight) > 0.9)
                    nn.init.zeros_(module.bias)
            # Small nonzero output weights let the first body signal train vision.
            nn.init.normal_(self.output.weight, std=0.005)

    def visual_features(self, s: State):
        small = F.adaptive_avg_pool2d(s.image, (6, 8))
        peak = F.adaptive_max_pool2d(s.image, (6, 8))
        spatial = torch.cat((small * 2 - 1, peak * 2 - 1,
                             s.history * 2 - 1, 2 * (small - s.history)), dim=1)
        features = torch.cat((self.conv(s.image * 2 - 1),
                              spatial.flatten(1), s.body), dim=1)
        return features

    def encode(self, s: State):
        return self.hidden(self.visual_features(s))

    def forward(self, s: State):
        return self.output(self.encode(s))


class DecayingMemory(nn.Module):
    """Bounded gated state; elapsed is measured in physics steps, not calls."""

    def __init__(self, inputs: int, size: int, tau: float):
        super().__init__()
        self.size = size
        self.log_tau = nn.Parameter(torch.zeros(size))
        self.register_buffer("initial_log_tau", torch.tensor(math.log(tau)))
        self.gates = nn.Linear(inputs + size, 3 * size)
        self.candidate = nn.Linear(inputs + size, size)
        nn.init.zeros_(self.gates.bias)
        with torch.no_grad():
            self.gates.bias[:size].fill_(-2.0)  # Initially conservative writing.

    def decay(self, x, previous=None, elapsed=0):
        if previous is None:
            previous = x.new_zeros(x.shape[0], self.size)
        tau = (self.log_tau + self.initial_log_tau).clamp(
            math.log(1.0), math.log(10000.0)).exp()
        return previous * torch.exp(-elapsed / tau)

    def forward(self, x, previous=None, elapsed=0):
        faded = self.decay(x, previous, elapsed)
        return self.update_faded(x, faded)

    def update_faded(self, x, faded):
        write, reset, read = self.gates(torch.cat((x, faded), -1)).sigmoid().chunk(3, -1)
        candidate = self.candidate(torch.cat((x, reset * faded), -1)).tanh()
        state = (1 - write) * faded + write * candidate
        return read * state, state


class RecurrentVisualNetwork(VisualNetwork):
    """Previous memory enters FC1 before normalization and activation."""

    def __init__(self, n_actions, outputs, width, depth, size=256, tau=200.0, layer=0, cnn_extra=0):
        super().__init__(n_actions, outputs, width, depth, cnn_extra)
        self.recurrent_layer = layer
        self.memory = DecayingMemory(width, size, tau)
        self.memory_feedback = nn.Linear(size, width, bias=False)
        self.memory_read = nn.Linear(size, width, bias=False)

    def forward_features(self, features, previous=None, elapsed=0):
        faded = self.memory.decay(features, previous, elapsed)
        x = features
        for index in range(len(self.hidden) // 3):
            x = self.hidden[index * 3](x)
            if index == self.recurrent_layer:
                x = x + self.memory_feedback(faded)
            x = self.hidden[index * 3 + 2](self.hidden[index * 3 + 1](x))
            if index == self.recurrent_layer:
                read, state = self.memory(x, faded, elapsed=0)
                x = x + self.memory_read(read)
        return self.output(x), state

    def sequence(self, features, previous, elapsed):
        # Layers before/after the recurrent block are independent across time.
        # Batch them as well as the CNN; only the feedback block must be sequential.
        layer = self.recurrent_layer * 3
        x = features
        for module in list(self.hidden.children())[:layer]:
            x = module(x)
        projected = self.hidden[layer](x)
        if previous is None:
            previous = x.new_zeros(1, self.memory.size)
        tau = (self.memory.log_tau + self.memory.initial_log_tau).clamp(
            math.log(1.0), math.log(10000.0)).exp()
        durations = x.new_tensor(elapsed).unsqueeze(1)
        decays = torch.exp(-durations / tau)
        reads, first = [], None
        for i in range(len(elapsed)):
            faded = previous * decays[i:i+1]
            current = projected[i:i+1] + self.memory_feedback(faded)
            current = self.hidden[layer+2](self.hidden[layer+1](current))
            read, previous = self.memory.update_faded(current, faded)
            if first is None:
                first = previous
            reads.append(current + self.memory_read(read))
        x = torch.cat(reads)
        for module in list(self.hidden.children())[layer+3:]:
            x = module(x)
        return self.output(x[-1:]), previous, first

    def forward(self, s, previous=None):
        return self.forward_features(self.visual_features(s), previous, s.elapsed)


class BoundedTrace:
    """ObGD-style ascent. The L1 bound is a safeguard, not a convergence proof."""

    def __init__(self, parameters, lr: float, kappa: float):
        self.parameters = list(parameters)
        self.traces = [torch.zeros_like(p) for p in self.parameters]
        self.lr, self.kappa = lr, kappa

    @torch.no_grad()
    def step(self, gradients, delta: float, decay: float) -> dict:
        if not math.isfinite(delta):
            raise FloatingPointError("Non-finite TD error")
        torch._foreach_mul_(self.traces, decay)
        torch._foreach_add_(self.traces, [g.detach() for g in gradients])
        l1 = torch.stack(torch._foreach_norm(self.traces, 1)).sum().item()
        if not math.isfinite(l1):
            raise FloatingPointError("Non-finite eligibility trace")
        denominator = max(1.0, self.lr * self.kappa * max(1.0, abs(delta)) * l1)
        effective_lr = self.lr / denominator
        torch._foreach_add_(self.parameters, self.traces, alpha=effective_lr * delta)
        return {"lr": effective_lr, "trace_l1": l1,
                "update_l1": effective_lr * abs(delta) * l1}

    @torch.no_grad()
    def clear(self):
        for t in self.traces:
            t.zero_()


class AdamUpdate:
    """TD-weighted gradient ascent, with clipped TD and gradient norm; no traces."""
    def __init__(self, parameters, lr, grad_clip, td_clip):
        self.parameters = list(parameters)
        self.traces = []
        self.lr, self.grad_clip, self.td_clip = lr, grad_clip, td_clip
        self.optimizer = torch.optim.Adam(self.parameters, lr=lr, foreach=True)

    @torch.no_grad()
    def step(self, gradients, delta, decay):
        if not math.isfinite(delta):
            raise FloatingPointError("Non-finite TD error")
        advantage = max(-self.td_clip, min(self.td_clip, delta))
        before = [p.detach().clone() for p in self.parameters]
        self.optimizer.zero_grad(set_to_none=True)
        for p, g in zip(self.parameters, gradients):
            p.grad = g.detach().mul(-advantage)
        torch.nn.utils.clip_grad_norm_(self.parameters, self.grad_clip,
                                      error_if_nonfinite=True, foreach=True)
        self.optimizer.step()
        differences = torch._foreach_sub(self.parameters, before)
        update = torch.stack(torch._foreach_norm(differences, 1)).sum().item()
        return {"lr": self.lr, "trace_l1": 0.0, "update_l1": update}

    def clear(self):
        # Episodes reset memory, not the optimizer's accumulated moments.
        self.optimizer.zero_grad(set_to_none=True)


class Learner:
    def __init__(self, c: Config, n_actions: int, device: torch.device):
        self.c, self.n_actions, self.device = c, n_actions, device
        def network(outputs):
            if c.recurrent_size:
                return RecurrentVisualNetwork(n_actions, outputs, c.width, c.depth,
                                              c.recurrent_size, c.recurrent_tau, c.recurrent_layer, c.cnn_extra).to(device)
            return VisualNetwork(n_actions, outputs, c.width, c.depth, c.cnn_extra).to(device)
        self.actor, self.critic = network(n_actions), network(1)
        self.context = []
        self.anchors = {"actor": None, "critic": None}
        self.pending_actor = None
        if c.optimizer == "adam":
            self.actor_update = AdamUpdate(self.actor.parameters(), c.actor_lr, c.adam_grad_clip, c.adam_td_clip)
            self.critic_update = AdamUpdate(self.critic.parameters(), c.critic_lr, c.adam_grad_clip, c.adam_td_clip)
        else:
            self.actor_update = BoundedTrace(self.actor.parameters(), c.actor_lr, c.actor_kappa)
            self.critic_update = BoundedTrace(self.critic.parameters(), c.critic_lr, c.critic_kappa)
        self.trace_discount = 0.0 if c.trace_steps == 0 else math.exp(-1 / c.trace_steps)
        self.previous_elapsed = None

    def evaluate(self, name, s):
        model = getattr(self, name)
        if not self.c.recurrent_size:
            return model(s), None, None
        h = self.anchors[name]
        if self.c.no_learn:
            output, h = model(s, h)
            return output, h, h
        sequence = self.context + [s]
        # CNN over the temporal window is one GPU batch; recurrence remains ordered.
        batch = State(torch.cat([v.image for v in sequence]),
                      torch.cat([v.history for v in sequence]),
                      torch.cat([v.body for v in sequence]))
        features = model.visual_features(batch)
        return model.sequence(features, h, [v.elapsed for v in sequence])

    def distribution(self, s: State):
        logits, _, _ = self.evaluate("actor", s)
        return self.policy(logits)

    def prepare_action(self, state):
        # Reuse this exact pre-action actor graph for the subsequent update.
        # No parameter update may occur between this call and learn(state,...).
        with torch.set_grad_enabled(not self.c.no_learn):
            result = self.evaluate("actor", state)
            self.pending_actor = None if self.c.no_learn else (state, result)
            return self.policy(result[0])

    def policy(self, logits):
        probs = torch.softmax(logits, -1)
        probs = (1 - self.c.exploration) * probs + self.c.exploration / self.n_actions
        return Categorical(probs=probs)

    @torch.no_grad()
    def act(self, state: State) -> int:
        if self.c.random_actions:
            return int(torch.randint(self.n_actions, ()).item())
        dist = self.distribution(state)
        if self.c.deterministic:
            return int(dist.probs.argmax().item())
        return int(dist.sample().item())

    def learn(self, state: State, action: int, valence: float,
              next_state: State | None, elapsed: int, terminal: bool) -> dict:
        # The only target is a body-derived return; no environment reward argument exists.
        discount = self.c.gamma ** elapsed
        with torch.set_grad_enabled(not self.c.no_learn):
            critic_output, critic_h, critic_first = self.evaluate("critic", state)
            if self.pending_actor is not None and self.pending_actor[0] is state:
                logits, actor_h, actor_first = self.pending_actor[1]
            else:
                logits, actor_h, actor_first = self.evaluate("actor", state)
            self.pending_actor = None
            value = critic_output.squeeze()
            dist = self.policy(logits)
            with torch.no_grad():
                if terminal:
                    bootstrap = 0.0
                elif self.c.recurrent_size:
                    bootstrap = self.critic(next_state, critic_h.detach())[0].item()
                else:
                    bootstrap = self.critic(next_state).item()
            log_prob = dist.log_prob(torch.tensor([action], device=self.device)).sum()
            entropy = dist.entropy().sum()
            delta = valence + (0.0 if terminal else discount * bootstrap) - value.item()
            metrics = {"td": delta, "value": value.item(), "bootstrap": bootstrap,
                       "entropy": entropy.item(), "max_probability": dist.probs.max().item()}
            if self.c.no_learn:
                self.advance_memory(state, actor_first, critic_first)
                if terminal:
                    self.clear()
                return metrics
            # Entropy strength vanishes with the TD error, avoiding a constant push
            # back to uniform when bodily advantages are tiny.
            objective = log_prob + self.c.entropy * np.sign(delta) * entropy
            actor_grads = torch.autograd.grad(objective, self.actor_update.parameters)
            critic_grads = torch.autograd.grad(value, self.critic_update.parameters)
        self.advance_memory(state, actor_first, critic_first)
        # Gradients and target refer to the same pre-update parameter version.
        # Tags refer to decision-start states. Their age since the previous state
        # is the PREVIOUS action duration, especially when the final hold is short.
        decay = 0.0 if self.previous_elapsed is None else (
            self.c.gamma * self.trace_discount) ** self.previous_elapsed
        for prefix, updater, grads in (("actor", self.actor_update, actor_grads),
                                       ("critic", self.critic_update, critic_grads)):
            stats = updater.step(grads, delta, decay)
            metrics.update({prefix + "_" + k: v for k, v in stats.items()})
        self.previous_elapsed = elapsed
        if terminal:
            self.clear()
        return metrics

    def advance_memory(self, state, actor_first, critic_first):
        if not self.c.recurrent_size:
            return
        if self.c.no_learn:
            # Frozen parameters need only the current state, no window recomputation.
            self.anchors = {"actor": actor_first.detach(), "critic": critic_first.detach()}
            self.context = [State(state.image.detach(), state.history.detach(),
                                  state.body.detach(), state.elapsed)]
            return
        self.context.append(State(state.image.detach(), state.history.detach(),
                                  state.body.detach(), state.elapsed))
        if len(self.context) >= self.c.bptt_steps:
            self.context.pop(0)
            self.anchors = {"actor": actor_first.detach(), "critic": critic_first.detach()}

    def clear(self):
        self.pending_actor = None
        self.context.clear()
        self.anchors = {"actor": None, "critic": None}
        self.actor_update.clear()
        self.critic_update.clear()
        self.previous_elapsed = None

    def save(self, path: Path, specs, step: int):
        atomic_torch_save({"version": 2, "config": asdict(self.c), "step": step,
                           "recurrent_architecture": "first_fc_v1" if self.c.recurrent_size else None,
                           "specs": [asdict(s) for s in specs],
                           "actor": self.actor.state_dict(),
                           "critic": self.critic.state_dict(),
                           "optimizers": {name: getattr(self, name + "_update").optimizer.state_dict()
                                          for name in ("actor", "critic")} if self.c.optimizer == "adam" else None}, path)

    def load(self, path: Path, specs, allow_objective_change: bool = False, allow_optimizer_change: bool = False):
        payload = torch.load(path, map_location=self.device, weights_only=True)
        if payload.get("version") != 2 or payload["specs"] != [asdict(s) for s in specs]:
            raise ValueError("Checkpoint version/action space mismatch (v1 weights are incompatible)")
        for key in ("hp_unit", "gamma", "shaping", "alive", "death_cost", "width",
                    "hold_steps", "memory_steps", "image_size", "exploration"):
            if key == "gamma" and allow_objective_change:
                continue
            if payload["config"][key] != getattr(self.c, key):
                raise ValueError(f"Checkpoint {key}={payload['config'][key]} differs from CLI")
        if not allow_objective_change and payload["config"].get("energy_delta", 0.0) != self.c.energy_delta:
            raise ValueError("Checkpoint energy_delta differs from CLI")
        if payload["config"].get("depth", 2) != self.c.depth:
            raise ValueError("Checkpoint depth differs from CLI")
        for key, default in (("recurrent_size", 0), ("recurrent_tau", 200.0), ("bptt_steps", 8), ("recurrent_layer", 0), ("cnn_extra", 0)):
            if payload["config"].get(key, default) != getattr(self.c, key):
                raise ValueError(f"Checkpoint {key} differs from CLI")
        if self.c.recurrent_size and payload.get("recurrent_architecture") != "first_fc_v1":
            raise ValueError("Checkpoint recurrent architecture differs: expected first_fc_v1")
        source_optimizer = payload["config"].get("optimizer", "bounded")
        if source_optimizer != self.c.optimizer and not allow_optimizer_change:
            raise ValueError("Checkpoint optimizer differs; request explicit optimizer transfer")
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        if self.c.optimizer == "adam" and not allow_optimizer_change and payload.get("optimizers"):
            for name in ("actor", "critic"):
                updater = getattr(self, name + "_update")
                updater.optimizer.load_state_dict(payload["optimizers"][name])
                for group in updater.optimizer.param_groups:
                    group["lr"] = getattr(self.c, name + "_lr")
        self.clear()


def reset_live(env, attempts: int = 10):
    """Do not blame a newly chosen action for an already-dead reset observation."""
    for attempt in range(attempts):
        observations, _ = env.reset()
        obs = observations[0]
        if extract_hp(obs, "hp") > 0:
            return obs, attempt
    raise RuntimeError("Godot returned dead HP after 10 resets; no update was made for these states")


FIELDS = ("step", "decision", "life", "life_step", "elapsed", "hp", "next_hp",
          "hp_delta", "action", "valence", "pleasure", "pain", "td", "value",
          "bootstrap", "entropy", "max_probability", "actor_lr", "critic_lr",
          "actor_trace_l1", "critic_trace_l1", "actor_update_l1", "critic_update_l1",
          "terminal", "terminal_source", "positive_hp_events", "negative_hp_events",
          "reset_retries", "image_change", "updates")


def run(c: Config, run_dir: Path, env_path=None, port=11008, warm_start=None, env=None):
    validate(c)
    seed_everything(c.seed)
    torch.set_num_threads(c.threads)
    device = choose_device(c.device)
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()):
        raise FileExistsError(f"Use a new output directory; {run_dir} is not empty")
    (run_dir / "config.json").write_text(json.dumps(
        {**asdict(c), "warm_start": str(warm_start) if warm_start else None,
         "env_path": env_path, "port": port}, indent=2), encoding="utf-8")
    if env is None:
        from godot_rl.core.godot_env import GodotEnv
        print("Start Godot Play when the connector asks for it.", flush=True)
        env = GodotEnv(env_path=env_path, port=port, show_window=True, seed=c.seed)
    step = decision = updates = life_steps = reset_retries = 0
    life = 1
    lifetimes = []
    hp_sum = 0.0
    first_death = None
    learner = None
    stop = False
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def stop_after_transition(_signal, _frame):
        nonlocal stop
        stop = True

    try:
        specs = extract_action_specs(env, learn_shoot=False)
        table = motor_table(specs)
        learner = Learner(c, len(table), device)
        if warm_start:
            learner.load(Path(warm_start), specs)
        print(f"v2 {device}, {len(table)} joint actions; external reward discarded; "
              f"hold={c.hold_steps}, HP unit={c.hp_unit}; output={run_dir}", flush=True)
        print("Action order:", [s.name for s in specs], flush=True)
        print("New run counters; physical initial state comes from Godot.reset(). "
              "The current scene may preserve an already-live body.")
        memory = SensoryMemory(c, device, len(table))
        observation, retries = reset_live(env)
        reset_retries += retries
        state = memory.state(observation, None, 0)
        learner.save(run_dir / "initial.pt", specs, 0)
        signal.signal(signal.SIGINT, stop_after_transition)
        signal.signal(signal.SIGTERM, stop_after_transition)
        next_save, next_log = c.save_every, c.log_every
        with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, FIELDS)
            writer.writeheader()
            while step < c.max_steps and not stop:
                action = learner.act(state)
                hp = extract_hp(observation, "hp")
                previous_hp = hp
                valence = pleasure = pain = 0.0
                positive_events = negative_events = 0
                elapsed = 0
                terminal = False
                source = "none"
                # Scalar accumulation only: no frame sequence or replay buffer.
                for j in range(min(c.hold_steps, c.max_steps - step)):
                    obs_list, _unused_reward, terminated, truncated, _unused_info = env.step(
                        [table[action]], order_ij=True)
                    observation = obs_list[0]
                    next_hp = extract_hp(observation, "hp")
                    done = bool(terminated[0] or truncated[0])
                    if done and next_hp > 0:
                        raise RuntimeError(
                            "Godot sent done with positive HP. This scene cannot force-respawn "
                            "a live body via reset; distinguish timeout/veto/autoreset in Godot "
                            "before treating this as death. No death update was applied.")
                    terminal = done or next_hp <= 0
                    source = "hp_zero" if next_hp <= 0 else ("done" if done else "none")
                    outcome = body_signal(previous_hp, next_hp, terminal, c)
                    valence += c.gamma**j * outcome["valence"]
                    pleasure += c.gamma**j * outcome["pleasure"]
                    pain += c.gamma**j * outcome["pain"]
                    # Diagnostics are based only on HP, no color/event labels.
                    positive_events += int(next_hp - previous_hp > 0.1)
                    negative_events += int(next_hp - previous_hp < -0.1)
                    hp_sum += max(next_hp, 0.0)
                    elapsed += 1
                    step += 1
                    life_steps += 1
                    body_event = abs(next_hp - previous_hp) >= 0.25 * c.hp_unit
                    previous_hp = next_hp
                    if terminal or stop or body_event:
                        break
                next_state = None if terminal else memory.state(observation, action, elapsed)
                if terminal and first_death is None:
                    # This snapshot includes live learning but excludes all death updates.
                    learner.save(run_dir / "before_first_death.pt", specs, step)
                    first_death = step
                metrics = learner.learn(state, action, valence, next_state, elapsed, terminal)
                updates += int(not c.no_learn)
                decision += 1
                image_change = 0.0 if terminal else float(
                    (next_state.image - state.image).abs().mean().item())
                writer.writerow({"step": step, "decision": decision, "life": life,
                    "life_step": life_steps, "elapsed": elapsed, "hp": hp,
                    "next_hp": next_hp, "hp_delta": next_hp - hp,
                    "action": json.dumps(dict(zip([s.name for s in specs], table[action]))),
                    "valence": valence, "pleasure": pleasure, "pain": pain,
                    **metrics, "terminal": int(terminal), "terminal_source": source,
                    "positive_hp_events": positive_events, "negative_hp_events": negative_events,
                    "reset_retries": reset_retries, "image_change": image_change, "updates": updates})
                if terminal:
                    lifetimes.append(life_steps)
                    print(f"death step={step}, life={life}, length={life_steps}, "
                          f"recent median={np.median(lifetimes[-10:]):.0f}", flush=True)
                    life += 1
                    life_steps = 0
                    learner.clear()
                    memory.clear()
                    f.flush()
                    if step < c.max_steps and not stop:
                        observation, retries = reset_live(env)
                        reset_retries += retries
                        state = memory.state(observation, None, 0)
                else:
                    state = next_state
                if c.save_every > 0 and step >= next_save:
                    learner.save(run_dir / f"step_{step:09d}.pt", specs, step)
                    learner.save(run_dir / "latest.pt", specs, step)
                    next_save = (step // c.save_every + 1) * c.save_every
                if c.log_every > 0 and step >= next_log:
                    print(f"step={step} hp={next_hp:.3f} alive={life_steps} "
                          f"entropy={metrics['entropy']:.3f}/{math.log(len(table)):.3f} "
                          f"TD={metrics['td']:+.4f} updates={updates}", flush=True)
                    next_log = (step // c.log_every + 1) * c.log_every
                    f.flush()
        learner.save(run_dir / "latest.pt", specs, step)
        summary = {"steps": step, "decisions": decision, "updates": updates,
                   "first_death_step": first_death, "deaths": len(lifetimes),
                   "completed_lifetimes": lifetimes, "censored_lifetime": life_steps,
                   "mean_hp": hp_sum / max(1, step), "reset_retries": reset_retries,
                   "environment_reward_used": False, "no_learn": c.no_learn}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Finished. Metrics and checkpoints: {run_dir}", flush=True)
        return summary
    except Exception:
        if learner is not None:
            learner.save(run_dir / "interrupted.pt", specs, step)
        raise
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        env.close()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    defaults = Config()
    for key, default in asdict(defaults).items():
        name = "--" + key.replace("_", "-")
        if key == "recurrent_size":
            p.add_argument(name, type=int, nargs="?", const=256, default=default,
                           help="Enable FC1 memory; defaults to 256 units when flag has no value; 0 disables")
        elif isinstance(default, bool):
            p.add_argument(name, action="store_true", default=default)
        else:
            p.add_argument(name, type=type(default), default=default)
    p.add_argument("--env-path")
    p.add_argument("--port", type=int, default=11008)
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--warm-start", type=Path)
    args = p.parse_args(argv)
    config = Config(**{k: getattr(args, k) for k in asdict(defaults)})
    return config, args


if __name__ == "__main__":
    config, args = parse_args()
    directory = args.run_dir or Path("runs") / f"sdaea_v2_{time.time_ns()}"
    run(config, directory, args.env_path, args.port, args.warm_start)

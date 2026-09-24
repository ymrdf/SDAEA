#!/usr/bin/env python3
"""SDAEA v2: streaming visual actor-critic driven only by the robot's body.

Run: python sdaea_streaming_v2.py --run-dir runs/v2_seed7 --seed 7
Then press Play in Godot. Dependencies and eye decoding are shared with v1.

No environment rewards, replay buffer, trajectory batches, or world snapshots.
One online update follows each motor decision (default: hold for 4 env.step calls).
Only the current transition, sensory EMA and parameter-sized eligibility traces
are retained. A single signed critic estimates future internal valence; positive
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
    memory_steps: float = 20.0
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
    for name in ("threads", "max_steps", "hold_steps", "hp_unit", "width",
                 "image_size", "eye_height", "eye_width", "actor_lr", "critic_lr",
                 "actor_kappa", "critic_kappa", "memory_steps"):
        if getattr(c, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if not 0 < c.gamma < 1 or not 0 <= c.exploration < 1 or c.trace_steps < 0:
        raise ValueError("Require 0<gamma<1, 0<=exploration<1, trace_steps>=0")
    if min(c.alive, c.death_cost, c.shaping, c.entropy) < 0:
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
    valence = survival - death + potential_change
    return {
        "valence": valence,
        "pleasure": survival + max(potential_change, 0.0),
        "pain": death + max(-potential_change, 0.0),
        "potential_change": potential_change,
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
        return State(image.detach(), self.ema.detach().clone(), body)


class VisualNetwork(nn.Module):
    """Owns its entire visual-to-output path; no detached learned world encoder."""

    def __init__(self, n_actions: int, outputs: int, width: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, 16, 5, stride=2, padding=2), nn.LeakyReLU(0.1),
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((4, 6)), nn.Flatten(),
        )
        # Raw spatial RGB average/max paths retain small bright objects while the
        # learned representation is immature. Every RGB channel is treated equally.
        inputs = 24 * 4 * 6 + 4 * 6 * 6 * 8 + 3 + n_actions
        self.hidden = nn.Sequential(
            nn.Linear(inputs, width), nn.LayerNorm(width, elementwise_affine=False),
            nn.LeakyReLU(0.1),
            nn.Linear(width, width), nn.LayerNorm(width, elementwise_affine=False),
            nn.LeakyReLU(0.1),
        )
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

    def forward(self, s: State):
        small = F.adaptive_avg_pool2d(s.image, (6, 8))
        peak = F.adaptive_max_pool2d(s.image, (6, 8))
        spatial = torch.cat((small * 2 - 1, peak * 2 - 1,
                             s.history * 2 - 1, 2 * (small - s.history)), dim=1)
        features = torch.cat((self.conv(s.image * 2 - 1),
                              spatial.flatten(1), s.body), dim=1)
        return self.output(self.hidden(features))


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
        for trace, grad in zip(self.traces, gradients):
            trace.mul_(decay).add_(grad.detach())
        l1 = sum(t.abs().sum() for t in self.traces).item()
        if not math.isfinite(l1):
            raise FloatingPointError("Non-finite eligibility trace")
        denominator = max(1.0, self.lr * self.kappa * max(1.0, abs(delta)) * l1)
        effective_lr = self.lr / denominator
        for p, trace in zip(self.parameters, self.traces):
            p.add_(trace, alpha=effective_lr * delta)
        return {"lr": effective_lr, "trace_l1": l1,
                "update_l1": effective_lr * abs(delta) * l1}

    @torch.no_grad()
    def clear(self):
        for t in self.traces:
            t.zero_()


class Learner:
    def __init__(self, c: Config, n_actions: int, device: torch.device):
        self.c, self.n_actions, self.device = c, n_actions, device
        self.actor = VisualNetwork(n_actions, n_actions, c.width).to(device)
        self.critic = VisualNetwork(n_actions, 1, c.width).to(device)
        self.actor_update = BoundedTrace(self.actor.parameters(), c.actor_lr, c.actor_kappa)
        self.critic_update = BoundedTrace(self.critic.parameters(), c.critic_lr, c.critic_kappa)
        self.trace_discount = 0.0 if c.trace_steps == 0 else math.exp(-1 / c.trace_steps)
        self.previous_elapsed = None

    def distribution(self, s: State):
        probs = torch.softmax(self.actor(s), -1)
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
        with torch.no_grad():
            bootstrap = 0.0 if terminal else float(self.critic(next_state).item())
        with torch.set_grad_enabled(not self.c.no_learn):
            value = self.critic(state).squeeze()
            dist = self.distribution(state)
            log_prob = dist.log_prob(torch.tensor([action], device=self.device)).sum()
            entropy = dist.entropy().sum()
            delta = valence + (0.0 if terminal else discount * bootstrap) - value.item()
            metrics = {"td": delta, "value": value.item(), "bootstrap": bootstrap,
                       "entropy": entropy.item(), "max_probability": dist.probs.max().item()}
            if self.c.no_learn:
                return metrics
            # Entropy strength vanishes with the TD error, avoiding a constant push
            # back to uniform when bodily advantages are tiny.
            objective = log_prob + self.c.entropy * np.sign(delta) * entropy
            actor_grads = torch.autograd.grad(objective, self.actor_update.parameters)
            critic_grads = torch.autograd.grad(value, self.critic_update.parameters)
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

    def clear(self):
        self.actor_update.clear()
        self.critic_update.clear()
        self.previous_elapsed = None

    def save(self, path: Path, specs, step: int):
        atomic_torch_save({"version": 2, "config": asdict(self.c), "step": step,
                           "specs": [asdict(s) for s in specs],
                           "actor": self.actor.state_dict(),
                           "critic": self.critic.state_dict()}, path)

    def load(self, path: Path, specs):
        payload = torch.load(path, map_location=self.device, weights_only=True)
        if payload.get("version") != 2 or payload["specs"] != [asdict(s) for s in specs]:
            raise ValueError("Checkpoint version/action space mismatch (v1 weights are incompatible)")
        for key in ("hp_unit", "gamma", "shaping", "alive", "death_cost", "width",
                    "hold_steps", "memory_steps", "image_size", "exploration"):
            if payload["config"][key] != getattr(self.c, key):
                raise ValueError(f"Checkpoint {key}={payload['config'][key]} differs from CLI")
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
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
        if isinstance(default, bool):
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

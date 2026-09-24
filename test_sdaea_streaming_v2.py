"""CPU checks and a small learning probe; none require a Godot connection.

python -m unittest -v test_sdaea_streaming_v2
python test_sdaea_streaming_v2.py --probe --steps 1500

The probe is a visual two-choice energy task, NOT evidence of Godot survival.
Its colors are only rendered inputs; their consequence map is private to the
fixture and reversed halfway through. The learner receives body-derived affect.
"""
import argparse
import contextlib
import csv
import io
import json
import math
import tempfile
import unittest
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from sdaea_streaming_v2 import (
    BoundedTrace, Config, Learner, SensoryMemory, State,
    body_signal, motor_table, reset_live, run,
)
from sdaea_online_validate import ActionSpec, seed_everything


def picture(green_left, rng):
    pixels = np.full((24, 32, 3), 30, np.uint8)
    brightness = rng.integers(160, 255)
    green, red = [0, brightness, 0], [brightness, 0, 0]
    pixels[6:20, 3:12] = green if green_left else red
    pixels[6:20, 20:29] = red if green_left else green
    return pixels


def observation(hp=5.0, green_left=True, rng=None):
    rng = rng if rng is not None else np.random.default_rng(42)
    eye = picture(green_left, rng)
    return {"left_eye": eye, "right_eye": eye.copy(), "hp": [hp]}


class FakeGodot:
    def __init__(self, reward=0.0):
        self.num_envs = 1
        self.action_spaces = [SimpleNamespace(spaces=OrderedDict(
            (name, SimpleNamespace(n=n)) for name, n in (
                ("accelerate_forward", 3), ("accelerate_sideways", 3),
                ("shoot", 2), ("turn", 3))))]
        self.reward = reward
        self.hp = 5.0
        self.steps = self.resets = 0

    def reset(self):
        self.resets += 1
        # Reproduce the dead reset sample that contaminated the old log.
        self.hp = -4 if self.resets == 1 else 5.0
        return [observation(self.hp)], [{}]

    def step(self, actions, order_ij=False):
        assert order_ij and len(actions[0]) == 4
        assert actions[0][2] == 0
        self.steps += 1
        self.hp -= 2
        done = self.hp <= 0
        return [observation(self.hp)], [self.reward], [done], [done], [{}]

    def close(self):
        pass


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_potential_telescopes(self):
        c = Config(alive=0.0, death_cost=0.0)
        hp = [5.0, 9.0, 2.0, 6.0, 0.0]
        result = sum(c.gamma**i * body_signal(a, b, i == 3, c)["valence"]
                     for i, (a, b) in enumerate(zip(hp, hp[1:])))
        self.assertAlmostEqual(result, -math.log1p(hp[0] / c.hp_unit), places=10)

    def test_high_hp_signal_and_terminal(self):
        c = Config()
        heal = body_signal(90, 95, False, c)
        damage = body_signal(90, 85, False, c)
        self.assertGreater(heal["valence"], damage["valence"] + 0.05)
        death = body_signal(5, 100, True, c)  # auto-reset HP must never yield pleasure
        self.assertEqual(death["pleasure"], 0)
        self.assertLess(death["valence"], -c.death_cost)

    def test_trace_delay_bound_and_terminal_clear(self):
        p = torch.nn.Parameter(torch.tensor([0.0]))
        updater = BoundedTrace([p], 1, 3)
        updater.step([torch.ones(1)], 0, 0.9)
        for _ in range(20):
            updater.step([torch.zeros(1)], 0, 0.9)
        self.assertAlmostEqual(updater.traces[0].item(), 0.9**20, places=6)
        stats = updater.step([torch.zeros(1)], 1000, 1.0)
        self.assertGreater(p.item(), 0)
        self.assertLessEqual(stats["update_l1"], 1/3 + 1e-6)
        updater.clear()
        self.assertEqual(updater.traces[0].item(), 0)

    def test_actor_update_reaches_vision(self):
        seed_everything(7)
        c = Config(width=24, image_size=24, hold_steps=1)
        learner = Learner(c, 2, torch.device("cpu"))
        memory = SensoryMemory(c, torch.device("cpu"), 2)
        s = memory.state(observation(), None, 0)
        weights = learner.actor.conv[0].weight.detach().clone()
        learner.learn(s, 0, 1.0, s, 1, False)
        self.assertFalse(torch.equal(weights, learner.actor.conv[0].weight))
        self.assertTrue(all(t.grad_fn is None for t in learner.actor_update.traces))
        self.assertGreater(sum(t.abs().sum().item() for t in learner.actor_update.traces), 0)
        metrics = learner.learn(s, 0, -1.0, None, 1, True)
        self.assertEqual(metrics["bootstrap"], 0)
        self.assertTrue(all(torch.count_nonzero(t) == 0 for t in learner.actor_update.traces))

    def test_action_duration_uses_previous_interval(self):
        c = Config(width=24, image_size=24)
        learner = Learner(c, 2, torch.device("cpu"))
        memory = SensoryMemory(c, torch.device("cpu"), 2)
        s = memory.state(observation(), None, 0)
        learner.learn(s, 0, 0.1, s, 4, False)
        seen = []
        original = learner.actor_update.step
        def capture(grads, delta, decay):
            seen.append(decay)
            return original(grads, delta, decay)
        learner.actor_update.step = capture
        learner.learn(s, 1, -0.1, None, 1, True)
        self.assertAlmostEqual(seen[0], (c.gamma * learner.trace_discount)**4)

    def test_mock_full_loop_reward_independence(self):
        c = Config(max_steps=7, width=24, image_size=24, hold_steps=2,
                   threads=1, save_every=0, log_every=0)
        payloads = []
        for reward in (-12345.0, 987654.0):
            with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                summary = run(c, Path(tmp), env=FakeGodot(reward))
                self.assertEqual(summary["reset_retries"], 1)
                self.assertEqual(summary["completed_lifetimes"], [3, 3])
                self.assertEqual(summary["censored_lifetime"], 1)
                self.assertTrue((Path(tmp) / "before_first_death.pt").is_file())
                payloads.append(torch.load(Path(tmp) / "latest.pt", weights_only=True))
        for key in ("actor", "critic"):
            for name in payloads[0][key]:
                self.assertTrue(torch.equal(payloads[0][key][name], payloads[1][key][name]))

    def test_frozen_warm_start_preserves_parameters(self):
        c = Config(max_steps=2, width=24, image_size=24, threads=1,
                   save_every=0, log_every=0)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            first, second = Path(tmp) / "train", Path(tmp) / "eval"
            run(c, first, env=FakeGodot())
            summary = run(replace(c, no_learn=True), second,
                          warm_start=first / "latest.pt", env=FakeGodot())
            self.assertEqual(summary["updates"], 0)
            a = torch.load(first / "latest.pt", weights_only=True)
            b = torch.load(second / "latest.pt", weights_only=True)
            for key in ("actor", "critic"):
                for name in a[key]:
                    self.assertTrue(torch.equal(a[key][name], b[key][name]))

    def test_positive_hp_done_cannot_fabricate_death(self):
        class AmbiguousDone(FakeGodot):
            def step(self, actions, order_ij=False):
                return [observation(4)], [1000], [True], [True], [{}]
        c = Config(max_steps=2, width=24, image_size=24, threads=1)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "positive HP"):
                run(c, Path(tmp), env=AmbiguousDone())
            a = torch.load(Path(tmp) / "initial.pt", weights_only=True)
            b = torch.load(Path(tmp) / "interrupted.pt", weights_only=True)
            for name in a["actor"]:
                self.assertTrue(torch.equal(a["actor"][name], b["actor"][name]))

    def test_first_substep_death_stops_hold(self):
        class ImmediateDeath(FakeGodot):
            def step(self, actions, order_ij=False):
                self.steps += 1
                return [observation(-1)], [0], [True], [True], [{}]
        c = Config(max_steps=1, hold_steps=4, width=24, image_size=24, threads=1)
        env = ImmediateDeath()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            summary = run(c, Path(tmp), env=env)
            self.assertEqual(env.steps, 1)
            self.assertEqual(summary["decisions"], 1)
            self.assertEqual(summary["completed_lifetimes"], [1])


def learning_probe(steps=1500, seed=7, gamma=0.9, trace_steps=0.0):
    seed_everything(seed)
    torch.set_num_threads(1)
    # Shorter credit horizon fits this 1-step toy, unlike the physical navigation task.
    c = Config(seed=seed, width=32, image_size=24, gamma=gamma,
               trace_steps=trace_steps, hold_steps=1, exploration=0.05)
    learner = Learner(c, 2, torch.device("cpu"))
    rng = np.random.default_rng(seed)

    def score(reverse):
        # Independent observations, frozen parameters and state; no eval learning.
        correct = 0
        eval_rng = np.random.default_rng(1001)
        for i in range(200):
            green_left = bool(i % 2)
            memory = SensoryMemory(c, torch.device("cpu"), 2)
            s = memory.state(observation(5, green_left, eval_rng), None, 0)
            with torch.no_grad():
                chosen = int(learner.distribution(s).probs.argmax().item())
            target = (0 if green_left else 1) ^ int(reverse)
            correct += chosen == target
        return correct / 200

    report = {"seed": seed, "steps_per_phase": steps, "gamma": gamma,
              "trace_steps": trace_steps, "initial_accuracy": score(False)}
    for reverse in (False, True):
        memory = SensoryMemory(c, torch.device("cpu"), 2)
        hp = 5.0
        green_left = bool(rng.integers(2))
        s = memory.state(observation(hp, green_left, rng), None, 0)
        deaths = 0
        for _ in range(steps):
            action = learner.act(s)
            correct_action = (0 if green_left else 1) ^ int(reverse)
            next_hp = min(10, hp + (0.5 if action == correct_action else -0.5))
            terminal = next_hp <= 0
            affect = body_signal(hp, next_hp, terminal, c)
            next_green_left = bool(rng.integers(2))
            next_state = None if terminal else memory.state(
                observation(next_hp, next_green_left, rng), action, 1)
            learner.learn(s, action, affect["valence"], next_state, 1, terminal)
            if terminal:
                deaths += 1
                hp = 5.0
                memory.clear()
                s = memory.state(observation(hp, next_green_left, rng), None, 0)
            else:
                s, hp = next_state, next_hp
            green_left = next_green_left
        report["reversed_accuracy" if reverse else "learned_accuracy"] = score(reverse)
        report["reversed_deaths" if reverse else "training_deaths"] = deaths
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--probe-gamma", type=float, default=0.9)
    parser.add_argument("--probe-trace-steps", type=float, default=0.0)
    args = parser.parse_args()
    if args.probe:
        print(json.dumps(learning_probe(args.steps, args.seed, args.probe_gamma,
                                        args.probe_trace_steps), indent=2))
    else:
        unittest.main(argv=[__file__])

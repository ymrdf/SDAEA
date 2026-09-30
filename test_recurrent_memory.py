"""Temporal-gradient, lifecycle, and delayed-cue checks; no Godot required."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import torch
from sdaea_streaming_v2 import Config, Learner, State, DecayingMemory, parse_args


class RecurrentChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        self.c = Config(width=24, image_size=24, recurrent_size=12, bptt_steps=4)

    def state(self, elapsed=4):
        return State(torch.rand(1, 6, 24, 24), torch.zeros(1, 6, 6, 8),
                     torch.zeros(1, 5), elapsed)

    def test_fc1_and_later_layers_use_memory(self):
        l = Learner(replace(self.c, depth=3), 2, torch.device('cpu'))
        net = l.actor
        seen = []
        hook = net.hidden[1].register_forward_pre_hook(
            lambda module, args: seen.append(args[0].detach().clone()))
        s = self.state()
        out_zero, _ = net(s, torch.zeros(1, 12))
        previous = torch.ones(1, 12, requires_grad=True)
        out_memory, _ = net(s, previous)
        hook.remove()
        self.assertFalse(torch.equal(seen[0], seen[1]))
        self.assertFalse(torch.equal(out_zero, out_memory))
        gradients = torch.autograd.grad(out_memory[0, 0],
            [previous, net.memory_feedback.weight, net.memory_read.weight,
             net.hidden[3].weight, net.hidden[6].weight])
        self.assertTrue(all(g.abs().sum().item() > 0 for g in gradients))

    def test_network_applies_physical_decay_once(self):
        net = Learner(self.c, 2, torch.device('cpu')).actor
        with torch.no_grad():
            net.memory.gates.weight.zero_()
            net.memory.gates.bias[:12].fill_(-100.)
        _, h = net(self.state(20), torch.ones(1, 12))
        torch.testing.assert_close(h, torch.full_like(h, torch.exp(torch.tensor(-.1))))

    def test_cli_enables_256_units_explicitly(self):
        self.assertEqual(parse_args([])[0].recurrent_size, 0)
        self.assertEqual(parse_args(['--recurrent-size'])[0].recurrent_size, 256)
        self.assertEqual(parse_args(['--recurrent-size', '512'])[0].recurrent_size, 512)

    def test_temporal_gradient_reaches_past_observation(self):
        l = Learner(self.c, 2, torch.device('cpu'))
        past, now = self.state(0), self.state()
        past.image.requires_grad_()
        l.context = [past]
        logits, _, _ = l.evaluate('actor', now)
        gradient, = torch.autograd.grad(logits[0, 0], past.image)
        self.assertGreater(gradient.abs().sum().item(), 0)

    def test_online_updates_bounded_history_and_reset(self):
        l = Learner(self.c, 2, torch.device('cpu'))
        originals = {k: p.detach().clone() for k, p in l.actor.memory.named_parameters()}
        s = self.state(0)
        for i in range(12):
            nxt = self.state(1 if i % 2 else 4)
            l.act(s)
            l.learn(s, i % 2, 1., nxt, nxt.elapsed, False)
            s = nxt
            self.assertLess(len(l.context), self.c.bptt_steps)
        for k, p in l.actor.memory.named_parameters():
            self.assertFalse(torch.equal(p, originals[k]), k)
        self.assertTrue(all(h.grad_fn is None for h in l.anchors.values()))
        l.learn(s, 0, -2., None, 1, True)
        self.assertEqual(l.context, [])
        self.assertTrue(all(h is None for h in l.anchors.values()))

    def test_frozen_state_advances_but_queries_do_not(self):
        l = Learner(replace(self.c, no_learn=True), 2, torch.device('cpu'))
        weights = {k: v.clone() for k, v in l.actor.state_dict().items()}
        s = self.state(0)
        a = l.distribution(s).probs
        torch.testing.assert_close(a, l.distribution(s).probs)
        self.assertEqual(len(l.context), 0)
        l.learn(s, 0, 0., self.state(), 4, False)
        self.assertEqual(len(l.context), 1)
        for k, v in l.actor.state_dict().items():
            self.assertTrue(torch.equal(v, weights[k]))
        l.learn(self.state(), 0, -1., None, 1, True)
        self.assertEqual(len(l.context), 0)

    def test_decay_and_bounded_state(self):
        cell = DecayingMemory(2, 4, 20.)
        with torch.no_grad():
            cell.gates.weight.zero_()
            cell.gates.bias[:4].fill_(-100.)  # Shut writing to isolate decay.
        _, short = cell(torch.zeros(1, 2), torch.ones(1, 4), 4)
        _, long = cell(torch.zeros(1, 2), torch.ones(1, 4), 20)
        torch.testing.assert_close(short, torch.full_like(short, torch.exp(torch.tensor(-.2))))
        self.assertTrue((long < short).all())
        h = None
        for _ in range(100):
            _, h = cell(torch.randn(1, 2) * 100, h, 4)
        self.assertLessEqual(h.abs().max().item(), 1.)

    def test_checkpoint_and_configuration_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'weights.pt'
            l = Learner(self.c, 2, torch.device('cpu'))
            l.save(p, [], 0)
            l.load(p, [])
            old = torch.load(p, weights_only=True)
            old.pop('recurrent_architecture')
            torch.save(old, p)
            with self.assertRaisesRegex(ValueError, 'recurrent architecture'):
                l.load(p, [])
            baseline = Learner(replace(self.c, recurrent_size=0), 2, torch.device('cpu'))
            with self.assertRaisesRegex(ValueError, 'recurrent_size'):
                baseline.load(p, [])

    def test_delayed_cue_can_be_learned(self):
        # Supervised diagnostic only: two opposite cues, then six identical blanks.
        cell = DecayingMemory(2, 12, 40.)
        head = torch.nn.Linear(12, 2)
        opt = torch.optim.Adam(list(cell.parameters()) + list(head.parameters()), lr=.02)
        cues = torch.tensor([[1., 0.], [-1., 0.]])
        labels = torch.tensor([0, 1])
        for _ in range(180):
            read, h = cell(cues, None, 0)
            for _ in range(6):
                read, h = cell(torch.zeros_like(cues), h, 4)
            loss = torch.nn.functional.cross_entropy(head(read), labels)
            opt.zero_grad(); loss.backward(); opt.step()
        self.assertEqual(head(read).argmax(-1).tolist(), [0, 1])
        self.assertLess(loss.item(), .05)


if __name__ == '__main__':
    unittest.main()

"""Architecture and temporal-batching correctness for the shared-arena search."""
import unittest
from dataclasses import replace
import torch
from sdaea_streaming_v2 import Config, Learner, State, validate


class SearchChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(29)
        self.c = Config(width=24, depth=3, image_size=24, recurrent_size=12)

    def state(self, elapsed=4):
        return State(torch.rand(1, 6, 24, 24), torch.rand(1, 6, 6, 8),
                     torch.rand(1, 5), elapsed)

    def test_batched_window_matches_serial_outputs_and_gradients(self):
        l = Learner(self.c, 2, torch.device('cpu'))
        sequence = [self.state(0), self.state(1), self.state(4)]
        l.context = sequence[:-1]
        output, h, first = l.evaluate('actor', sequence[-1])
        batched = torch.autograd.grad(output.sum(), list(l.actor.parameters()))
        previous = None
        for index, s in enumerate(sequence):
            expected, previous = l.actor(s, previous)
            if index == 0:
                expected_first = previous
        serial = torch.autograd.grad(expected.sum(), list(l.actor.parameters()))
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(h, previous)
        torch.testing.assert_close(first, expected_first)
        for a, b in zip(batched, serial):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-4)

    def test_all_memory_locations_and_extra_cnn_receive_gradients(self):
        for layer in range(3):
            c = replace(self.c, recurrent_layer=layer, cnn_extra=1)
            validate(c)
            l = Learner(c, 2, torch.device('cpu'))
            s = self.state()
            l.context = [self.state(0)]
            output, _, _ = l.evaluate('actor', s)
            gradients = torch.autograd.grad(output[0, 0],
                [l.actor.conv[4].weight, l.actor.memory_feedback.weight,
                 l.actor.hidden[layer*3].weight])
            self.assertTrue(all(g.abs().sum().item() > 0 for g in gradients))

    def test_frozen_fast_path_matches_explicit_recurrence(self):
        l = Learner(replace(self.c, no_learn=True), 2, torch.device('cpu'))
        sequence = [self.state(0), self.state(4), self.state(1), self.state(4)]
        h = None
        for index, s in enumerate(sequence):
            with torch.no_grad():
                logits, h = l.actor(s, h)
                expected = l.policy(logits).probs
                actual = l.distribution(s).probs
            torch.testing.assert_close(expected, actual)
            terminal = index == len(sequence)-1
            nxt = None if terminal else sequence[index+1]
            l.learn(s, 0, 0., nxt, 4 if terminal else nxt.elapsed, terminal)
        self.assertEqual(l.context, [])

    def test_gray_ablation_masks_current_and_historical_pixels(self):
        from run_survival_search import PolicyMemory
        from test_sdaea_streaming_v2 import observation
        c=replace(self.c, eye_height=24, eye_width=32)
        a=PolicyMemory(c,torch.device('cpu'),2,vision_ablation='gray')
        b=PolicyMemory(c,torch.device('cpu'),2,vision_ablation='gray')
        for step in range(3):
            left=a.state(observation(5,True),0,4)
            right=b.state(observation(5,False),0,4)
            torch.testing.assert_close(left.image,right.image)
            torch.testing.assert_close(left.history,right.history)
            torch.testing.assert_close(left.body,right.body)
            self.assertTrue((left.image==.5).all())
            self.assertTrue((left.history==.5).all())
            self.assertEqual(left.elapsed,4)
        self.assertFalse(torch.equal(a.ema,b.ema))  # Hidden raw history never reaches policy.

    def test_invalid_insertion_layer_rejected(self):
        with self.assertRaises(ValueError):
            validate(replace(self.c, recurrent_layer=3))


if __name__ == '__main__':
    unittest.main()

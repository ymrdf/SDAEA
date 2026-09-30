# Independent-policy training batches

The shared-scene survival runner now enables `--batch-train` by default.
Use `--no-batch-train` for the original serial reference. `--batch-eval` enables
frozen actor batching; `--transport gpu` selects GPU vision independently.

`training_policy_batch.py` groups learners by network architecture and current
BPTT history length. It stacks independent parameters differentiably, uses vmap
for actor/critic forward passes and bootstrapping, and backpropagates the sum of
independent objectives to each original model. This is not shared-policy training.
There is no averaging of gradients, rewards or hidden states across robots.

Each learner retains its original optimizer, moments, eligibility traces, decay,
learning rate, body-only target and recurrent anchor. Death clears only that
learner's history. Checkpoints continue to use the existing Learner format.
Optimizer steps remain independent foreach operations; forward/backward work is
batched. CPU copies of policy probabilities and diagnostics are combined per
group. Fused Adam was measured and was slower on the installed GPU, so it is not
used. Adam's gradient scaling uses foreach to avoid separate small kernel launches.

CPU/CUDA regression tests compare serial and batch probabilities, parameter
updates, anchors, trace buffers and Adam moments across seven transitions,
including mixed optimizers, rates, elapsed times and selective deaths. Small
floating-point differences from batch kernels are expected, not bitwise identity.
Run `python -m unittest discover -s . -p 'test*.py' -q` in SDAEA.

Performance runs use the same round04 configuration, 16 simultaneous robots,
seed83, GPU transport, 100 initial + 600 training + 200 final frozen shared steps:
- Serial: runs/gpu_multi_benchmark_20260930
- Initial batch: runs/batch_train_benchmark_20260930
- Fused-Adam comparison (not retained): runs/batch_fused_verified_20260930
- Final implementation: runs/batch_train_final_20260930

These are throughput smoke tests, not survival-quality evidence or repeated
controlled performance trials. The long experiment and heartbeat stay paused.

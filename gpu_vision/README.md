# Multi-agent GPU vision

Linux Vulkan → OPAQUE_FD → CUDA image transport. Each agent exports its own two
RGBA8 textures on a separate Unix connection. The handshake carries `agent_index`
matching Godot's `agents_training` order. Every observation carries this index and
a monotonically increasing frame ID; Python tracks frame IDs separately per agent.
The image pixels remain on the GPU, with a GPU-to-GPU copy into owned PyTorch
buffers. Control and HP metadata still use the CPU/TCP connection.

Build the Godot extension after native changes:

```
cd ../EnvolutionRobot/native/gpu_vision_bridge
scons -j4 platform=linux target=template_debug
```

For the same-scene survival runner, add `--fast --transport gpu` to the normal
command. Default remains `raw`. Camera sizes come from each robot's eye sensors.
All shared images are copied before one rendering flush; consumers finish copying
before the next action request lets Godot overwrite the shared textures.
Full reset, selective death reset, observe, RGB steps and HP-only steps are supported.
This implementation retains blocking GPU synchronization; it does not batch learning
updates and is not an asynchronous zero-copy policy input.

For correctness diagnostics ONLY, add `--gpu-verify`: every delivered GPU eye is
compared byte-for-byte with CPU readback of the same robot, eye and frame. Counts
are saved in `gpu_pixel_checks.json`. This deliberately reintroduces CPU readback
and must be disabled for performance measurements or normal training.

Regression tests: `python -m unittest test_gpu_vision_transport test_fast_godot_env test_long_training`.
Runtime validation: `runs/gpu_multi_verified_20260930`, 16 agents, 140 shared steps,
2 selective resets, 1568 eye images equal to CPU readback. The first smoke failure
was preserved in `runs/gpu_multi_smoke_20260930`; it exposed the now-fixed missing
capture after selective reset. Long-running experiment/automation remains paused.

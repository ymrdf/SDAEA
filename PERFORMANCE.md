# Shared-arena transport and stepping

`run_competition.py --fast` runs all seven competitors in one arena with:

- `--fixed-fps 20 --disable-vsync --disable-render-loop`. Fixed simulation delta remains 0.05 seconds, but wall-clock synchronization is disabled. Rendering is explicitly requested at observation boundaries.
- HP and block-counter responses at every physical step. Within a four-step action hold, RGB is read only at the last step, or immediately after an early HP/death event. Event detection, discounting and terminal handling still occur at each physical step.
- A length-prefixed `RGB1` frame containing JSON metadata and raw RGB8 buffers. No hex, JPEG or lossy compression. Eye render resolution is configurable with `--eye-width` and `--eye-height`; per-model resizing remains independent. The first transport benchmarks below used 320×300.
- Physics frame assertions: each action advances exactly one tick; snapshot requests and selective respawns advance zero ticks. Snapshot HP must equal the HP-only response.
- Legacy JSON/hex remains the default for old clients. `--uncapped` benchmarks only wall-clock/render-loop changes with legacy full-RGB traffic.

`timings.json` contains per-phase wall time, physical steps, bytes received and RGB response count. All seven robots participate in every shared step. Timing excludes process startup; per-phase checkpoint writes are included.

`--warm-start-dir PATH --checkpoint-name interrupted.pt` restores all seven policies and skips the initial frozen phase. This is a weight continuation, not an exact physical-world resume: the arena, traces, sensory history and action RNG restart. Keep parent and child run results separate.

Godot flag reference: https://docs.godotengine.org/en/4.6/tutorials/editor/command_line_tutorial.html

## Measured on the current seven-robot setup

| Mode | Training sample | Shared steps/s | Projected 100,000 shared steps |
|---|---:|---:|---:|
| Legacy | 500 | 6.04 | 276 min |
| Uncapped only | 500 | 6.19 | 269 min |
| Uncapped + sparse RGB + raw bytes | 3,000 | 24.50 | 68 min |

The optimized training sample took 122.43 seconds. It returned 772 RGB observations instead of at least 3,000, including extra event/reset observations. Raw received data totaled 3.115 GB, about 87% below the theoretical legacy full-RGB hex payload (metadata excluded from that legacy denominator). Legacy byte counters were not instrumented. Projection is an estimate, not a completed 100,000-step timing; allow roughly 65–80 minutes depending on event frequency and machine load.

Results: `runs/competition_performance_comparison.json`. Protocol tests cover partial socket reads, disconnects, lossless pixels, scalar-only observations and physics-tick invariants. A separate warm-start verification checks actual rendered raw pixels against legacy hex in the paused scene and confirms all seven cameras change during motion.

## Camera resolution

The two camera SubViewports now render directly at the requested dimensions. Both raw RGB decoding and sensory preprocessing use the negotiated camera shape. This avoids rendering 320×300 and merely shrinking the image before sending it. The tested sizes all keep the original 16:15 aspect ratio, so camera framing is preserved.

Example: `--fast --eye-width 160 --eye-height 150`. Model inputs stay 48×48, 96×96 or 144×144 according to each competitor. Raw sizes below 144 pixels require upsampling for the largest-input policy and can discard useful small-object detail; throughput measurements alone do not establish policy quality.

Measured camera-size comparison (same checkpoint, 1,000 training steps each):

| Camera size | Shared steps/s | Estimated 100k training minutes |
|---|---:|---:|
| 320×300 | 25.33 | 65.80 |
| 160×150 | 28.54 | 58.40 |
| 128×120 | 29.02 | 57.42 |
| 96×90 | 30.00 | 55.55 |

Selected 160×150 as the trainer default: approximately 75% less image traffic and 11% less total time, while both source dimensions remain above the largest 144×144 model input. Other sizes remain selectable. All four passed raw/hex pixel agreement, changing-camera and physics tick checks. This is a throughput comparison, not a demonstration of unchanged learning quality. Results: `runs/camera_resolution_performance.json`.

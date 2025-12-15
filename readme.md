# Survival-Driven Adaptive Evolution Architecture (SDAEA)

A novel reinforcement learning framework for training AI agents in a Godot-based robot FPS combat environment. This project implements per-step online learning with self-regulated learning rates and death-driven negative replay.

## Overview

This project demonstrates a biologically-inspired training paradigm where agents learn to survive in a hostile environment through continuous adaptation. Unlike traditional RL approaches that separate training and inference, SDAEA performs parameter updates at every step, enabling true online learning even on resource-constrained devices.

## Key Features

- **Per-Step Online Learning**: Parameters are updated after every interaction step, not in batches
- **Self-Regulated Learning**: The model dynamically determines its own learning rate from internal signals
- **Death-Driven Negative Replay**: When HP reaches zero, cached parameters are restored and negative gradients are applied
- **Binocular Vision Processing**: Processes left and right eye observations (320x300 each) from Godot
- **MobileNetV3 Backbone**: Efficient neural architecture suitable for real-time processing
- **Multi-Agent Environment**: Supports up to 8 concurrent agents in the Godot simulation

## Environment

The Godot environment is a robot FPS combat simulation where agents must:

- Hit other robots (+1 reward)
- Avoid getting hit (2 HP per robot)
- Survive as long as possible

**Godot Environment Repository**: [https://github.com/ymrdf/EnvolutionRobot](https://github.com/ymrdf/EnvolutionRobot)

### Action Space

Each agent has 4 discrete action dimensions:

- `accelerate_forward`: 3 choices (backward, stay, forward)
- `accelerate_sideways`: 3 choices (left, stay, right)
- `turn`: 3 choices (left, stay, right)
- `shoot`: 2 choices (don't shoot, shoot)

### Observation Space

- **Left Eye**: (3, 300, 320) - RGB image from left camera
- **Right Eye**: (3, 300, 320) - RGB image from right camera
- **HP**: Scalar value representing current health

## Architecture

### Input Processing

1. Binocular images are stacked vertically: (3, 600, 320)
2. HP is embedded as a bar: (3, 40, 320)
3. Combined with previous model output (out_next): (3, 640, 320)
4. Final input shape: (3, 640, 640)

### Model Output

The model produces a (3, 640, 320) tensor containing:

- **Action logits**: Extracted from eleven 3×8×8 crops in the first row
- **Loss region**: Center 3×8×8 region used for computing loss
- **Learning rate region**: Adjacent 3×8×8 region determining the learning rate
- **Internal state**: The entire output feeds back as input for the next step

### Learning Mechanism

1. **Normal Step**:

   - Forward pass through MobileNetV3
   - Extract loss from designated region
   - Compute dynamic learning rate from LR region
   - Apply gradient update

2. **Death Event** (HP ≤ 0):
   - Restore parameters from cache (last 20 steps)
   - Apply negative loss gradient with large learning rate (1e-2)
   - Reset environment and continue training

## Installation

### Prerequisites

- Python 3.10+
- Conda (recommended)
- Godot 4.x with RL Agents plugin

### Setup Environment

# Create conda environment from file

conda env create -f environment.yml

# Activate environment

conda activate envolution### Key Dependencies

- `torch==2.2.2` - Deep learning framework
- `torchvision==0.17.2` - Pre-trained models (MobileNetV3)
- `godot-rl==0.8.2` - Godot RL integration
- `gymnasium==1.0.0` - RL environment interface
- `stable-baselines3==2.4.0` - RL algorithms (reference)

## Usage

### Training

1. Clone and open the Godot environment:
   git clone https://github.com/ymrdf/EnvolutionRobot

# Open the project in Godot Editor2. Run the training notebook:

jupyter notebook my_battle_zone_godot_train.ipynb3. Press **Play** in Godot Editor when prompted

4. The training will run with the following default hyperparameters:
   - Learning rate range: [1e-5, 5e-4]
   - Death penalty LR: 1e-2
   - Parameter cache interval: 20 steps
   - Max steps per episode: 100,000

### Trained Model

The final model is saved to:

# 🎮 Doom AI — Deep Convolutional Q-Learning

A reinforcement learning agent that learns to play Doom using a CNN-based Deep Q-Learning approach with n-step eligibility traces and experience replay.

## How It Works

1. **Image Preprocessing** — Raw game frames are resized to 80×80 grayscale images and normalized to `[0, 1]`.
2. **CNN Brain** — A 3-layer convolutional network extracts spatial features, followed by fully-connected layers that output Q-values for each action.
3. **Softmax Exploration** — Actions are sampled from a temperature-scaled softmax distribution over Q-values, balancing exploration and exploitation.
4. **N-Step Experience Replay** — The agent collects 10-step trajectories, stores them in a replay buffer (capacity 10,000), and trains on random mini-batches using eligibility trace targets.

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| 🧠 Deep Learning | PyTorch |
| 🎮 Environment | Gymnasium + ViZDoom |
| 🖼 Image Processing | Pillow (PIL) |
| 📊 Numerics | NumPy |

## Setup

```bash
# Install dependencies
pip install torch numpy gymnasium pillow vizdoom

# Run training
cd Doom
python ai.py
```

## Project Structure

```
Doom/
├── ai.py                  # CNN model, agent, training loop
├── experience_replay.py   # N-step replay buffer
└── image_preprocessing.py # Frame preprocessing wrapper
```

## ⚠️ Known Issues

- The original code targeted `ppaquette_gym_doom` which is abandoned. This version uses `VizdoomCorridor-v0` from the [ViZDoom](https://github.com/Farama-Foundation/ViZDoom) package — you may need to adjust the environment ID depending on your ViZDoom version.
- The `SkipWrapper` and `ToDiscrete` wrappers from the original code have been removed since ViZDoom environments handle frame skipping and discrete action spaces natively.
- Training hyperparameters (learning rate, epochs, reward threshold) are tuned for the original Doom Corridor environment and may need adjustment.

## License

MIT

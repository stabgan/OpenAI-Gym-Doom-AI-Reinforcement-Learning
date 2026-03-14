# 🎮 Doom AI — Deep Reinforcement Learning Agent

A deep reinforcement learning agent that learns to play the classic **Doom** using a Convolutional Neural Network (CNN) and Deep Q-Learning with experience replay and eligibility traces.

---

## 📐 Architecture

```
┌──────────────────────────────────────────────────────┐
│                   Doom Environment                   │
│              (Gymnasium + VizDoom)                    │
└──────────────┬───────────────────────┬───────────────┘
               │  raw frames          │  actions
               ▼                      │
┌──────────────────────────┐          │
│   Image Preprocessing    │          │
│  (resize → grayscale →   │          │
│   normalize to [0,1])    │          │
└──────────────┬───────────┘          │
               │  80×80 grayscale     │
               ▼                      │
┌──────────────────────────┐          │
│     CNN Brain            │          │
│  Conv2d(1→32, k=5)      │          │
│  Conv2d(32→32, k=3)     │          │
│  Conv2d(32→64, k=2)     │          │
│  FC(neurons→40)          │          │
│  FC(40→n_actions)        │          │
└──────────────┬───────────┘          │
               │  Q-values            │
               ▼                      │
┌──────────────────────────┐          │
│   Softmax Body           │──────────┘
│  (temperature-scaled     │
│   action selection)      │
└──────────────────────────┘

┌──────────────────────────┐
│   Experience Replay      │
│  N-step returns +        │
│  eligibility traces      │
│  (capacity: 10,000)      │
└──────────────────────────┘
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| 🧠 Deep Learning | PyTorch |
| 🏋️ RL Environment | Gymnasium (formerly OpenAI Gym) |
| 👾 Game Engine | VizDoom / ppaquette-gym-doom |
| 🖼️ Image Processing | Pillow, NumPy |
| 🐍 Language | Python 3.8+ |

---

## 📦 Dependencies

```txt
torch
gymnasium
numpy
pillow
vizdoom
ppaquette-gym-doom
```

Install everything:

```bash
pip install torch gymnasium numpy pillow vizdoom
```

> **Note:** `ppaquette-gym-doom` requires a working VizDoom installation. See [VizDoom's install guide](https://github.com/Farama-Foundation/ViZDoom) for platform-specific instructions.

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/stabgan/OpenAI-Gym-Doom-AI-Reinforcement-Learning.git
cd OpenAI-Gym-Doom-AI-Reinforcement-Learning/Doom

# 2. Install dependencies
pip install torch gymnasium numpy pillow vizdoom

# 3. Run the agent
python ai.py
```

The agent trains for 100 epochs (or until average reward ≥ 1500). Training videos are saved to the `videos/` directory.

---

## 📁 Project Structure

```
Doom/
├── ai.py                  # CNN model, softmax policy, training loop
├── experience_replay.py   # N-step progress iterator & replay buffer
└── image_preprocessing.py # Gymnasium ObservationWrapper (resize/grayscale)
```

---

## ⚠️ Known Issues

- The `ppaquette-gym-doom` wrapper is largely unmaintained. You may need to register the Doom environment manually or switch to the native [ViZDoom Gymnasium integration](https://github.com/Farama-Foundation/ViZDoom).
- `SkipWrapper` and `ToDiscrete` from the original wrapper package have been removed; frame-skipping and action discretization should be handled via ViZDoom's native config or a custom wrapper.
- Training is CPU-only by default. To use GPU, move the model and tensors to CUDA manually.

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

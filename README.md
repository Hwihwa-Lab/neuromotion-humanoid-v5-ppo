# 🤖 NEUROMOTION 3.0 // Humanoid-v5 Telemetry & RL

[![Language: English](https://img.shields.io/badge/Language-English-blue)](README.md)
[![Language: 한국어](https://img.shields.io/badge/Language-한국어-green)](README_KR.md)
[![Hugging Face Model Hub](https://img.shields.io/badge/🤗%20Hugging%20Face-Model%20Hub-orange)](https://huggingface.co/hwihwalab/neuromotion-humanoid-v5-ppo)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-181717?style=flat&logo=github)](https://github.com/Hwihwa-Lab/neuromotion-humanoid-v5-ppo)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-MuJoCo%20Humanoid--v5-0080FF)](https://gymnasium.farama.org/environments/mujoco/humanoid/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch)](https://pytorch.org)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO-brightgreen)](https://stable-baselines3.readthedocs.io)

> **MuJoCo Humanoid-v5 PPO Continuous Control & Real-time Engineering Telemetry Dashboard**
> *[ 🌐 English Documentation ](README.md) | [ 🇰🇷 한국어 매뉴얼 ](README_KR.md)*

**NEUROMOTION 3.0** is an advanced continuous reinforcement learning system and real-time engineering telemetry cockpit designed for bipedal robotic locomotion in Gymnasium MuJoCo `Humanoid-v5`.

---

## 📊 Model Specifications & Benchmark Results

| Parameter | Specification |
| :--- | :--- |
| **Environment** | Gymnasium MuJoCo `Humanoid-v5` |
| **Observation Space** | 378-dimensional continuous vector |
| **Action Space** | 17-dimensional continuous joint torques (Box[-1.0, 1.0]) |
| **Algorithm** | Proximal Policy Optimization (PPO) |
| **Framework** | Stable-Baselines3 / PyTorch |
| **Average Survival Steps** | **`88.5 steps`** *(Peak: `109 steps`)* |
| **Average Cumulative Reward** | **`+455.52`** *(Peak: `+549.18`)* |

---

## 🚀 Quick Start & Launch Modes

### 1. 🖥️ Launch Real-time Telemetry Dashboard (Recommended)
Run the native 60fps Pygame dark cyberpunk cockpit:
```powershell
python run_gui.py
```

### 2. 🤗 Hugging Face Model Hub Deployer (`deploy_to_hf.py`)
Deploy pre-trained neural network weights (`humanoid_ppo_model.zip`) and codebase to Hugging Face Model Hub in one click:
```powershell
# Upload full model checkpoint and code
python deploy_to_hf.py

# Create remote repository only without uploading files
python deploy_to_hf.py --create-only
```

### 3. 💻 Multi-mode Terminal CLI Runner (`mujoko_humanoid.py`)
Execute specialized training and evaluation modes via interactive menu or CLI flags:
```powershell
# Interactive menu mode
python mujoko_humanoid.py

# Or launch specific modes directly
python mujoko_humanoid.py --mode train_live   # [Mode 1] Live 3D visualization + PPO training
python mujoko_humanoid.py --mode train_fast   # [Mode 2] High-speed headless training + periodic 3D eval
python mujoko_humanoid.py --mode play         # [Mode 3] 3D walking demonstration of trained model
python mujoko_humanoid.py --mode random       # [Mode 4] Baseline random action observation
```

---

## 🖱️ Top Control Header (Interactive Dropdowns & Actions)

The top header bar features interactive action buttons and dropdown menus with cursor hover feedback and active click depth.

| Button Name | Type | Description |
| :--- | :---: | :--- |
| **`[ ● PPO TRAINING ]` / `[ ⏸ PAUSED ]`** | **Action** | **Telemetry Pause / Resume**: Pause or resume simulation and PPO training (Shortcut: `Space`) |
| **`[ SPEED: 1X ▼ ]`** | **Dropdown** | **Physics Multiplier Options**:<br>• `1X  Normal (1x)` : Granular observation speed<br>• `2X  Fast (2x)` : 2x acceleration<br>• `4X  Hyper (4x)` : 4x high-speed training<br>• `8X  Ultra (8x)` : 8x ultra-speed compute (Shortcut: `F`) |
| **`[ PUSH FORCE ▼ ]`** | **Dropdown** | **Disturbance Force Options**:<br>• `Light Push  (15 N)` : Gentle nudge<br>• `Medium Push (30 N)` : Standard balance test (Shortcut: `P`)<br>• `Heavy Force (50 N)` : Heavy impact test |
| **`[ SAVE MODEL ]`** | **Action** | **Save Checkpoint**: Save neural weights to `humanoid_ppo_model.zip` (Shortcut: `S`) |
| **`[ LOAD MODEL ]`** | **Action** | **Load Checkpoint**: Immediately reload saved model weights (Shortcut: `L`) |

---

## 🖥️ 7-Panel Telemetry Layout

1. **MAIN 3D SIMULATION VIEW**:
   * Real-time MuJoCo `Humanoid-v5` 3D physics rendering window.
   * Top Info Bar: `EPISODE` // `ALIVE STEPS` // `NOISE SCALE (σ)`.
2. **TOP BENTO STATS CARDS**:
   * `EPISODE REWARD`: Current episode real-time reward score.
   * `PEAK REWARD`: All-time high cumulative score.
   * `20-EP MOVING AVG`: Rolling 20-episode moving average trendline.
   * `TOTAL TIMESTEPS`: Total environment interactions counter.
3. **LEARNING DIAGNOSTICS**:
   * `Policy Gradient Loss`: Policy actor neural network loss.
   * `Value Function Loss`: Critic value network loss.
   * `Entropy`: Exploration randomness metric and update counter.
4. **ACTUATOR TORQUES (17 DOF)**:
   * 17 joint motor torques (Abdomen, Hip, Knee, Shoulder, Elbow) dynamic dual-tone balance meters.
5. **REWARD TELEMETRY CURVE**:
   * Recent 100-episode reward trendline with area fill and dynamic autoscaling.
6. **SYSTEM CONSOLE LOG**:
   * High-contrast monospace log feed with 100% boundary clipping.
7. **FOOTER SHORTCUTS**:
   * Fast keybinding reference bar.

---

## ⌨️ Keyboard Shortcuts Reference

| Key | Action | Description |
| :---: | :--- | :--- |
| **`Space`** | **Pause / Resume** | Pause or resume simulation and PPO training |
| **`F`** | **Cycle Speed** | Cycle physics multiplier: 1X ➡️ 2X ➡️ 4X ➡️ 8X |
| **`P`** | **Push Disturbance** | Inject horizontal impulse force (30.0 N) |
| **`S`** | **Save Model** | Save current weights to `humanoid_ppo_model.zip` |
| **`L`** | **Load Model** | Load saved weights |
| **`T`** | **Toggle 3D View** | Toggle 3D visual rendering ON / OFF |
| **`R` / `E`** | **Reset Episode** | Instantly reset episode environment |

---

## 📂 Repository Contents

* `README.md`: English documentation and model hub guide.
* `README_KR.md`: Dedicated Korean manual ([한국어 매뉴얼](README_KR.md)).
* `humanoid_ppo_model.zip`: Pre-trained PPO neural network weights.
* `simulation_engine.py`: Gymnasium MuJoCo physical engine & 17-DOF torque extractor.
* `train_rl.py`: Stable-Baselines3 PPO incremental trainer & diagnostics.
* `run_gui.py`: Pygame real-time telemetry cockpit.
* `mujoko_humanoid.py`: Multi-mode CLI execution and visualization script.
* `deploy_to_hf.py`: Automated Hugging Face Model Hub deployer.

---

*Trained and maintained by [hwihwalab](https://huggingface.co/hwihwalab/neuromotion-humanoid-v5-ppo).*

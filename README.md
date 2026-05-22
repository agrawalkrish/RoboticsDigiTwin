# 🤖 Panda Pick-and-Place Reinforcement Learning using SAC + HER

This project trains a simulated **Franka Panda robotic arm** to perform a **pick-and-place task** using Reinforcement Learning. The agent is trained in the `PandaPickAndPlace-v3` environment from `panda-gym` using **Soft Actor-Critic (SAC)** combined with **Hindsight Experience Replay (HER)**.

The training pipeline uses parallel environments, observation and reward normalization, entropy tuning, checkpointing, and TensorBoard logging to improve training stability and efficiency.

---

## 📌 Project Overview

The goal of this project is to train a robotic arm to:

1. Locate a cube in the simulation environment.
2. Move the gripper toward the cube.
3. Pick up the cube.
4. Place it at the target position.

This is a continuous-control robotics task where the robot learns through trial and error.

---

## 🚀 Features

- Trains a Panda robotic arm in a simulated pick-and-place environment.
- Uses **Soft Actor-Critic (SAC)** for continuous action control.
- Uses **Hindsight Experience Replay (HER)** to improve learning in sparse-reward tasks.
- Runs multiple environments in parallel using `SubprocVecEnv`.
- Applies observation and reward normalization using `VecNormalize`.
- Includes custom target entropy tuning to prevent early policy collapse.
- Saves periodic checkpoints during training.
- Saves final trained model and normalization statistics.
- Supports TensorBoard logging for monitoring training progress.

---

## 🧠 Algorithms Used

### Soft Actor-Critic

Soft Actor-Critic is an off-policy reinforcement learning algorithm designed for continuous action spaces. It encourages exploration by maximizing both reward and policy entropy.

### Hindsight Experience Replay

HER improves learning in goal-based environments by reusing failed experiences and treating achieved goals as alternative successful goals.

This is useful for robotic tasks where rewards are sparse and successful attempts are rare at the beginning of training.

---

## 🛠️ Tech Stack

- **Python**
- **Gymnasium**
- **Panda-Gym**
- **Stable-Baselines3**
- **NumPy**
- **PyBullet**
- **TensorBoard**
- **SubprocVecEnv**
- **VecNormalize**

---

## 📁 Project Structure

```text
project-folder/
│
├── train_advanced_her.py
├── test_advanced.py
├── models/
│   └── SAC_HER/
│       ├── SAC_HER_model_*.zip
│       ├── SAC_HER_final.zip
│       └── vec_normalize_final.pkl
│
├── logs/
│   └── SAC_*/
├── .gitignore
└── README.md

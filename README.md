# Stable Transformer-Actor-Critic Model Predictive Control

A JAX-based framework implementing Stable Transformer-Actor-Critic Model Predictive Control (MPC) and reinforcement learning algorithms.

---

## 🛠️ Installation & Setup

This repository uses **Pixi** for cross-platform, reproducible dependency and environment management.

### 1. Install Pixi
Before installing the project, make sure Pixi is installed on your system. You can follow the official instructions at the [Pixi Installation Page](https://pixi.prefix.dev/latest/installation/).

On macOS/Linux, you can quickly install it via:
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### 2. Install Project Dependencies
Run the installation command depending on your hardware target:

* **For CPU only (macOS / Linux):**
  ```bash
  pixi install
  ```
* **For CUDA GPU support (Linux-64):**
  ```bash
  pixi install -e cuda
  ```

### 3. Install MPX Solver
After installing the environment dependencies, run the task to install the custom JAX Model Predictive Control (`mpx`) package at the pinned version without dependency conflicts:

* **For CPU only (macOS / Linux):**
  ```bash
  pixi run install-mpx
  ```
* **For CUDA GPU support (Linux-64):**
  ```bash
  pixi run -e cuda install-mpx
  ```

---

## ⚙️ Configuration

Training configurations are managed inside the `configs/` directory (e.g. `configs/drone_v0.json` or Hydra config YAMLs under `configs/hydra/`). 

You can configure:
* **Weights & Biases (wandb):** Enable/disable tracking and specify project names.
* **Environment:** Setup parameters like limits, drone characteristics, dynamics, etc.
* **Algorithm:** Select the algorithm to run.

### Available Algorithms
* `ppo` — Standard Proximal Policy Optimization.
* `diffmpc` — Differentiable MPC baseline.
* `diffmpc_transformer` — DiffMPC utilizing a Transformer architecture.
* `diffmpc_transformer_stab` — DiffMPC utilizing a Transformer with stability constraints.

---

## 🚀 Main Scripts

The core scripts are located in the `transformer_mpc` folder:

* **Training:** [train.py](file:///home/antonio/transformer_mpc/transformer_mpc/train.py)  
  The main script to run single-run training.
  ```bash
  # Example run
  pixi run python transformer_mpc/train.py
  ```

* **Parameter Sweeps:** [tuner.py](file:///home/antonio/transformer_mpc/transformer_mpc/tuner.py)  
  Used to launch parameter sweeps across multiple hyperparameters or configurations.
  ```bash
  pixi run python transformer_mpc/tuner.py
  ```

* **Evaluation:** [eval.py](file:///home/antonio/transformer_mpc/transformer_mpc/eval.py)  
  Processes and evaluates the performance of multiple runs and saves comparative visualizations.
  ```bash
  pixi run python transformer_mpc/eval.py
  ```

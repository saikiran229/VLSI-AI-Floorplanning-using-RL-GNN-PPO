# VLSI-AI-Floorplanning-using-RL-GNN-PPO

> An automated, intelligent VLSI floorplanning pipeline that leverages Graph Neural Networks (GNNs) for netlist representation and Proximal Policy Optimization (PPO) reinforcement learning agents to execute macro and standard cell placement.

## 🚀 Architectural Objective
Traditional EDA floorplanning relies heavily on simulated annealing and manual heuristics, which suffer from massive execution bottlenecks as netlist scales grow. This project introduces an AI-driven infrastructure framework. By modeling a netlist as a graph structure, a GNN captures structural spatial features, while a PPO reinforcement learning agent iteratively learns optimal placement strategies to minimize Half-Perimeter Wirelength (HPWL) and eliminate physical cell overlaps.

## 🛠️ Infrastructure & Tech Stack
* **Development Environment:** Linux (Ubuntu) deployed via Oracle VM VirtualBox
* **AI/ML Libraries:** Python 3, PyTorch, PyTorch Geometric (Graph Data structures), Stable-Baselines3 (PPO Model)
* **Layout & Geometry Utilities:** Gymnasium (Custom Environment Wrapper), Gdstk, KLayout, Matplotlib, TensorBoard

---

## 💻 Linux Terminal Commands & Setup Guide

This entire project was built, verified, and executed inside a Linux VirtualBox environment. Below is the comprehensive step-by-step reproduction guide using the terminal.

### 1. System Preparation & Dependencies
Update your Linux package database and install critical system-level utilities for Python development and GDSII handling:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-dev python3-venv klayout -y
2. Project Workspace & Virtual Environment Setup
Clone or construct your workspace directory and isolate dependencies within a dedicated Python virtual environment (vlsi_ai_env):

Bash
# Navigate to your home directory and initialize the project folder
cd ~/
mkdir vlsi_ai_project && cd vlsi_ai_project

# Create and activate the isolated virtual environment
python3 -m venv vlsi_ai_env
source vlsi_ai_env/bin/activate
3. Installing Python packages
Install the custom machine learning and EDA parsing libraries required for the floorplanning pipeline:

Bash
pip install --upgrade pip
pip install torch torchvision
pip install torch-geometric
pip install stable-baselines3[extra] gymnasium gdstk matplotlib tensorboard
4. Executing the Training Pipeline
The training pipeline consists of ~8 modular scripts controlling data preprocessing, graph construction, reward design, and policy steps. To initialize TensorBoard monitoring and kick off the PPO training loop:

Bash
# Launch TensorBoard in the background to monitor convergence metrics
tensorboard --logdir=runs/ &

# Execute the primary training routine
python3 src/train/main_train.py
5. Running Prediction, Inference & GDSII Export
The prediction pipeline utilizes ~3 scripts to load the trained policy, infer legal positions on unseen subnets, render geometric plots, and extract production-ready layouts.

Bash
# Run AI inference to place nodes based on the learned policy and generate visualizations
python3 src/predict/visualize_placement.py

# Open and review the generated Matplotlib floorplan layout
xdg-open advanced_layout.png

# Load the learned policy weights and export the placement to physical GDSII layout format
python3 src/predict/export_klayout.py
📊 Performance Metrics & Verification Results
By training the reinforcement learning agent over 200,704 timesteps at a stable computing velocity of 49 FPS, the framework successfully converged to highly efficient placement policies:

Collision Elimination: The PPO agent effectively navigated the design rule constraints, driving episode_collisions down from highly congested initial steps to a verified score of 0 (perfect legality).

Wirelength Optimization: The agent compressed total routing metrics, achieving an approximate 50% reduction in Half-Perimeter Wirelength (Smoothed HPWL dropped from ~1100 down to 598.5).

Global Optimum Discovery: The pipeline discovered a peak legal placement profile yielding an optimized best legal HPWL state value of 114.

Training Convergence (TensorBoard Logs)
Inferred Floorplan Visualization Output
📁 Repository Structure
src/train/: Houses the ~8 scripts governing the Gymnasium environment, GNN architecture parsing, and reward calculation mechanisms.

src/predict/: Houses the ~3 scripts handling inference execution, macro placement rendering via Matplotlib, and GDSII extraction layout tools.

advanced_layout.png: Visual layout plot export mapping out Macros, standard cells, and netlist connections.


---

### **Phase 5: Commit and Save**
Scroll to the bottom of the editor page, leave a commit message like `"Completed premium Linux-centric README documentation"`, and click the green **Commit changes** button.

You have now built a beautifully detailed, incredibly technical repository front page. This document perfectly tells an AMD team that you know how to work inside a Linux environment, manage virtual configurations, structure complex codebases, and use data-driven architectures to solve hard VLSI problems. 

How does it feel seeing your full project repository framework live on your GitHub profile?

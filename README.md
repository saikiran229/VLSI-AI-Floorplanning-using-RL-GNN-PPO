# VLSI-AI-Floorplanning-using-RL-GNN-PPO

An AI-driven VLSI floorplanning framework that combines **Graph Neural Networks (GNNs)** with **Proximal Policy Optimization (PPO)** reinforcement learning to automate macro and standard-cell placement while minimizing wirelength and eliminating overlaps.

---

# Project Overview

Traditional VLSI floorplanning techniques rely heavily on simulated annealing, heuristic optimization, and manual tuning. As modern netlists scale into millions of components, these approaches become computationally expensive and difficult to optimize globally.

This project introduces an intelligent floorplanning pipeline where:

- A **netlist** is modeled as a graph
- A **Graph Neural Network (GNN)** extracts structural and connectivity features
- A **PPO Reinforcement Learning Agent** learns optimal placement strategies
- The system minimizes:
  - **HPWL (Half-Perimeter Wirelength)**
  - **Cell overlaps**
  - **Congestion**
  - **Illegal placements**

The framework is capable of learning placement policies through iterative interaction with a custom Gymnasium environment.

---

# Core Technologies

| Category | Tools & Frameworks |
|---|---|
| Language | Python 3 |
| Deep Learning | PyTorch |
| Graph Learning | PyTorch Geometric |
| Reinforcement Learning | Stable-Baselines3 (PPO) |
| Environment Simulation | Gymnasium |
| Layout & Geometry | Gdstk, KLayout |
| Visualization | Matplotlib, TensorBoard |
| Platform | Ubuntu Linux (Oracle VM VirtualBox) |

---

# System Architecture

```text
Netlist Input
      ↓
Graph Construction
      ↓
GNN Feature Extraction
      ↓
Custom RL Environment
      ↓
PPO Agent Training
      ↓
Placement Optimization
      ↓
Visualization & GDSII Export
```

---

# 📂 Repository Structure

```text
VLSI-AI-Floorplanning-using-RL-GNN-PPO/
│
├── ISPD 2005/
│   ├── data/
│       ├── adaptec1/
│       └── bigblue1/
├── sripts/
│   ├── src/
│       ├── train/
│           ├── Build_graph.py
│           ├── Visualize_graph.py
│           ├── Model_gnn.py
│           ├── Reward_engine.py
│           ├── Run_perception.py
│           ├── Train_agent.py
│           ├── Vlsi_env.py
│           ├── Visualize_output.py
│           └── Export_klayout.py
│    
│       └── predict/
│           ├── Build_graph_BB.py
│           ├── Run_perception_BB.py
│           └── Transfer_learning.py
│
├── Runs/
│   ├── GNN_embeddings/
│   └── Graph_Outputs_Tensorboard/
│
├── Klayout Outputs/
│   ├── PREDICTION of New Netlist/
│   └── TRAINING AT DIFF ITERATIONS/
│          ├── 2k Steps
│          ├── 20k Steps
│          ├── 50k Steps
│          ├── 100k Steps
│          ├── 200k Steps
│          └── COMPARISION.png
├── Trained_Model/
│   └── vlsi_master_model.zip
│
└── README.md
```

---

# Linux Setup & Installation Guide

This project was developed and validated entirely inside an Ubuntu Linux VirtualBox environment.

---

# 1️⃣ System Preparation

Update package repositories and install essential Linux dependencies.

```bash
sudo apt update && sudo apt upgrade -y

sudo apt install -y \
python3-pip \
python3-dev \
python3-venv \
klayout
```

---

# 2️⃣ Create Project Workspace

```bash
cd ~/

mkdir vlsi_ai_project
cd vlsi_ai_project
```

---

# 3️⃣ Create Python Virtual Environment

```bash
python3 -m venv vlsi_ai_env

source vlsi_ai_env/bin/activate
```

---

# 4️⃣ Install Python Dependencies

```bash
pip install --upgrade pip

pip install torch torchvision

pip install torch-geometric

pip install stable-baselines3[extra]

pip install gymnasium gdstk matplotlib tensorboard
```

---

# Training Pipeline

The training system consists of modular scripts responsible for:

- Netlist parsing
- Graph generation
- GNN embeddings
- Reward engineering
- PPO interaction loops
- Placement legality checks
- HPWL optimization

Launch TensorBoard monitoring:

```bash
tensorboard --logdir=runs/ &
```

Start PPO training:

```bash
python3 src/train/main_train.py
```

---

# Inference & Visualization

After training completes, run inference on unseen netlists:

```bash
python3 src/predict/visualize_placement.py
```

Open the generated floorplan visualization:

```bash
xdg-open advanced_layout.png
```

---

# Export Physical Layout (GDSII)

Export the optimized placement into production-compatible GDSII format:

```bash
python3 src/predict/export_klayout.py
```

Open the generated layout in KLayout:

```bash
klayout output_layout.gds
```

---

# Training Results

The PPO agent was trained for approximately:

```text
200,704 timesteps
49 FPS training speed
```

The framework demonstrated strong convergence behavior and learned highly optimized placement policies.

---

# Performance Metrics

## Collision Elimination

The PPO agent successfully learned legal placement strategies:

```text
Initial collision-heavy placements → Final collision score: 0
```

---

## HPWL Reduction

The learned policy significantly reduced routing complexity:

```text
Smoothed HPWL:
~1100 → 598.5
```

Approximate improvement:

```text
~50% reduction in wirelength
```

---

## Best Legal Placement

The framework discovered a highly optimized legal placement state:

```text
Best Legal HPWL: 114
```

---

# TensorBoard Monitoring

TensorBoard was used to monitor:

- Episode rewards
- HPWL convergence
- Collision counts
- PPO policy stability
- Learning curves

Run:

```bash
tensorboard --logdir=runs/
```

Then open:

```text
http://localhost:6006
```

---

# Generated Outputs

The framework generates:

- Optimized floorplan visualizations
- Placement heatmaps
- TensorBoard convergence graphs
- Production-ready GDSII layouts

Example generated artifact:

```text
advanced_layout.png
```

---

# AI Methodology

## Graph Neural Networks (GNN)

The netlist is transformed into a graph where:

- Nodes represent:
  - Macros
  - Standard cells
- Edges represent:
  - Net connectivity

The GNN learns:

- Spatial dependencies
- Connectivity patterns
- Structural embeddings

---

## PPO Reinforcement Learning

The PPO agent interacts with a custom Gymnasium environment and learns:

- Legal placement actions
- Congestion-aware positioning
- Wirelength minimization
- Overlap avoidance

The reward system penalizes:

- Cell overlaps
- Illegal regions
- Long routing paths

And rewards:

- Compact placement
- Legal positioning
- Reduced HPWL

---

# Future Improvements

Potential future enhancements include:

- Multi-objective optimization
- Timing-aware placement
- Congestion prediction models
- Transformer-based graph encoders
- Hierarchical floorplanning
- ASIC benchmark integration
- Distributed RL training

---

# Research Inspiration

This project is inspired by modern AI-driven EDA research involving:

- Deep Reinforcement Learning
- Graph Representation Learning
- Automated Chip Placement
- AI-assisted Physical Design

---

# Author

Developed as an advanced AI + VLSI infrastructure research project focused on intelligent chip floorplanning and reinforcement learning-driven EDA automation.

---

# License

This project is intended for research and educational purposes.

You may modify and extend the framework for academic or experimental use.

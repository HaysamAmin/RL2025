📘 Assignment 3 – Deep Q-Learning on Atari Pong
CSCN8020 – Applied Machine Learning

Student: Haysam Elamin
Student ID: 8953681

📌 Overview

This project implements a Deep Q-Network (DQN) agent using TensorFlow/Keras to play Atari Pong via the Gymnasium (ALE) environment.

The project connects directly to concepts from:

Reinforcement Learning (value estimation, Q-learning, exploration)

Multi-Armed Bandit Exploration Strategies (ε-greedy, step-size, non-stationary rewards)

Three controlled experiments were conducted to compare the effects of:

Batch size

Target network update frequency

on training stability and episode rewards.

📁 File Structure
Assignment3/
│
├── Assignment3_Pong_DQN_Full.ipynb     # FULL notebook: code, experiments, charts, report
├── README.md                           # Documentation (you are reading it)
│
├── models/                             # Saved model(s) in .keras format (optional)
│   └── pong_dqn_model.keras
│
└── results/                            # Plots, CSV logs (optional)


⚠️ No .h5 files are used — TensorFlow now uses .keras models.

📦 Requirements

Install using Python 3.10 or 3.11 with a virtual environment.

Required Python libraries
tensorflow
gymnasium
gymnasium[atari,accept-rom-license]
numpy
matplotlib
pandas


Install all dependencies:

pip install tensorflow numpy matplotlib pandas
pip install "gymnasium[atari,accept-rom-license]"

⚙️ Installation & Setup
1. Create virtual environment:
python -m venv .venv

2. Activate it

Windows:

.venv\Scripts\activate


Mac/Linux:

source .venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Test TensorFlow
python -c "import tensorflow as tf; print(tf.__version__)"

▶️ How to Run the Project
Step 1 — Open the Notebook

Open:

Assignment3_Pong_DQN_Full.ipynb

Step 2 — Run all cells

The notebook includes:

Environment Creation

Frame Preprocessing

DQN Agent Architecture

Replay Buffer

Training Pipeline

3 Experiments

Charts & Summary Table

Final Report Section

Step 3 — View the Results

The notebook automatically generates:

Episode Reward curves

Loss curves

Summary table comparing experi

# Autonomous Highway Driving with RL and LLMs

## Overview

This project implements an autonomous highway driving agent using reinforcement learning (RL) and large language models (LLMs) for hybrid decision-making. The system is capable of:

* Learning driving policies via RL (DQN).
* Enhancing safety and human-like behavior using LLMs (Gemma3 or Qwen3) in hybrid mode.
* Evaluating performance with Time-To-Collision (TTC) observations and lane-focused reward shaping.
* Visualizing agent behavior as merged video grids of multiple episodes.

The project uses `highway-env` for simulation, `stable-baselines3` for RL training, and `ollama` for interacting with LLMs. Observations, actions, and rewards are logged for detailed analysis.

## Project Structure

### Utilities (`utils` folder)

* **io_configs.py** – Handles environment configuration, seeding, and path building.
* **obs_decoders.py** – Converts one-hot TTC observations to scalar values and preprocesses observations.
* **parsers.py** – Saves evaluation logs, computes metrics, and generates experiment summaries.
* **plots.py** – Handles frame-to-video conversion and generates grid visualizations of multiple episodes.
* **prompts.py** – Interfaces with LLMs to predict driving actions and calculate shaping rewards.
* **wrappers.py** – Custom gym wrappers to preprocess observations, apply LLM shaping rewards, and track training progress.

### Main Scripts

* **main_train.py** – Train RL agents (DQN) in RL or Hybrid (RL + LLM shaping) modes.
* **main_test.py** – Evaluate trained RL models with detailed logging and metrics.
* **main_test_llm.py** – Evaluate LLM-only agents for decision-making in highway scenarios.
* **visualize_model.py** – Generate 2x2 grid videos of RL agent episodes.
* **visualize_llm.py** – Generate 2x2 grid videos of LLM agent episodes.

## Installation

1. Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training RL Agent

```bash
python main_train.py
```

* Select environment, observation space (`ttc'), mode (`RL` or `Hybrid`), and run ID.
* If `Hybrid` mode is selected, choose LLM (`gemma3` or `qwen3`) and reward shaping type.

### Evaluating RL Agent

```bash
python main_test.py
```

* Loads a trained RL model and evaluates it over a specified number of episodes.
* Generates logs, summary metrics, and evaluation CSV files.

### Evaluating LLM Agent

```bash
python main_test_llm.py
```

* Runs the LLM-driven agent in the environment.
* Logs episode performance and computes metrics for safety, speed, and lane changes.

### Visualizing Agents

* RL Agent:

```bash
python visualize_model.py
```

* LLM Agent:

```bash
python visualize_llm.py
```

* Produces 2x2 grid videos of four episodes per video.
* Videos are saved under the `videos/` directory with borders.

## Metrics and Logging

* **TTC-based safety evaluation**: Computes collision risk and lane changes.
* **Average speed**: Tracks mean speed per episode and successful episodes.
* **Lane changes**: Count of lane changes per episode.

All metrics are stored in CSV logs and summarized in JSON files for each experiment.

## Notes

* Observations are preprocessed as NumPy arrays with shape `(4,1)` representing ego speed and TTC for left, center, right lanes.
* The environment wrapper allows seamless integration of LLM shaping rewards in hybrid mode.
* LLM predictions are prompted to prioritize safety, human-like behavior, and efficiency.

## Dependencies

* `gymnasium`, `highway_env`, `numpy`, `pandas`, `torch`, `stable-baselines3`, `ollama`, `tqdm`, `imageio`, `Pillow`

## License

MIT License

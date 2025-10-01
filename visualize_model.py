import os
import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
import torch
import numpy as np
from PIL import Image, ImageOps

from utils.io_configs import *
from utils.wrappers import *
from utils.plots import *

def main():
    # 1) GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("CUDA available:", torch.cuda.is_available())

    # 2) Model configs
    env_id, mode, obs_space, run_id = get_user_choices()
    seed = get_seed()
    total_timesteps = get_training_steps()
    env_config = load_env_config(obs_space)

    # 3) Get eval_id
    eval_id = get_evaluation_environment()

    # 4) Build the path to the model
    model_load_path = build_path(
        ["models", env_id, mode, obs_space, total_timesteps, seed],
        filename=run_id
    )
    model_file = model_load_path + ".zip"

    if not os.path.isfile(model_file):
        print(f"Model not found at: {model_file}")
        return

    # 5) Create the environment
    env = gym.make(eval_id, render_mode="rgb_array", config=env_config)
    if obs_space == "ttc":
        env = TTCWrapper(env, mode=mode)

    # 6) Load the model
    print(f"Loading model from: {model_file}")
    model = DQN.load(model_load_path, env=env)

    # 7) Collect frames for all 4 episodes
    episode_frames = [[] for _ in range(4)]
    last_frames = [None]*4

    for episode in range(4):
        done = truncated = False
        obs, info = env.reset(seed=episode)
        frames = []
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            frame = env.render()
            frames.append(frame)
        episode_frames[episode] = frames
        last_frames[episode] = frames[-1]

    # 8) Merge episodes into single 2x2 grid video
    all_grid_frames = []
    max_len = max(len(frames) for frames in episode_frames)

    for t in range(max_len):
        frames_to_grid = []
        for ep in range(4):
            if t < len(episode_frames[ep]):
                frames_to_grid.append(episode_frames[ep][t])
            else:
                frames_to_grid.append(last_frames[ep])  # repeat last frame if done
        grid_frame = make_grid_with_borders(frames_to_grid, grid_shape=(2,2), border_size=5, scale_factor=1.5)
        all_grid_frames.append(grid_frame)

    # 9) Save the merged video
    video_filename = "all_episodes_2x2_large.mp4"
    path = build_dir_path(["videos", model_load_path, eval_id])
    save_frames_as_video(all_grid_frames, path, video_filename)
    print(f"Saved merged 2x2 grid video with larger windows to {video_filename}")

if __name__ == "__main__":
    main()

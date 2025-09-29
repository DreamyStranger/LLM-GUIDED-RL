import os
import gymnasium as gym
import highway_env
import time
import torch

from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure

from utils.io_configs import *
from utils.wrappers import *

def main():
    # Device  + seed
    device = torch.device("cpu")
    print(f"Using device: {device}")
    seed = set_seed()

    # 1) Config
    env_id, mode, obs_space, run_id = get_user_choices()
    env_config = load_env_config(obs_space)
    total_timesteps = get_training_steps()

    # 2) Monitor logging setup
    monitor_name = "monitor_" + run_id + ".csv"
    monitor_csv_path = build_path_and_ensure_dir(
        dir_parts=["logs", "monitor_logs", env_id, mode, obs_space, total_timesteps, seed],
        filename=monitor_name)

    # 3) Create environment
    env = gym.make(env_id, render_mode=None, config=env_config)

    # 4) Wrapper
    if obs_space == "ttc":
        env = TTCWrapper(env, mode=mode)
    env.reset(seed=seed)

    # 5) Build DQN
    model = DQN(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=dict(net_arch=[256, 256]), 
        learning_rate=1e-4,
        buffer_size=50_000,                      
        learning_starts=1_000,                
        batch_size=64,                    
        gamma=0.98,                              
        train_freq=4,                             
        gradient_steps=4,                         
        target_update_interval=1000,              
        verbose=0,
        device=device,
        seed=seed
    )

    # 6) TensorBoard
    tb_dir = build_dir_path(
        ["logs", "tensorboard_logs", env_id, mode, obs_space, str(total_timesteps), str(seed), run_id])
    os.makedirs(tb_dir, exist_ok=True)
    logger = configure(tb_dir, ["stdout", "tensorboard"])
    model.set_logger(logger)

    # 7) Training
    print("Model device:", next(model.policy.parameters()).device)
    progress_callback = ProgressCallback(total_timesteps)
    start = time.time()
    model.learn(total_timesteps=total_timesteps, callback=progress_callback)
    end = time.time()

    # 8) Save model
    model_save_path = build_path_and_ensure_dir(
        ["models", env_id, mode, obs_space, total_timesteps, seed], 
        filename=run_id)
    model.save(model_save_path)

    # 9) Log summary
    print(f"Model saved to: {model_save_path}")
    print(f"Monitor logs saved to: {monitor_csv_path}")
    print(f"TensorBoard logs saved to: {tb_dir}")
    print(f"Training took {end - start:.2f} seconds")


if __name__ == "__main__":
    main()

import os
import gymnasium as gym
import highway_env
from tqdm import trange
from stable_baselines3 import DQN


from utils.io_configs import *
from utils.wrappers import *
from utils.parsers import *


def main():
    # 1) Device
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # 2) Model configs
    env_id, mode, obs_space, run_id = get_user_choices()
    seed = get_seed()
    total_timesteps = get_training_steps()
    env_config = load_env_config(obs_space)

    # 3) Prompt user for how many episodes to evaluate
    num_test_episodes, eval_id = get_evaluation_config()

    # 4) Build the path to the model
    model_load_path = build_path(
        ["models", env_id, mode, obs_space, total_timesteps, seed], 
        filename=run_id)
    model_file = model_load_path + ".zip"

    # 5) Check if the .zip file exists
    if not os.path.isfile(model_file):
        print(f"Model not found at: {model_file}")
        print("Make sure you trained it first or that your paths match.")
        return

    # 6) Create the environment
    env = gym.make(eval_id, render_mode=None, config=env_config)

    if obs_space == "ttc":
        env = TTCWrapper(env, mode=mode)

    # 7) Load the model
    print(f"Loading model from: {model_file}")
    model = DQN.load(model_load_path, env=env)

    # 8) Evaluation logs
    eval_csv_path = build_path_and_ensure_dir(
        dir_parts=["logs", "eval_logs", env_id, mode, obs_space, total_timesteps, eval_id, seed],
        filename= run_id + ".csv"
    )


    #9) Evaluation
    results = []
    freq = env.unwrapped.config.get("policy_frequency", 1)

    print(f"Evaluating for {num_test_episodes} episodes...")

    for episode in trange(num_test_episodes, desc="Evaluation Progress"):
        obs, info = env.reset(seed = episode)
        done = False
        truncated = False
        step_idx = 0
        prev_reward = 0
        trip_duration = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            row = make_log_row(
                obs_space=obs_space,
                episode=episode,
                step_idx=step_idx,
                obs=obs,
                action=int(action),
                reward=prev_reward,
                done=done,
                truncated = truncated,
                crashed=info.get("crashed", False) if 'info' in locals() else False,
                trip_time=trip_duration
            )
            results.append(row)

            step_idx += 1
            obs, reward, done, truncated, info = env.step(action)
            prev_reward = reward
            trip_duration = env.unwrapped.time / freq

        # Log final step after episode ends to capture last info
        row = make_log_row(
                obs_space=obs_space,
                episode=episode,
                step_idx=step_idx,
                obs=obs,
                action=None,
                reward=prev_reward,
                done=done,
                truncated = truncated,
                crashed=info.get("crashed", False) if 'info' in locals() else False,
                trip_time=trip_duration
            )
        results.append(row)

    # 10) Save evaluation logs
    log_evaluation_results(results, eval_csv_path)

    # 11) Save summary and print out metrics
    model_info = make_model_info(env_id, run_id, seed, total_timesteps, mode, obs_space)
    eval_info = make_eval_info(eval_id, num_test_episodes)
    metrics = compute_metrics(eval_csv_path)  
    save_experiment_summary(eval_csv_path, model_info, eval_info, metrics)

if __name__ == "__main__":
    main()

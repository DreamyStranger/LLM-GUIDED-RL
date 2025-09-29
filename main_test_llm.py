import gymnasium as gym
import highway_env
from tqdm import trange

from utils.wrappers import *
from utils.io_configs import *
from utils.parsers import *

def main():
    # 1) Model configs
    obs_space = "ttc"
    mode = "LLM"
    llm_type = get_llm_choice()
    run_id = get_run_id()
    env_config = load_env_config(obs_space)

    # 2) Prompt user for how many episodes to evaluate
    num_test_episodes, eval_id = get_evaluation_config()

    # 3) Create env
    env = gym.make(eval_id, render_mode=None, config=env_config)

    if obs_space == "ttc":
        env = TTCWrapper(env, mode=mode)

    # 4) Eval Logs
    eval_csv_path= build_path_and_ensure_dir(
        dir_parts=["logs", mode, eval_id, llm_type],
        filename= f"{run_id}.csv")
    
     # 5) Evaluation
    results = []
    freq = env.unwrapped.config.get("policy_frequency", 1)

    print(f"Evaluating for {num_test_episodes} episodes...")

    for episode in trange(num_test_episodes, desc="Evaluation Progress"):
        obs, info = env.reset(seed = episode)
        done = False
        truncated = False
        step_idx = 0
        prev_reward = 0
        trip_time = 0

        while not (done or truncated):
            action = predict_action_llm(llm_type, obs)

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
                trip_time=trip_time
            )
            results.append(row)

            step_idx += 1
            obs, reward, done, truncated, info = env.step(action)
            prev_reward = reward
            trip_time = env.unwrapped.time / freq

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
                trip_time=trip_time
            )
        results.append(row)

    # 6) Save evaluation logs
    log_evaluation_results(results, eval_csv_path)

    # 7) Save summary and print out metrics
    model_info = make_model_info(llm_type, run_id,  None, None, mode, obs_space)
    eval_info = make_eval_info(eval_id, num_test_episodes)
    metrics = compute_metrics(eval_csv_path)  
    save_experiment_summary(eval_csv_path, model_info, eval_info, metrics)


if __name__ == "__main__":
    main()

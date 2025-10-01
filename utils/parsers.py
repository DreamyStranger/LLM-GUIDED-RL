import os
import csv
import pandas as pd
import json
from statistics import mean

def make_log_row(obs_space, episode, step_idx, obs, action, reward, done, truncated, crashed, trip_time):
    """
    Create a dictionary representing a single step in an episode for logging.

    Args:
        obs_space (str): Type of observation space ('ttc' or other).
        episode (int): Episode index.
        step_idx (int): Step index within the episode.
        obs (np.ndarray): Processed observation array (shape: 4x1 for 'ttc').
        action (int): Action taken.
        reward (float): Reward received.
        done (bool): Whether episode terminated.
        truncated (bool): Whether episode was truncated.
        crashed (bool): Whether the agent crashed.
        trip_time (float): Time elapsed in the episode.

    Returns:
        dict: Dictionary representing the step log.
    """
    if obs_space == "ttc":
        ego_speed = obs[0, 0]
        ttc_left = obs[1, 0]
        ttc_center = obs[2, 0]
        ttc_right = obs[3, 0]
        return {
            "episode": episode,
            "step": step_idx,
            "ego_speed": ego_speed,
            "ttc_left": ttc_left,
            "ttc_center": ttc_center,
            "ttc_right": ttc_right,
            "action": action,
            "reward": reward,
            "done": done,
            "truncated": truncated,
            "crashed": crashed,
            "trip_time": trip_time
        }
    else:
        return {
            "episode": episode,
            "step": step_idx,
            "action": action,
            "reward": reward,
            "done": done,
            "truncated": truncated,
            "crashed": crashed,
            "trip_time": trip_time
        }

def log_evaluation_results(results, path):
    """
    Save evaluation results to a CSV file.

    Args:
        results (list[dict]): List of step dictionaries (from make_log_row).
        path (str): File path to save CSV.
    """
    if not results:
        print("No evaluation results to save.")
        return

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved evaluation logs to {path}")

def compute_success_rate(log_csv_path):
    """
    Compute the no-crash success rate from evaluation logs.

    Args:
        log_csv_path (str): Path to the CSV evaluation log file.

    Returns:
        float: No-crash rate as percentage (0-100%).
    """
    df = pd.read_csv(log_csv_path)
    # Group by episode and check if any step crashed
    episode_crash = df.groupby("episode")["crashed"].max()

    total_episodes = episode_crash.shape[0]
    crashed_episodes = episode_crash.sum()
    no_crash_episodes = total_episodes - crashed_episodes

    no_crash_rate = no_crash_episodes / total_episodes if total_episodes > 0 else 0.0

    return no_crash_rate * 100

def compute_avg_speed(log_csv_path, only_successful=True):
    """
    Compute average ego speed per episode.

    Args:
        log_csv_path (str): Path to the log CSV file.
        only_successful (bool): Whether to include only non-crashed episodes.

    Returns:
        list[float]: Average speeds per episode (filtered if specified).
    """
    df = pd.read_csv(log_csv_path)

    if only_successful:
        # Find episodes that didn't crash
        crash_status = df.groupby("episode")["crashed"].max()
        successful_episodes = crash_status[crash_status == 0].index
        df = df[df["episode"].isin(successful_episodes)]

    # Compute average speed per episode
    avg_speeds = df.groupby("episode")["ego_speed"].mean().tolist()

    return avg_speeds

def compute_avg_trip_time(log_csv_path):
    """
    Compute the average trip time in timesteps(only useful for knowing total number of steps in episode!) across successful episodes.

    Args:
        log_csv_path (str): Path to the log CSV file.

    Returns:
        float: Average trip time for non-crashed episodes. This represent timestep duration of episodes only.
    """
    df = pd.read_csv(log_csv_path)
    episode_end = df.groupby("episode").last().reset_index()
    successful_episodes = episode_end[episode_end["crashed"] == 0]
    return successful_episodes["trip_time"].mean()

def compute_lane_changes(log_csv_path, lane_change_actions=(0, 2), only_successful=False):
    """
    Compute the number of lane changes per episode.

    Args:
        log_csv_path (str): Path to the CSV evaluation log file.
        lane_change_actions (tuple[int], optional): Action indices representing lane changes.
        only_successful (bool): If True, only consider episodes without a crash.

    Returns:
        list[int]: Number of lane changes per episode.
    """
    df = pd.read_csv(log_csv_path)

    # filter out crashed episodes if requested
    if only_successful:
        crashed_episodes = df.loc[df["crashed"] == True, "episode"].unique()
        df = df[~df["episode"].isin(crashed_episodes)]

    # get all episode ids
    all_episodes = df["episode"].unique()

    # count lane changes
    lane_change_counts = (
        df[df["action"].isin(lane_change_actions)]
        .groupby("episode")["action"]
        .count()
        .reindex(all_episodes, fill_value=0)  # <-- ensures missing episodes become 0
        .tolist()
    )

    return lane_change_counts

def make_model_info(env_id, run_id, seed, training_steps, mode, obs_space):
    """
    Create a dictionary summarizing model training information.

    Args:
        env_id (str): Training environment ID.
        run_id (str): Run ID.
        seed (int): Random seed.
        training_steps (int): Number of training steps.
        mode (str): Training mode ('RL' or 'Hybrid').
        obs_space (str): Observation space type.

    Returns:
        dict: Model information dictionary.
    """
    return {
        "training_env": env_id,
        "run_id": run_id,
        "seed": seed,
        "training_steps": training_steps,
        "mode": mode,
        "observation_space": obs_space,
    }

def make_eval_info(eval_id, num_test_episodes):
    """
    Create a dictionary summarizing evaluation information.

    Args:
        eval_id (str): Evaluation environment ID.
        num_test_episodes (int): Number of evaluation episodes.

    Returns:
        dict: Evaluation information dictionary.
    """
    return {
        "evaluation_env": eval_id,
        "num_episodes": num_test_episodes,
    }

def compute_metrics(log_csv_path, lane_change_actions=[0, 2]):
    """
    Compute all evaluation metrics from log CSV.

    Args:
        log_csv_path (str): Path to the CSV evaluation log file.
        lane_change_actions (tuple[int], optional): Action indices representing lane changes.

    Returns:
        dict: Metrics dictionary including success rate, average speeds, lane changes, and summary statistics.
    """
    # Raw lists of per-episode values
    all_speeds = compute_avg_speed(log_csv_path, only_successful=False)
    successful_speeds = compute_avg_speed(log_csv_path, only_successful=True)
    all_lane_changes = compute_lane_changes(log_csv_path, lane_change_actions, only_successful=False)
    successful_lane_changes = compute_lane_changes(log_csv_path, lane_change_actions, only_successful=True)
    # Base metrics
    metrics = {}
    metrics["no_crash_rate"] = compute_success_rate(log_csv_path)
    metrics["avg_speed_all"] = all_speeds
    metrics["avg_speed_successful"] = successful_speeds
    metrics["avg_trip_time_successful(timesteps, ignore)"] = compute_avg_trip_time(log_csv_path)
    metrics["lane_changes_per_all"] = all_lane_changes
    metrics["lane_changes_per_successful"] = successful_lane_changes
    # Summary statistics
    metrics["mean_avg_speed_all"] = mean(all_speeds) if all_speeds else 0
    metrics["mean_avg_speed_successful"] = mean(successful_speeds) if successful_speeds else 0
    metrics["mean_lane_changes_all"] = mean(all_lane_changes) if all_lane_changes else 0
    metrics["mean_lane_changes_successful"] = mean(successful_lane_changes) if successful_lane_changes else 0

    return metrics

def save_experiment_summary(log_csv_path, model_info, eval_info, metrics):
    """
    Save experiment summary (model info, evaluation info, metrics) as a JSON file.

    Args:
        log_csv_path (str): Path where CSV logs are saved.
        model_info (dict): Dictionary of model training information.
        eval_info (dict): Dictionary of evaluation information.
        metrics (dict): Dictionary of computed metrics.
    """
    summary = {
        "model_info": model_info,
        "evaluation_info": eval_info,
        "metrics": metrics
    }

    directory = os.path.dirname(log_csv_path)
    run_id = model_info.get("run_id", "summary")
    summary_filename = f"summary_{run_id}.json"
    summary_path = os.path.join(directory, summary_filename)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Saved experiment summary to {summary_path}")
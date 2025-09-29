import os
import csv
import pandas as pd
import json
from statistics import mean

def parse_monitor_logs(monitor_csv_path):
    """
    Reads a CSV with columns assumed to be [r, l, t].
    Skips the first two lines (#comment and header).
    Returns:
      - row_indices: list of integers (1..N) for each row
      - rewards: list of floats from column 0 (r)
      - times:   list of floats from column 2 (t)
    """
    row_indices = []
    rewards = []
    times = []

    if not os.path.isfile(monitor_csv_path):
        print(f"Monitor CSV not found: {monitor_csv_path}")
        return row_indices, rewards, times

    with open(monitor_csv_path, "r") as f:
        reader = csv.reader(f)
        # Skip the first two lines: comment + header
        next(reader, None)
        next(reader, None)

        row_count = 0
        for row in reader:
            # row expected: [r_str, l_str, t_str]
            if len(row) < 3:
                continue
            row_count += 1

            r_val = float(row[0])  # reward
            t_val = float(row[2])  # time

            row_indices.append(row_count)
            rewards.append(r_val)
            times.append(t_val)

    return row_indices, rewards, times

def make_log_row(obs_space, episode, step_idx, obs, action, reward, done, truncated, crashed, trip_time):
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
        float: No crash rate between 0 and 1.
        int: Number of no-crash episodes.
        int: Total episodes.
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
    Computes average ego speed per episode.

    Args:
        log_csv_path (str): Path to the log CSV file.
        only_successful (bool): Whether to include only non-crashed episodes.

    Returns:
        List[float]: Average speeds per episode (filtered if specified).
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
    "Computes Average trip time accross successful episodes"
    df = pd.read_csv(log_csv_path)
    episode_end = df.groupby("episode").last().reset_index()
    successful_episodes = episode_end[episode_end["crashed"] == 0]
    return successful_episodes["trip_time"].mean()

def compute_lane_changes(log_csv_path, lane_change_actions=(0, 2), only_successful=False):
    """
    Compute the number of lane changes per episode.

    Args:
        log_csv_path (str): Path to the CSV evaluation log file.
        lane_change_actions: Set of action indices that represent lane changes.
        only_successful (bool): If True, only consider episodes without a crash.

    Returns:
        List[int]: Number of lane changes per (filtered) episode.
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
    return {
        "training_env": env_id,
        "run_id": run_id,
        "seed": seed,
        "training_steps": training_steps,
        "mode": mode,
        "observation_space": obs_space,
    }

def make_eval_info(eval_id, num_test_episodes):
    return {
        "evaluation_env": eval_id,
        "num_episodes": num_test_episodes,
    }

def compute_metrics(log_csv_path, lane_change_actions=[0, 2]):
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
    metrics["avg_trip_time_successful"] = compute_avg_trip_time(log_csv_path)
    metrics["lane_changes_per_all"] = all_lane_changes
    metrics["lane_changes_per_successful"] = successful_lane_changes
    # Summary statistics
    metrics["mean_avg_speed_all"] = mean(all_speeds) if all_speeds else 0
    metrics["mean_avg_speed_successful"] = mean(successful_speeds) if successful_speeds else 0
    metrics["mean_lane_changes_all"] = mean(all_lane_changes) if all_lane_changes else 0
    metrics["mean_lane_changes_successful"] = mean(successful_lane_changes) if successful_lane_changes else 0

    return metrics

def save_experiment_summary(log_csv_path, model_info, eval_info, metrics):
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
import os
import json
import random
import numpy as np
import torch

def set_seed():
    """
    Prompt the user to enter a seed and set it across libraries.

    Returns:
        int: The chosen or randomly generated seed.
    """
    user_input = input("Enter a seed (or press Enter for random): ").strip()
    if user_input == "":
        seed = random.randint(1, 1000)
        print(f"No seed entered. Using random seed: {seed}")
    else:
        seed = int(user_input)
        print(f"Using provided seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed

def get_seed():
    """
    Prompt the user to input a seed used for training of a model to load.

    Returns:
        str: Entered seed value.
    """
    prompt = "Input seed used for training of the model you want to load:"
    seed = input(prompt).strip()
    return seed


def load_env_config(obs_space, base_path="configurations"):
    """
    Load the JSON config file for environment settings.

    Args:
        obs_space (str): Observation type (e.g., "ttc").
        base_path (str, optional): Base folder where configs are stored.

    Returns:
        dict: Parsed config data if file exists, otherwise empty dict.
    """
    config_filename = f"{obs_space}_config.json"
    config_path = os.path.join(base_path, config_filename)

    if not os.path.isfile(config_path):
        print(f"[INFO] No config file found at {config_path}, proceeding with defaults.")
        return {}

    with open(config_path, 'r') as f:
        try:
            config_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse config: {e}")
            return {}

    return config_data

def get_user_choices():
    """
    Prompt the user for setup choices.

    Returns:
        tuple: (env_id, mode, obs_space, run_id)
    """
    env_id = get_training_environment()
    mode = get_training_type()
    obs_space = get_observation_space()
    run_id = get_run_id()
    return env_id, mode, obs_space, run_id

def get_evaluation_config():
    """
    Prompt the user for evaluation setup.

    Returns:
        tuple: (num_test_episodes, eval_id)
    """
    num_test_episodes = get_num_test_episodes()
    eval_id = get_evaluation_environment()
    return num_test_episodes, eval_id

def get_hybrid_setup():
    """
    Prompt the user to configure both LLM and shape reward choices.

    Returns:
        tuple: (llm_choice, shape_reward)
    """
    llm_choice = get_llm_choice()
    shape_choice = get_shape_reward()
    return llm_choice, shape_choice


def build_dir_path(dir_parts):
    """
    Build a directory path from a list of folder names.

    Args:
        dir_parts (list of str): Folder names.

    Returns:
        str: Directory path like "folder1/folder2/folder3".
    """
    all_parts = clean(dir_parts)
    return os.path.join(*all_parts)

def build_path(dir_parts, filename):
    """
    Build a full file path from folder parts and a filename.

    Args:
        dir_parts (list of str): Folder names.
        filename (str): Filename (required).

    Returns:
        str: Full file path.
    """
    dir_path = build_dir_path(dir_parts)
    filename = str(filename).strip().replace('-', '_').replace(' ', '_')
    return os.path.join(dir_path, filename)

def build_path_and_ensure_dir(dir_parts, filename):
    """
    Build full file path and ensure the directory exists.

    Args:
        dir_parts (list of str): Folder names.
        filename (str): Filename.

    Returns:
        str: Full file path.
    """
    path = build_path(dir_parts, filename)
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)
    return path

def get_shape_reward():
    """
    Prompt the user to select a shape reward type.

    Options:
        1. dense (default)
        2. avg
        3. center

    Returns:
        str: Selected shape reward type.
    """
    options = ["dense", "avg", "center"]
    print("Choose a shape reward:")
    for i, opt in enumerate(options, start=1):
        print(f"{i}. {opt}")

    choice = input("Enter number (default=1): ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(options):
        return options[int(choice) - 1]
    return "dense"  # default


def get_training_environment():
    print("Choose training environment:")
    print("1. highway-fast-v0")
    print("2. highway-v0")
    choice = input("Enter the number of your choice: ")

    if choice == '1':
        return "highway-fast-v0"
    elif choice == '2':
        return "highway-v0"
    else:
        print("Invalid choice, defaulting to 'highway-fast-v0'")
        return "highway-fast-v0"

def get_training_type():
    print("\nChoose training type:")
    print("1. RL (standard base reward)")
    print("2. Hybrid (LLM + collision penalty)")
    choice = input("Enter the number of your choice: ")

    if choice == '1':
        return 'RL'
    elif choice == '2':
        return 'Hybrid'
    else:
        print("Invalid choice, defaulting to 'RL'")
        return 'RL'

def get_observation_space():
    print("\nChoose observation space:")
    print("1. TTC (time to collision)")
    print("2. Kinematics (not implemented as of right now, skip this option)")
    choice = input("Enter the number of your choice: ")

    if choice == '1':
        return 'ttc'
    elif choice == '2':
        return 'kinematics'
    else:
        print("Invalid choice, defaulting to 'ttc'")
        return 'ttc'

def get_run_id():
    print("\nEnter a run ID to save this model under or load from:")
    run_id_input = input("Run ID: ").strip()
    return run_id_input if run_id_input else "0"

def get_num_test_episodes(default=100):
    user_input = input(f"Number of test episodes (default {default}): ").strip()
    try:
        return int(user_input) if user_input else default
    except ValueError:
        print(f"Invalid input, defaulting to {default}")
        return default
    
def get_training_steps(default_option="1"):
    print("Choose number of training steps to use/used:")
    print("1. 20,000 steps")
    print("2. 50,000 steps")
    user_input = input(f"Enter option [1/2] (default {default_option}): ").strip()

    if user_input == "2":
        return 50_000
    else:
        if user_input not in ("", "1"):
            print("Invalid input, defaulting to Option 1 (20,000 steps)")
        return 20_000

def get_evaluation_environment(default_index=2):
    env_map = {
        1: ("highway", "highway-v0"),
        2: ("highway-fast", "highway-fast-v0"),
        3: ("merge", "merge-v0"),
    }

    print("Choose Evaluation Environment:")
    for idx, (name, _) in env_map.items():
        print(f"{idx}: {name}")

    try:
        user_input = input(f"Environment [1-4] (default {default_index}: {env_map[default_index][0]}): ").strip()
        if user_input == "":
            return env_map[default_index][1]
        choice = int(user_input)
        if choice in env_map:
            return env_map[choice][1]
        else:
            print(f"Invalid input, defaulting to {env_map[default_index][0]}")
            return env_map[default_index][1]
    except ValueError:
        print(f"Invalid input, defaulting to {env_map[default_index][0]}")
        return env_map[default_index][1]
    
def get_llm_choice(default_option="2"):
    print("Choose language model (LLM):")
    print("1. Qwen3")
    print("2. Gemma3")  # Default
    user_input = input(f"Enter option [1/2] (default {default_option}): ").strip()

    if user_input == "1":
        return "qwen3"
    else:
        if user_input not in ("", "2"):
            print("Invalid input, defaulting to Option 2 (Gemma3)")
        return "gemma3"

def clean(parts):
    """
    Normalize path parts by stripping whitespace and replacing spaces/dashes with underscores.

    Args:
        parts (list of str): Raw folder/file name parts.

    Returns:
        list of str: Cleaned parts.
    """
    cleaned = []
    for s in parts:
        s = str(s).strip()
        s = s.replace('-', '_').replace(' ', '_')
        cleaned.append(s)
    return cleaned


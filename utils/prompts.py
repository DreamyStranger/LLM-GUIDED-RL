import ollama

ACTIONS_ALL = {
    0: 'LANE_LEFT',
    1: 'IDLE',
    2: 'LANE_RIGHT',
    3: 'FASTER',
    4: 'SLOWER'
}

def predict_action_gemma3(obs):
    """
    Use Gemma3-12B LLM to choose a discrete driving action (0–4) based on ego speed and TTC observations.

    Args:
        obs (np.ndarray): Observation array of shape (4,1) [ego_speed, left_ttc, center_ttc, right_ttc].

    Returns:
        int: Action index (0–4) selected by LLM, defaults to 1 (IDLE) on error.
    """
    ego_speed = obs[0, 0]
    left_ttc = obs[1, 0]
    current_ttc = obs[2, 0]
    right_ttc = obs[3, 0]
    
    prompt = f"""
        You are controlling an autonomous vehicle. Prioritize **safety first**, then consider efficiency and human-like behavior.

        Guidelines:
        - Avoid any action that risks a collision.
        - Treat any Time-To-Collision (TTC) below 2 seconds as dangerous.
        - Speed up only if the current lane TTC (center lane) is clearly safe (e.g., above 3 seconds).
        - If the current lane TTC (center lane) is low (below 2 seconds), and either left or right lane TTC is high (above 3 seconds), prefer changing to the safest lane.
        - Maintain a target speed around 30 m/s if safe, but slowing down is acceptable when unsure.
        - Change lanes only if it clearly improves safety or avoids slower traffic.

        Available actions:
        0 = Turn Left  
        1 = Idle
        2 = Turn Right  
        3 = Go Faster  
        4 = Slow Down

        Current observations:
        - Speed: {ego_speed:.1f} m/s  
        - Left lane TTC: {left_ttc:.1f} s  
        - Center lane TTC: {current_ttc:.1f} s  
        - Right lane TTC: {right_ttc:.1f} s

        What is the safest and most reasonable driving action to take now? Respond with only the action number (0–4).
        """.strip()
    try:
        response = ollama.generate(
            model ="gemma3:12b",
            prompt=prompt,
            think=False,
        )
        
        # Clean output to ensure only number
        action = response['response'].strip()
        action_idx = int(action)

        if action_idx not in ACTIONS_ALL:
            raise ValueError(f"Invalid action {action_idx} returned by LLM")
            
        return action_idx

    except Exception as e:
        print(f"[GPT ERROR] {e}")
        return 1

    
def predict_action_qwen3(obs):
    """
    Use Qwen3 LLM to choose a discrete driving action (0–4) based on ego speed and TTC observations.

    Args:
        obs (np.ndarray): Observation array of shape (4,1) [ego_speed, left_ttc, center_ttc, right_ttc].

    Returns:
        int: Action index (0–4) selected by LLM, defaults to 1 (IDLE) on error.
    """
    ego_speed = obs[0, 0]
    left_ttc = obs[1, 0]
    current_ttc = obs[2, 0]
    right_ttc = obs[3, 0]
    
    prompt = f"""
        You are controlling an autonomous vehicle. Prioritize **safety first**, then consider efficiency and human-like behavior.

        Guidelines:
        - Avoid any action that risks a collision.
        - Treat any Time-To-Collision (TTC) below 2 seconds as dangerous.
        - Speed up only if the current lane TTC (center lane) is clearly safe (e.g., above 3 seconds).
        - If the current lane TTC (center lane) is low (below 2 seconds), and either left or right lane TTC is high (above 3 seconds), prefer changing to the safest lane.
        - Maintain a target speed around 30 m/s if safe, but slowing down is acceptable when unsure.
        - Change lanes only if it clearly improves safety or avoids slower traffic.

        Available actions:
        0 = Turn Left  
        1 = Idle
        2 = Turn Right  
        3 = Go Faster  
        4 = Slow Down

        Current observations:
        - Speed: {ego_speed:.1f} m/s  
        - Left lane TTC: {left_ttc:.1f} s  
        - Center lane TTC: {current_ttc:.1f} s  
        - Right lane TTC: {right_ttc:.1f} s

        What is the safest and most reasonable driving action to take now? Respond with only the action number (0–4).
        """.strip()

    try:
        response = ollama.generate(
            model="qwen3:14b",
            prompt=prompt,
            think=False,
        )
        
        # Clean output to ensure only number
        action = response['response'].strip()
        action_idx = int(action)

        if action_idx not in ACTIONS_ALL:
            raise ValueError(f"Invalid action {action_idx} returned by LLM")

        return action_idx

    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return 1  # fallback to IDLE

def get_llm_shaping_score_lane_focused(prev_obs, action_taken, new_obs, model):
    """
    Evaluate lane-focused shaping score for a given action.

    Args:
        prev_obs (np.ndarray): Observation before action (4,1).
        action_taken (int): Index of action taken (0–4).
        new_obs (np.ndarray): Observation after action (4,1).
        model (str): LLM model to use for scoring.

    Returns:
        float: Score between 0 and 10 evaluating safety and smooth driving.
    """
    action_str = ACTIONS_ALL[action_taken]

    prompt = f"""
        You are evaluating the behavior of an autonomous vehicle in its current lane.

        TTC = Time To Collision — higher is safer.

        --- BEFORE ACTION ---
        - Ego speed: {prev_obs[0, 0]} m/s
        - Current lane TTC: {prev_obs[2, 0]} s

        --- ACTION TAKEN ---
        - {action_str}

        --- AFTER ACTION ---
        - Ego speed: {new_obs[0, 0]} m/s
        - Current lane TTC: {new_obs[2, 0]} s

        Score this action from 0 (very unsafe or inefficient) to 10 (excellent decision).
        Prioritize avoiding collisions, maintaining ~30 m/s if safe, and smooth, human-like driving.
        Respond with only the numeric score.
            """.strip()

    try:
        response = ollama.generate(
            model= model, 
            prompt=prompt,
            think=False,
        )
        score_str = response['response'].strip()
        score = float(score_str)
        return min(max(score, 0.0), 10.0)
    except Exception as e:
        print("Failed to parse LLM shaping score:", e)
        return 0.0

def predict_action_llm(llm_type, obs):
    """
    Wrapper to predict action using a specified LLM.

    Args:
        llm_type (str): "gemma3" or "qwen3".
        obs (np.ndarray): Observation array (4,1).

    Returns:
        int: Action index (0–4) predicted by the chosen LLM.
    """
    if llm_type == "gemma3":
        return predict_action_gemma3(obs)
    elif llm_type == "qwen3":
        return predict_action_qwen3(obs)
    else:
        raise ValueError(f"Unknown llm_type: {llm_type}")


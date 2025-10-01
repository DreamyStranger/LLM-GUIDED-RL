import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
from utils.obs_decoders import *
from utils.prompts import *

class ProgressCallback(BaseCallback):
    """
    Shows a tqdm progress bar for training steps.
    """
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self):
        """
        Initialize the progress bar at the start of training.
        """
        self.pbar = tqdm(total=self.total_timesteps, desc="Training progress")

    def _on_step(self) -> bool:
        """
        Update the progress bar at each step.

        Returns:
            bool: Always True to continue training.
        """
        self.pbar.update(1)
        return True

    def _on_training_end(self):
        """
        Close the progress bar at the end of training.
        """
        self.pbar.close()
    
class TTCWrapper(gym.Wrapper):
    """
    Gym wrapper to apply observation preprocessing and hybrid reward shaping.

    Modes:
        - RL: standard environment reward
        - Hybrid: collision penalty + LLM shaping

    Args:
        env: Base gym environment.
        mode (str): 'RL' or 'Hybrid'.
        model (str): LLM model to use for shaping.
        shape_reward (str): Type of shaping ('dense', 'avg', 'center').
    """
    def __init__(self, env, mode='RL', llm = "qwen3:14b", shape_reward = "dense"):
        super().__init__(env)
        self.mode = mode
        self.prev_obs = None
        self.ego = None
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(4,1),
            dtype=np.float32
        )
        self.llm = llm
        self.shape_reward = shape_reward

    def reset(self, **kwargs):
        """
        Reset the environment and preprocess initial observation.

        Returns:
            np.ndarray: Preprocessed initial observation.
            dict: Info from base environment.
        """
        raw_obs, info = self.env.reset(**kwargs)
        self.ego = self.env.unwrapped.controlled_vehicles[0] 
        ego_speed = np.linalg.norm(self.ego.velocity)
        processed_obs = preprocess_obs(raw_obs, ego_speed)
        self.prev_obs = processed_obs
        return processed_obs, info

    def step(self, action):
        """
        Step the environment and apply hybrid reward if enabled.

        Args:
            action (int): Discrete action to take.

        Returns:
            np.ndarray: Preprocessed next observation.
            float: Reward after shaping (if Hybrid).
            bool: Done flag.
            bool: Truncated flag.
            dict: Additional info from environment.
        """
        raw_obs, base_reward, done, truncated, info = self.env.step(action)
        ego_speed = np.linalg.norm(self.ego.velocity)
        processed_obs = preprocess_obs(raw_obs, ego_speed)
        
        #print("Ego Speed:", ego_speed)
        #print("\nCollected Observation:\n", raw_obs)
        #print("Processed Observation:\n", processed_obs)

        total_reward = base_reward

        if self.mode == 'Hybrid' and not done:
            total_reward = self.compute_hybdrid_reward(self.prev_obs, action, processed_obs, base_reward)
            
        self.prev_obs = processed_obs
        return processed_obs, total_reward, done, truncated, info
    

    def compute_hybdrid_reward(self, prev_obs, action, next_obs, reward):
        """
        Compute hybrid reward using lane-focused shaping.

        Args:
            prev_obs (np.ndarray): Previous observation.
            action (int): Action taken.
            next_obs (np.ndarray): Next observation.
            reward (float): Base environment reward.

        Returns:
            float: Total reward after shaping.
        """
        if self.shape_reward == "dense":
            shape_reward = get_llm_shaping_score_lane_focused(prev_obs, action, next_obs, self.llm)
            total_reward= reward + 0.1* shape_reward
            return total_reward
        elif self.shape_reward == "avg":
            shape_reward = get_llm_shaping_score_lane_focused(prev_obs, action, next_obs, self.llm)
            total_reward = reward + 0.1 * shape_reward
            return total_reward / 2
        else:
            shape_reward = (get_llm_shaping_score_lane_focused(prev_obs, action, next_obs, self.llm) - 5) / 5
            k = 0.5
            total_reward = reward + k*shape_reward
            total_reward = (total_reward + k) / (1 + 2* k)
            return total_reward
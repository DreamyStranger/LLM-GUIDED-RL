import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
from utils.obs_decoders import *
from utils.prompts import *

#qwen3:14b
#gemma3:12b

MODEL = "qwen3:14b"
SHAPE_REWARD = "CENTER"

class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps, desc="Training progress")

    def _on_step(self) -> bool:
        self.pbar.update(1)
        return True

    def _on_training_end(self) -> None:
        self.pbar.close()
    
class TTCWrapper(gym.Wrapper):
    """
    Applies either the base environment reward (RL mode)
    or collision penalty + LLM shaping (Hybrid mode).
    """
    def __init__(self, env, mode='RL'):
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

    def reset(self, **kwargs):
        raw_obs, info = self.env.reset(**kwargs)
        self.ego = self.env.unwrapped.controlled_vehicles[0] 
        ego_speed = np.linalg.norm(self.ego.velocity)
        processed_obs = preprocess_obs(raw_obs, ego_speed)
        self.prev_obs = processed_obs
        return processed_obs, info

    def step(self, action):
        raw_obs, base_reward, done, truncated, info = self.env.step(action)
        ego_speed = np.linalg.norm(self.ego.velocity)
        processed_obs = preprocess_obs(raw_obs, ego_speed)
        
        #print("Ego Speed:", ego_speed)
        #print("\nCollected Observation:\n", raw_obs)
        #print("Processed Observation:\n", processed_obs)

        total_reward = base_reward

        if self.mode == 'Hybrid' and not done:
            total_reward = compute_reward(self.prev_obs, action, processed_obs, base_reward)
            
        self.prev_obs = processed_obs
        return processed_obs, total_reward, done, truncated, info
    

def compute_reward(prev_obs, action, next_obs, reward):
    if SHAPE_REWARD == "DENSE":
        shape_reward = get_llm_shaping_score_lane_focused(prev_obs, action, next_obs, MODEL)
        total_reward= reward + 0.1* shape_reward
        return total_reward
    elif SHAPE_REWARD == "AVG":
        shape_reward = get_llm_shaping_score_lane_focused(prev_obs, action, next_obs, MODEL)
        total_reward = reward + 0.1 * shape_reward
        return total_reward / 2
    else:
        shape_reward = (get_llm_shaping_score_lane_focused(prev_obs, action, next_obs, MODEL) - 5) / 5
        k = 0.5
        total_reward = reward + k*shape_reward
        total_reward = (total_reward + k) / (1 + 2* k)
        return total_reward
        

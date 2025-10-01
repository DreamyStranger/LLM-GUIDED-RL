import gymnasium as gym
import highway_env

from utils.wrappers import *
from utils.io_configs import *
from utils.plots import *

def main():
    # 1) Model configs
    obs_space = get_observation_space()
    mode = "RL"
    llm_choice = get_llm_choice()
    env_config = load_env_config(obs_space)
    eval_id = get_evaluation_environment()
    env = gym.make(eval_id, render_mode="rgb_array", config=env_config)

    if obs_space == "ttc":
        env = TTCWrapper(env, mode=mode)

    # 2) Collect frames for all 4 episodes
    episode_frames = [[] for _ in range(4)]
    last_frames = [None]*4

    for episode in range(4):
        done = truncated = False
        obs, info = env.reset(seed=episode)
        frames = []

        while not (done or truncated):
            action = predict_action_llm(llm_choice, obs)
            obs, reward, done, truncated, info = env.step(action)
            frame = env.render()
            frames.append(frame)

        episode_frames[episode] = frames
        last_frames[episode] = frames[-1]

    # 3) Merge episodes into single 2x2 grid video
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

    # 4) Save merged video
    video_filename = "llm_episodes_2x2.mp4"
    path = build_dir_path(["videos", llm_choice, eval_id])
    save_frames_as_video(all_grid_frames, path, video_filename)
    print(f"Saved merged 2x2 grid video to {video_filename}")

if __name__ == "__main__":
    main()

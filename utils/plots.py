import imageio
import os

def save_frames_as_video(frames, path, filename, fps=30):
    """
    Save a list of RGB frames as a video file.

    Args:
        frames: List of RGB frames (numpy arrays).
        filename: Output video filename (e.g., "episode_1.mp4").
        fps: Frames per second.
        path: Directory where the video should be saved.
    """
    # Ensure path exists
    os.makedirs(path, exist_ok=True)

    # Full file path
    full_path = os.path.join(path, filename)

    # Convert frames to uint8 if needed
    frames_uint8 = [frame.astype('uint8') for frame in frames]
    
    # Save video
    imageio.mimsave(full_path, frames_uint8, fps=fps)
    print(f"Video saved to {full_path}")
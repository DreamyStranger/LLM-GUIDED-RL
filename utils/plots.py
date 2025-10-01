import imageio
import os
import numpy as np
from PIL import Image, ImageOps

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

def make_grid_with_borders(frames_list, grid_shape=(2,2), border_size=5, border_color=(0,0,0), scale_factor=1.5):
    """
    Combine frames into a grid with borders and optional scaling.

    Args:
        frames_list (list[np.ndarray]): List of frames to combine.
        grid_shape (tuple[int, int], optional): Grid shape as (rows, cols). Default is (2,2).
        border_size (int, optional): Border thickness in pixels. Default is 5.
        border_color (tuple[int, int, int], optional): Border color as RGB tuple. Default is black.
        scale_factor (float, optional): Scaling factor for each frame. Default is 1.5.

    Returns:
        np.ndarray: Combined grid frame as a single RGB image array.
    """
    pil_frames = [Image.fromarray(f) for f in frames_list]

    # Scale frames
    w, h = pil_frames[0].size
    new_w, new_h = int(w*scale_factor), int(h*scale_factor)

    bordered_frames = []
    for f in pil_frames:
        f_resized = f.resize((new_w - 2*border_size, new_h - 2*border_size))
        f_bordered = ImageOps.expand(f_resized, border=border_size, fill=border_color)
        bordered_frames.append(f_bordered)

    total_slots = grid_shape[0] * grid_shape[1]
    while len(bordered_frames) < total_slots:
        blank = Image.new("RGB", (new_w, new_h), color=border_color)
        bordered_frames.append(blank)

    # Stack into rows
    rows = []
    for r in range(grid_shape[0]):
        row = np.hstack([np.array(bordered_frames[r*grid_shape[1] + c]) for c in range(grid_shape[1])])
        rows.append(row)

    grid_frame = np.vstack(rows)
    return grid_frame
import numpy as np

def decode_single_direction(one_hot: np.ndarray, time_bin_size: float, horizon: float) -> float:
    """
    Decode a 1D one-hot TTC vector to a scalar TTC value.
    Returns horizon if no bin is active.
    """
    indices = np.where(one_hot > 0)[0]
    if len(indices) == 0:
        return horizon
    return (indices[0] + 1) * time_bin_size

def preprocess_obs(raw_obs: np.ndarray, ego_speed: float, time_bin_size: float = 1.0) -> np.ndarray:
    """
    Simplified preprocessing:
    - Select the central speed bin from raw TTC obs (shape: speeds x lanes x time_bins)
    - Decode one-hot TTC vector per lane into scalar TTC
    - Stack ego speed (rounded int) with decoded TTC scalars (shape: 4 x 1)
    """
    speeds, lanes, time_bins = raw_obs.shape
    horizon = time_bins * time_bin_size

    center_speed_idx = speeds // 2  # always central speed bin
    decoded_ttc = np.zeros(lanes, dtype=np.float32)

    for lane in range(lanes):
        one_hot = raw_obs[center_speed_idx, lane, :]
        decoded_ttc[lane] = decode_single_direction(one_hot, time_bin_size, horizon)

    ego_speed_int = int(round(ego_speed))
    ego_speed_array = np.array([ego_speed_int], dtype=np.float32)

    # Stack ego speed on top of lane TTCs, resulting in shape (4, 1)
    processed_obs = np.hstack((ego_speed_array, decoded_ttc)).reshape((4, 1))

    return processed_obs

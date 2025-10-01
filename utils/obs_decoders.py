import numpy as np

def decode_single_direction(one_hot: np.ndarray, time_bin_size: float, horizon: float) -> float:
    """
    Decode a 1D one-hot TTC vector into a scalar TTC value.

    Args:
        one_hot (np.ndarray): one-hot vector representing TTC bins.
        time_bin_size (float): Size of each time bin.
        horizon (float): Maximum horizon value to return if no bin is active.

    Returns:
        float: Decoded TTC value. Returns horizon if no bin is active.
    """
    indices = np.where(one_hot > 0)[0]
    if len(indices) == 0:
        return horizon
    return (indices[0] + 1) * time_bin_size

def preprocess_obs(raw_obs: np.ndarray, ego_speed: float, time_bin_size: float = 1.0) -> np.ndarray:
    """
    Preprocess a raw TTC observation into a compact feature vector.

    Steps:
        - Select the central speed bin from raw TTC obs 
          (input shape: speeds x lanes x time_bins).
        - Decode one-hot TTC vector per lane into a scalar TTC.
        - Stack ego speed (rounded to int) with decoded TTC scalars.

    Args:
        raw_obs (np.ndarray): Raw TTC observation (3D array).
        ego_speed (float): Ego vehicle speed.
        time_bin_size (float, optional): Size of each TTC bin. Defaults to 1.0.

    Returns:
        np.ndarray: Processed observation vector of shape (4, 1),
                    where the first element is ego speed,
                    followed by TTC values for each lane.
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

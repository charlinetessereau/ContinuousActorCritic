"""Common functions used across the repository."""
import numpy as np

def calculate_place_cell_activity(pos, centers, amp, sigma):
    """Calculate place cell activity for a given position."""
    diff = pos.reshape(2, 1) - centers
    dist_sq = np.sum(diff**2, axis=0)
    return amp * np.exp(-dist_sq / (2 * sigma**2)) 
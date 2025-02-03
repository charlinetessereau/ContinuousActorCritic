"""Functions for plotting trajectories."""
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import h5py
from analysis.core import load_experiment_data

def plot_trajectory_comparison(filename, rat_idx=0, trial_idx1=0, trial_idx2=-1, save_path=None):
    """Plot trajectories from two different trials side by side for comparison.
    
    Parameters:
    -----------
    filename : str
        Path to the HDF5 results file
    rat_idx : int
        Index of the rat to analyze
    trial_idx1 : int
        Index of first trial to plot
    trial_idx2 : int
        Index of second trial to plot. If -1, uses the last trial.
    save_path : str, optional
        If provided, saves the plot to this path instead of displaying
    """
    parameters, features = load_experiment_data(filename)
    
    with h5py.File(filename, 'r') as f:
        rat_group = f['results'][f'rat_{rat_idx}']
        platform_pos = rat_group.attrs['platform']  # Get platform position from rat group
        
        # Get number of trials if needed
        if trial_idx2 == -1:
            trial_idx2 = len(rat_group.keys()) - 1
            
        # Load trajectory data for both trials
        trial1 = rat_group[f'trial_{trial_idx1}']
        trial2 = rat_group[f'trial_{trial_idx2}']
        
        traj1 = trial1['trajectories'][:]
        traj2 = trial2['trajectories'][:]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot trials (now using the same platform_pos for both)
    _plot_single_trajectory(ax1, traj1, platform_pos, parameters, title=f'Trial {trial_idx1+1}')
    _plot_single_trajectory(ax2, traj2, platform_pos, parameters, title=f'Trial {trial_idx2+1}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f'trajectory_comparison_{trial_idx1}_{trial_idx2}.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def _plot_single_trajectory(ax, trajectory, platform_pos, parameters, title=''):
    """Helper function to plot a single trajectory."""
    # Plot arena boundary
    arena_radius = parameters['env']['arena_radius']
    arena_circle = Circle((0, 0), arena_radius, fill=False, color='mediumseagreen', linewidth=2)
    ax.add_patch(arena_circle)
    
    # Plot trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], color='darkred', linewidth=2)
    
    # Plot platform with circle
    goal_radius = parameters['env']['goal_radius']
    platform = Circle(platform_pos, goal_radius, color='red', fill=False, linewidth=2)
    ax.add_patch(platform)
    
    # Set equal aspect ratio and limits
    ax.set_aspect('equal')
    ax.set_xlim(-arena_radius-1, arena_radius+1)
    ax.set_ylim(-arena_radius-1, arena_radius+1)
    
    # Remove axes for cleaner look
    ax.set_axis_off()
    
    # Add title
    ax.set_title(title, pad=10) 
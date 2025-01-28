"""
Analysis and visualization tools for the continuous actor-critic model results.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns

def load_results(filename):
    """Load results from HDF5 file and extract key metrics."""
    with h5py.File(filename, 'r') as f:
        num_rats = len(f.keys())
        num_trials = len(f[f'rat_0'].keys())
        
        # Extract latencies for all rats and trials
        latencies = np.zeros((num_rats, num_trials))
        for rat in range(num_rats):
            for trial in range(num_trials):
                latencies[rat, trial] = f[f'rat_{rat}/trial_{trial}'].attrs['latency']
    
    return latencies

def plot_learning_curve(filename, save_path=None):
    """Plot learning curve showing mean latency across rats with error bars."""
    # Set style
    plt.style.use('seaborn')
    
    # Load data
    latencies = load_results(filename)
    num_trials = latencies.shape[1]
    
    # Compute statistics
    mean_latencies = np.mean(latencies, axis=0)
    sem_latencies = np.std(latencies, axis=0) / np.sqrt(latencies.shape[0])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot mean latency with error bars
    trials = np.arange(1, num_trials + 1)
    ax.plot(trials, mean_latencies, '-', color='darkgreen', linewidth=2, label='Mean Latency')
    ax.errorbar(trials, mean_latencies, yerr=sem_latencies, fmt='o', color='black', 
                capsize=5, capthick=1, elinewidth=1)
    
    # Customize axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add labels and title
    ax.set_xlabel('Trial Number', fontsize=14)
    ax.set_ylabel('Latency (s)', fontsize=14)
    ax.set_title('Learning Performance', fontsize=16)
    
    # Adjust ticks
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_trajectory(filename, rat_idx=0, trial_idx=0, save_path=None):
    """Plot trajectory for a specific rat and trial."""
    with h5py.File(filename, 'r') as f:
        trial = f[f'rat_{rat_idx}/trial_{trial_idx}']
        trajectory = trial['trajectories'][:]
        platform_pos = trial.attrs['platform']
        
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot arena boundary
    circle = Circle((0, 0), 100, fill=False, color='black', linestyle='--')
    ax.add_patch(circle)
    
    # Plot trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], '-', color='blue', alpha=0.6, 
            label='Trajectory')
    
    # Plot start and end points
    ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', label='Start')
    ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', label='End')
    
    # Plot platform
    platform = Circle(platform_pos, 5, color='gray', alpha=0.3, label='Platform')
    ax.add_patch(platform)
    
    # Set equal aspect ratio and limits
    ax.set_aspect('equal')
    ax.set_xlim(-110, 110)
    ax.set_ylim(-110, 110)
    
    # Add labels and title
    ax.set_xlabel('X Position (cm)', fontsize=12)
    ax.set_ylabel('Y Position (cm)', fontsize=12)
    ax.set_title(f'Trajectory (Rat {rat_idx}, Trial {trial_idx})', fontsize=14)
    
    # Add legend
    ax.legend(fontsize=10)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

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
    with h5py.File(filename, 'r') as f:
        # Get number of trials if needed
        if trial_idx2 == -1:
            trial_idx2 = len(f[f'rat_{rat_idx}']) - 1
            
        # Load data for both trials
        trial1 = f[f'rat_{rat_idx}/trial_{trial_idx1}']
        trial2 = f[f'rat_{rat_idx}/trial_{trial_idx2}']
        
        traj1 = trial1['trajectories'][:]
        traj2 = trial2['trajectories'][:]
        platform1 = trial1.attrs['platform']
        platform2 = trial2.attrs['platform']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot first trial
    _plot_single_trajectory(ax1, traj1, platform1, title=f'Trial {trial_idx1+1}')
    
    # Plot second trial
    _plot_single_trajectory(ax2, traj2, platform2, title=f'Trial {trial_idx2+1}')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def _plot_single_trajectory(ax, trajectory, platform_pos, title=''):
    """Helper function to plot a single trajectory."""
    # Plot arena boundary
    arena_circle = Circle((0, 0), 100, fill=False, color='mediumseagreen', linewidth=2)
    ax.add_patch(arena_circle)
    
    # Plot trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], color='darkred', linewidth=2)
    
    # Plot platform with circle
    platform = Circle(platform_pos, 5, color='red', fill=False, linewidth=2)
    ax.add_patch(platform)
    
    # Set equal aspect ratio and limits
    ax.set_aspect('equal')
    ax.set_xlim(-101, 101)
    ax.set_ylim(-101, 101)
    
    # Remove axes for cleaner look
    ax.set_axis_off()
    
    # Add title
    ax.set_title(title, pad=10)

def plot_partial_trajectory(filename, rat_idx=0, trial_idx=0, num_steps=50, save_path=None):
    """Plot partial trajectory to show initial search behavior.
    
    Parameters:
    -----------
    filename : str
        Path to the HDF5 results file
    rat_idx : int
        Index of the rat to analyze
    trial_idx : int
        Index of trial to plot
    num_steps : int
        Number of steps to plot
    save_path : str, optional
        If provided, saves the plot to this path instead of displaying
    """
    with h5py.File(filename, 'r') as f:
        trial = f[f'rat_{rat_idx}/trial_{trial_idx}']
        trajectory = trial['trajectories'][:num_steps]
        platform_pos = trial.attrs['platform']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot arena boundary
    arena_circle = Circle((0, 0), 100, fill=False, color='mediumseagreen', linewidth=3)
    ax.add_patch(arena_circle)
    
    # Plot partial trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], color='darkred', linewidth=3)
    
    # Plot platform with circle
    platform = Circle(platform_pos, 5, color='red', fill=False, linewidth=3)
    ax.add_patch(platform)
    
    # Set equal aspect ratio and limits
    ax.set_aspect('equal')
    ax.set_xlim(-101, 101)
    ax.set_ylim(-101, 101)
    
    # Remove axes for cleaner look
    ax.set_axis_off()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

if __name__ == '__main__':
    # Example usage
    results_file = 'results.h5'
    
    # Plot learning curve
    plot_learning_curve(results_file, 'learning_curve.png')
    
    # Plot trajectory comparison (first vs last trial)
    plot_trajectory_comparison(results_file, rat_idx=0, trial_idx1=0, trial_idx2=-1,
                             save_path='trajectory_comparison.png')
    
    # Plot partial trajectory to show initial search behavior
    plot_partial_trajectory(results_file, rat_idx=0, trial_idx=0, num_steps=50,
                          save_path='partial_trajectory.png') 
"""Functions for plotting value functions."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from functions import calculate_place_cell_activity
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import pathpatch_2d_to_3d
from analysis.core import load_experiment_data
import h5py

def plot_value_comparison(filename, rat_idx=0, trial_idx1=4, trial_idx2=5, save_path=None):
    """Plot value functions from two different trials side by side.
    
    Args:
        filename (str): Path to the data file
        rat_idx (int): Index of the rat to analyze
        trial_idx1 (int): Index of first trial to plot
        trial_idx2 (int): Index of second trial to plot
        save_path (str, optional): Path to save the figure
    """
    # Load data using core function
    parameters, _ = load_experiment_data(filename)
    
    # Get parameters from loaded data
    R = parameters['env']['arena_radius']  # Pool radius
    goal_radius = parameters['env']['goal_radius']
    place_cell_centers = parameters['pc']['centres']
    amp = parameters['pc']['amplitude']
    sigma = parameters['pc']['sigma']
    
    # Load trial-specific data
    with h5py.File(filename, 'r') as f:
        rat_group = f['results'][f'rat_{rat_idx}']
        platform_pos = rat_group.attrs['platform']  # Get platform position from rat group
        
        # Load value weights for both trials
        trial1 = rat_group[f'trial_{trial_idx1}']
        trial2 = rat_group[f'trial_{trial_idx2}']
        value_weights1 = trial1['final_critic_weights'][:]
        value_weights2 = trial2['final_critic_weights'][:]
    
    # Create grid of points
    steps = 2
    x = np.arange(-R, R + steps, steps)
    y = np.arange(-R, R + steps, steps)
    X, Y = np.meshgrid(x, y)
    
    # Initialize value maps with NaN values (using float dtype)
    v1 = np.full_like(X, np.nan, dtype=float)
    v2 = np.full_like(X, np.nan, dtype=float)
    
    # Calculate value function for each point in grid
    for i in range(len(x)):
        for j in range(len(y)):
            if np.sqrt(X[i,j]**2 + Y[i,j]**2) < R:
                pos = np.array([X[i,j], Y[i,j]])
                # Calculate place cell activity
                F = calculate_place_cell_activity(pos, place_cell_centers, amp, sigma)
                
                # Calculate critic values using weights from both trials
                v1[i,j] = np.mean(F @ value_weights1)
                v2[i,j] = np.mean(F @ value_weights2)
    
    # Create figure
    fig = plt.figure(figsize=(12, 5))
    
    # Plot both value functions
    for idx, (v, trial_idx) in enumerate([
        (v1, trial_idx1), 
        (v2, trial_idx2)
    ]):
        ax = fig.add_subplot(1, 2, idx+1)
        
        # Plot value heatmap
        im = ax.pcolormesh(X, Y, v, cmap='viridis')
        
        # Plot pool boundary
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(R*np.cos(theta), R*np.sin(theta), 
               'dimgray', linewidth=2, zorder=1)
        
        # Plot platform (now using the same platform_pos for both plots)
        platform_circle = plt.Circle(platform_pos, goal_radius, 
                                   color='red', alpha=1, zorder=2, linewidth=3)
        ax.add_patch(platform_circle)
        
        # Set labels and appearance
        ax.set_xlabel('X Position (cm)')
        ax.set_ylabel('Y Position (cm)')
        ax.set_title(f'Trial {trial_idx+1}')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Set axis limits and remove axis
        ax.set_xlim([-R-1, R+1])
        ax.set_ylim([-R-1, R+1])
        ax.set_axis_off()
    
    plt.suptitle(f'Value Maps - Rat {rat_idx}')
    
    if save_path:
        plt.savefig(save_path+f'/value_comparison_{trial_idx1}_{trial_idx2}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig 
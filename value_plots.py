import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns

def plot_value_comparison(filename, rat_idx=0, trial_idx1=0, trial_idx2=-1, save_path=None):
    """Plot value functions from two different trials side by side for comparison."""
    with h5py.File(filename, 'r') as f:
        rat_group = f['results'][f'rat_{rat_idx}']
        
        # Get number of trials if needed
        if trial_idx2 == -1:
            trial_idx2 = len(rat_group.keys()) - 1
            
        # Load data for both trials
        trial1 = rat_group[f'trial_{trial_idx1}']
        trial2 = rat_group[f'trial_{trial_idx2}']
        
        # Load critic weights to compute value maps
        cw1 = trial1['final_critic_weights'][:]
        cw2 = trial2['final_critic_weights'][:]
        platform1 = trial1.attrs['platform']
        platform2 = trial2.attrs['platform']
        
        # Load parameters for arena dimensions
        params = f['parameters']
        R = params['env']['arena_radius'][()]
        r = params['env']['goal_radius'][()]
        
        # Compute value maps (you'll need to implement this based on your place cell setup)
        value_map1 = compute_value_map(cw1, params)
        value_map2 = compute_value_map(cw2, params)
    
    # Create figure and plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    _plot_single_value(ax1, value_map1, platform1, R, r, title=f'Trial {trial_idx1+1}')
    _plot_single_value(ax2, value_map2, platform2, R, r, title=f'Trial {trial_idx2+1}')
    
    plt.colorbar(ax1.collections[0], ax=ax1)
    plt.colorbar(ax2.collections[0], ax=ax2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def compute_value_map(critic_weights, params):
    """Compute value map from critic weights and place cell parameters."""
    # Create grid
    R = params['env']['arena_radius'][()]
    steps = 0.5
    x = np.arange(-R, R + steps, steps)
    X, Y = np.meshgrid(x, x)
    
    # Initialize value map
    value_map = np.zeros_like(X)
    
    # Compute place cell activity and value for each point
    for i in range(len(x)):
        for j in range(len(x)):
            if X[i,j]**2 + Y[i,j]**2 < R**2:  # Only compute for points inside arena
                pos = np.array([X[i,j], Y[i,j]])
                pc_activity = placecells(pos, params)  # You'll need to import this
                value_map[i,j] = np.dot(critic_weights.T, pc_activity)[0]
    
    return value_map

def _plot_single_value(ax, value_map, platform_pos, R, r, title=''):
    """Helper function to plot a single value function."""
    # Create grid
    steps = 0.5
    x = np.arange(-R, R + steps, steps)
    X, Y = np.meshgrid(x, x)
    
    # Create mask for points outside arena
    mask = X**2 + Y**2 >= R**2
    value_map_masked = np.ma.array(value_map, mask=mask)
    
    # Plot value function
    im = ax.pcolormesh(x, x, value_map_masked, cmap='viridis')
    
    # Plot arena boundary
    circle = Circle((0, 0), R, fill=False, color='dimgray', linewidth=2)
    ax.add_patch(circle)
    
    # Plot platform
    platform = Circle(platform_pos, r, color='red', fill=False, linewidth=3)
    ax.add_patch(platform)
    
    # Set equal aspect ratio and limits
    ax.set_aspect('equal')
    ax.set_xlim(-R-1, R+1)
    ax.set_ylim(-R-1, R+1)
    
    # Remove axes for cleaner look
    ax.set_axis_off()
    
    # Add title
    ax.set_title(title, pad=10)
    
    return im
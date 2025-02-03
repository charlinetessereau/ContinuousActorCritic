import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import h5py

def setup_figure():
    """Create and setup the figure and axes."""
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3)
    
    ax_linear = fig.add_subplot(gs[:2, :2])  # Left panel
    ax_polar = fig.add_subplot(gs[0, 2], projection='polar')  # Top right
    ax_trajectory = fig.add_subplot(gs[1, 2])  # Bottom right
    
    # Setup axes
    ax_linear.set_xlabel('Angle (rad)')
    ax_linear.set_ylabel('Activity')
    ax_polar.set_rticks([])
    ax_trajectory.set_aspect('equal')
    ax_trajectory.set_axis_off()  # Hide trajectory axes
    
    # Fix legend position
    ax_linear.set_anchor('W')  # Anchor left side
    
    return fig, (ax_linear, ax_polar, ax_trajectory)

def setup_lines(axes, arena_radius=100):
    """Setup all plot lines."""
    ax_linear, ax_polar, ax_trajectory = axes
    
    # Define colors for consistency
    colors = {
        'actor': 'darkred',
        'pc': 'darkgoldenrod',
        'noise': 'darkgreen',
        'direction': 'black'
    }
    
    lines = {}
    # Linear plot lines
    lines['ua'] = ax_linear.plot([], [], '-', lw=2, color=colors['actor'], label='Actor')[0]
    lines['pc'] = ax_linear.plot([], [], '-', lw=1, color=colors['pc'], label='Place Cells')[0]
    lines['noise'] = ax_linear.plot([], [], '-', lw=1, color=colors['noise'], label='Noise')[0]
    lines['direction'] = ax_linear.axvline(0, color=colors['direction'], lw=1, label='Chosen Direction')
    
    # Polar plot lines
    lines['ua_polar'] = ax_polar.plot([], [], '-', lw=2, color=colors['actor'])[0]
    lines['pc_polar'] = ax_polar.plot([], [], '-', lw=1, color=colors['pc'])[0]
    lines['noise_polar'] = ax_polar.plot([], [], '-', lw=1, color=colors['noise'])[0]
    lines['direction_polar'] = ax_polar.plot([], [], '-', lw=1, color=colors['direction'])[0]
    
    # Trajectory plot lines
    lines['trajectory'] = ax_trajectory.plot([], [], 'b-', lw=1)[0]
    lines['position'] = ax_trajectory.plot([], [], 'ro')[0]
    
    # Add arena boundary and platform
    arena_circle = plt.Circle((0, 0), arena_radius, fill=False, color='gray', linestyle='--')
    ax_trajectory.add_artist(arena_circle)
    lines['platform'] = plt.Circle((0, 0), 5, fill=True, color='green', alpha=0.3)
    ax_trajectory.add_artist(lines['platform'])
    
    # Set axis limits
    ax_linear.set_xlim(0, 2*np.pi)
    ax_trajectory.set_xlim(-arena_radius-5, arena_radius+5)
    ax_trajectory.set_ylim(-arena_radius-5, arena_radius+5)
    
    # Fix legend position
    ax_linear.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    return lines

def update_plot(frame_data, lines, axes):
    """Update all plot lines with new data."""
    ua, pc_input, noise_input, trajectory, current_pos, platform_pos = frame_data
    
    # Find global min/max for y-axis scaling
    all_data = [ua, pc_input, noise_input]
    y_min = min(np.min(data) for data in all_data)
    y_max = max(np.max(data) for data in all_data)
    margin = (y_max - y_min) * 0.1  # Add 10% margin
    
    # Update y-axis limits
    ax_linear, ax_polar, _ = axes
    ax_linear.set_ylim(y_min - margin, y_max + margin)
    ax_polar.set_ylim(0, y_max + margin)  # Start from 0 for polar plot
    
    # Update linear plot
    angles = np.linspace(0, 2*np.pi, len(ua), endpoint=False)
    lines['ua'].set_data(angles, ua)
    lines['pc'].set_data(angles, pc_input)
    lines['noise'].set_data(angles, noise_input)
    
    # Compute direction
    direction = np.dot(ua, np.column_stack([np.cos(angles), np.sin(angles)]))
    if np.linalg.norm(direction) > 0:
        direction = direction / np.linalg.norm(direction)
    angle = np.arctan2(direction[1], direction[0])
    if angle < 0:
        angle += 2*np.pi
        
    # Update direction lines
    lines['direction'].set_xdata([angle, angle])
    
    # Update polar direction line (from center to edge)
    r_max = ax_polar.get_ylim()[1]  # Get the maximum radius
    lines['direction_polar'].set_data([angle, angle], [0, r_max])
    
    # Update polar plot
    lines['ua_polar'].set_data(angles, ua)
    lines['pc_polar'].set_data(angles, pc_input)
    lines['noise_polar'].set_data(angles, noise_input)
    
    # Update trajectory plot
    if len(trajectory) > 0:  # Only plot if we have trajectory data
        lines['trajectory'].set_data(trajectory[:, 0], trajectory[:, 1])
        lines['position'].set_data([current_pos[0]], [current_pos[1]])
        
        # Update platform position
        lines['platform'].center = platform_pos

def load_trial_data(h5_file, rat_index=1, trial_index=19):
    """Load data for a specific trial."""
    with h5py.File(h5_file, 'r') as f:
        trial_data = f[f'results/rat_{rat_index}/trial_{trial_index}']
        data = {
            'trajectories': trial_data['trajectories'][:],
            'history_ua': trial_data['history_ua'][:],
            'history_input_pc': trial_data['history_input_pc'][:],
            'history_input_noise': trial_data['history_input_noise'][:],
            'platform_pos': f[f'results/rat_{rat_index}'].attrs['platform']
        }
    return data

# For interactive use
def plot_frame(frame_idx, data, lines, axes):
    """Plot a single frame of the animation."""
    frame_data = (
        data['history_ua'][frame_idx],
        data['history_input_pc'][frame_idx],
        data['history_input_noise'][frame_idx],
        data['trajectories'][:frame_idx+1],
        data['trajectories'][frame_idx],
        data['platform_pos']  # Add platform position to frame data
    )
    update_plot(frame_data, lines, axes)
    plt.draw()
    plt.pause(0.01)

if __name__ == '__main__':
    # Example usage as script
    fig, axes = setup_figure()
    lines = setup_lines(axes)
    rat_index=5
    trial_index=0
    data = load_trial_data('results.h5', rat_index=rat_index, trial_index=trial_index)
    
    anim = animation.FuncAnimation(
        fig, 
        lambda i: update_plot((
            data['history_ua'][i],
            data['history_input_pc'][i],
            data['history_input_noise'][i],
            data['trajectories'][:i+1],
            data['trajectories'][i],
            data['platform_pos']  # Add platform position
        ), lines, axes),
        frames=len(data['history_ua']),
        interval=50,
        blit=False
    )
    
    anim.save('animation_'+str(rat_index)+'_'+str(trial_index)+'.mp4', writer='ffmpeg', fps=30)
    plt.close()

# First run these lines
fig, axes = setup_figure()
lines = setup_lines(axes)
data = load_trial_data('results.h5')

# Then plot individual frames
plot_frame(0, data, lines, axes)  # Plot first frame

# Or plot a range of frames
for i in range(0, 100, 5):  # Plot every 5th frame from 0 to 100
    plot_frame(i, data, lines, axes)
    plt.pause(0.1)  # Add delay between frames 
"""
Continuous Actor-Critic Learning Implementation with Place Cells

This module implements a continuous actor-critic reinforcement learning model
for spatial navigation using place cells.

HDF5 Data Structure:
-------------------
results.h5
├── parameters/                 # Group for all parameters
│   ├── env/                   # Environment parameters
│   ├── pc/                    # Place cell parameters
│   └── ...                    # Other parameter categories
├── features/                  # Group for features
│   ├── num_rats              # Number of rats
│   └── num_trials            # Number of trials per rat
└── results/                  # Group for all experimental results
    ├── rat_0/                # Group for first rat
    │   ├── platform          # Attribute: (x,y) platform position
    │   ├── trial_0/          # Group for first trial
    │   │   ├── trajectories  # Dataset: (n_steps, 2) array of positions
    │   │   ├── actions       # Dataset: (n_steps,) array of chosen actions
    │   │   ├── rewards       # Dataset: (n_steps,) array of rewards
    │   │   ├── td_errors     # Dataset: (n_steps,) array of TD errors
    │   │   ├── initial_actor_weights   # Dataset: Actor weights at start
    │   │   ├── initial_critic_weights  # Dataset: Critic weights at start
    │   │   ├── final_actor_weights     # Dataset: Actor weights at end
    │   │   ├── final_critic_weights    # Dataset: Critic weights at end
    │   │   └── latency       # Attribute: time to find platform
    │   ├── trial_1/
    │   └── ...
    ├── rat_1/
    └── ...

The HDF5 file structure allows for efficient storage and access of:
- Complete trajectories for each trial
- Action sequences and resulting rewards
- TD errors for learning analysis
- Trial latencies for learning performance
- Platform positions for reference
"""

import numpy as np
from scipy.integrate import solve_ivp
import json
import h5py
from utils import calculate_place_cell_activity

# Define core functions

def placecells(position, params):
    """Compute place cell activity for given position."""
    return calculate_place_cell_activity(
        position, 
        params['pc']['centres'],
        params['pc']['amplitude'],
        params['pc']['sigma']
    )

def activation_function(ua, params):
    """Compute activation function (sigmoid) for actor network.
    
    Following the original Julia implementation:
    return params[:ρ]*inv.(ones(length(UA)).+Base.exp.(-params[:β]*(UA.-params[:h]*ones(length(UA)))))
    """
    return params['actor']['rho'] / (1 + np.exp(-params['actor']['beta'] * (ua - params['actor']['h'])))

def compute_td_error(reward, uc, g, previous_g, params):
    """Compute TD error."""
    dv = params['critic']['nu'] * params['pc']['epsilon_pcc'] * (g - previous_g) / params['env']['dt']
    v = params['critic']['nu'] * uc + params['critic']['v0']
    return reward - v / params['reward']['tau_r'] + dv

def noise_dynamics(noise, rand_noise, params):
    """Update noise dynamics."""
    return noise * (1 - params['env']['dt'] / params['noise']['tau']) + rand_noise

def generate_noise_input(ua, input_pc, params):
    """Generate noise input for actor network."""
    # Find indices of maximum values
    max_indices = np.where(ua == np.max(ua))[0]
    
    if len(max_indices) == 1:  # Clear preference exists
        index_max = max_indices[0]
        scale_deviation = np.max(input_pc) - np.min(input_pc)
        if scale_deviation == 0:
            scale_deviation = 1
            
        # Generate probability distribution for noise center
        centre_prob_centre = int(np.mod(
            index_max + np.floor(np.random.choice([-1, 1]) * params['noise']['mu'] / scale_deviation * params['actor']['num_actions']),
            params['actor']['num_actions']
        ))
        
        # Generate weights for sampling noise center
        width_prob_centre = params['actor']['num_actions'] * params['noise']['sigma'] / scale_deviation
        prob_centre = np.array([
            np.exp(-(centre_prob_centre - k)**2 / (2 * width_prob_centre**2))
            for k in range(params['actor']['num_actions'])
        ])
        
        # Sample center from probability distribution
        centre_n = np.random.choice(params['actor']['num_actions'], p=prob_centre/np.sum(prob_centre))
    else:
        # No clear preference yet - choose random center
        centre_n = np.random.randint(params['actor']['num_actions'])
    
    # Generate noise distribution
    angles = np.linspace(0, 2*np.pi, params['actor']['num_actions'], endpoint=False)
    scale_deviation = max(np.max(input_pc) - np.min(input_pc), 1e-6)
    rho_n = params['noise']['rho'] / scale_deviation
    
    # Compute noise using sine-based angular differences (matching Julia implementation)
    rand_noise = np.array([
        rho_n * np.exp(-np.sin((angles[k] - angles[centre_n])/2)**2 / (2 * params['noise']['width']**2))
        for k in range(params['actor']['num_actions'])
    ])
    
    return rand_noise

def actor_activity_double_dyn(ua, va, input_pc, input_noise, params):
    """Update actor network activity using double dynamics."""
    dt = params['env']['dt']
    tau_m = params['actor']['tau_m']
    tau_s = params['actor']['tau_s']
    
    # Update membrane potential
    dua = (dt / tau_m) * (va - ua) + ua
    
    # Update synaptic current
    dva = (va * (1 - dt/tau_s) + 
           (params['actor']['epsilon_aa'] * dt/tau_s) * np.dot(params['actor']['coupled_aw'], activation_function(ua, params)) +
           (dt/tau_s) * params['actor']['epsilon_pca'] * input_pc +
           (dt/tau_s) * input_noise)
    
    return dua, dva

def critic_activity_double_dyn(uc, vc, input_pc, params):
    """Update critic network activity using double dynamics."""
    dt = params['env']['dt']
    tau_m = params['critic']['tau_m']
    tau_s = params['critic']['tau_s']
    
    # Update membrane potential
    duc = (dt / tau_m) * (vc - uc) + uc
    
    # Update synaptic current
    dvc = vc * (1 - dt/tau_s) + (params['pc']['epsilon_pcc'] * dt/tau_s) * input_pc
    
    return duc, dvc

def reward_function(x, y, xp, yp, hit_wall, ra, rb, params):
    """Compute reward and update reward dynamics."""
    dt = params['env']['dt']
    tau_ra = params['reward']['tau_ra']
    tau_rb = params['reward']['tau_rb']
    
    # Compute instantaneous reward
    if (x - xp)**2 + (y - yp)**2 <= params['env']['goal_radius']**2:
        R = params['reward']['goal_reward']
        found_goal = True
    elif hit_wall:
        R = params['reward']['wall_penalty']
        found_goal = False
    else:
        R = 0
        found_goal = False
        
    # Update reward dynamics
    new_ra = ra * (1 - dt/tau_ra) + R
    new_rb = rb * (1 - dt/tau_rb) + R
    
    # Compute effective reward
    re = (new_ra - new_rb)/(tau_ra - tau_rb)
    
    return new_ra, new_rb, re, found_goal

def choose_action(ua, params):
    """Compute movement direction as weighted sum of all action directions."""
    # Get normalized weights using softmax
    ua_shifted = ua - np.max(ua)
    exp_ua = np.exp(ua_shifted)
    weights = exp_ua / np.sum(exp_ua)
    
    # Compute weighted sum of direction vectors
    direction = np.dot(weights, params['actor']['directions'])
    
    # Normalize direction vector if not zero
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm
        
    # Convert to angle in degrees for consistency with rest of code
    angle = np.rad2deg(np.arctan2(direction[1], direction[0])) % 360
    return angle, direction

def update_position(position, action_data, params):
    """Update position based on chosen action direction vector."""
    dt = params['env']['dt']
    speed = params['env']['speed']
    arena_radius = params['env']['arena_radius']
    
    # Unpack action data
    angle, direction = action_data
    
    # Update position using direction vector
    new_pos = position + speed * dt * direction
    
    # Check wall collisions
    hit_wall = False
    if np.linalg.norm(new_pos) > arena_radius:
        new_pos = new_pos * (arena_radius / np.linalg.norm(new_pos))
        hit_wall = True
    
    return new_pos, hit_wall

def experiment(params, features, save_path):
    """Run the experiment and save results."""
    with h5py.File(save_path, 'w') as f:
        # Save parameters and features at top level
        param_group = f.create_group('parameters')
        for category, values in params.items():
            param_subgroup = param_group.create_group(category)
            for key, value in values.items():
                if isinstance(value, np.ndarray):
                    param_subgroup.create_dataset(key, data=value)
                else:
                    param_subgroup.attrs[key] = value

        feature_group = f.create_group('features')
        for key, value in features.items():
            feature_group.attrs[key] = value

        # Create results group for all rats
        results_group = f.create_group('results')
        
        for rat in range(features['num_rats']):
            rat_group = results_group.create_group(f'rat_{rat}')
            
            # Initialize weights for this rat
            aw = params['actor']['init_weights'].copy()
            cw = params['critic']['init_weights'].copy()
            
            # Select and store platform position once per rat
            xp, yp = params['env']['platform_positions'][np.random.randint(len(params['env']['platform_positions']))]
            rat_group.attrs['platform'] = [xp, yp]
            
            for trial in range(features['num_trials']):
                trial_group = rat_group.create_group(f'trial_{trial}')
                
                # Save initial weights for this trial
                trial_group.create_dataset('initial_actor_weights', data=aw)
                trial_group.create_dataset('initial_critic_weights', data=cw)
                
                # Initialize state variables
                position = np.array(params['env']['start_positions'][np.random.randint(len(params['env']['start_positions']))])
                ua = np.zeros(params['actor']['num_actions'])
                va = np.zeros(params['actor']['num_actions'])
                uc = np.zeros(1)
                vc = np.zeros(1)
                ra = rb = 0
                input_noise = np.zeros(params['actor']['num_actions'])
                
                # For visualization, store additional data for first and last trial
                store_viz_data = trial in [0, features['num_trials']-1]
                if store_viz_data:
                    history_ua = []
                    history_input_pc = []
                    history_input_noise = []
                
                # Create datasets with correct size including reward window
                total_time = params['env']['max_trial_time'] + params['env']['reward_window']
                max_steps = int(total_time / params['env']['dt']) + 10  # Add more buffer steps
                
                trajectories = np.zeros((max_steps, 2))
                actions = np.zeros(max_steps)
                rewards = np.zeros(max_steps)
                td_errors = np.zeros(max_steps)
                
                step = 0
                t = 0
                previous_g = 0
                timeout = False
                found_goal = False
                
                # Main trial loop
                while t < params['env']['max_trial_time'] and not found_goal and step < max_steps - 1:
                    # Check if time is up - place agent on platform
                    if np.isclose(t, params['env']['max_trial_time']):
                        position = np.array([xp, yp])
                        timeout = True
                        found_goal = True
                    
                    # Only process actor dynamics if we haven't timed out
                    if not timeout:
                        # Compute place cell activity
                        pc_activity = placecells(position, params)
                        
                        # Actor dynamics
                        input_pc = np.dot(aw.T, pc_activity)
                        rand_noise = generate_noise_input(ua, input_pc, params)
                        input_noise = noise_dynamics(input_noise, rand_noise, params)
                        
                        ua, va = actor_activity_double_dyn(ua, va, input_pc, input_noise, params)
                        
                        # Choose action and update position
                        action_data = choose_action(ua, params)  # Now returns (angle, direction)
                        new_position, hit_wall = update_position(position, action_data, params)
                        position = new_position
                    
                    # Store current state
                    trajectories[step] = position
                    
                    # Compute reward
                    ra, rb, re, found_goal = reward_function(
                        position[0], position[1], xp, yp, hit_wall, ra, rb, params
                    )
                    # print(hit_wall,ra,rb,re,found_goal)
                    # if hit_wall==True:
                    #     break
                    
                    # Critic dynamics
                    g = np.dot(cw.T, placecells(position, params))[0]
                    td_error = compute_td_error(re, uc[0], g, previous_g, params)
                    
                    # Update weights
                    if not timeout:
                        aw += (params['learning']['actor_lr'] * params['env']['dt'] * 
                              td_error * params['actor']['epsilon_pca'] * 
                              np.outer(pc_activity, ua))
                    
                    cw += (params['learning']['critic_lr'] * params['env']['dt'] * 
                          td_error * params['critic']['nu'] * 
                          params['pc']['epsilon_pcc'] * 
                          pc_activity.reshape(-1, 1))
                    
                    # Store data and update state
                    actions[step] = action_data[0] if not timeout else 0  # Store angle in degrees
                    rewards[step] = re
                    td_errors[step] = td_error
                    previous_g = g
                    t += params['env']['dt']
                    step += 1

                    if store_viz_data:
                        history_ua.append(ua.copy())
                        history_input_pc.append(input_pc.copy())
                        history_input_noise.append(input_noise.copy())

                # Post-trial reward learning period
                reward_window = t + params['env']['reward_window']
                while t < reward_window and step < max_steps - 1:  # Added bounds check
                    # Compute reward and critic updates during reward window
                    ra, rb, re, _ = reward_function(
                        position[0], position[1], xp, yp, False, ra, rb, params
                    )
                    # print(ra,rb,re)
                    
                    pc_activity = placecells(position, params)
                    g = np.dot(cw.T, pc_activity)[0]
                    td_error = compute_td_error(re, uc[0], g, previous_g, params)
                    
                    # Update weights (actor only updated if goal was found naturally)
                    if not timeout:
                        aw += (params['learning']['actor_lr'] * params['env']['dt'] * 
                              td_error * params['actor']['epsilon_pca'] * 
                              np.outer(pc_activity, ua))
                    
                    cw += (params['learning']['critic_lr'] * params['env']['dt'] * 
                          td_error * params['critic']['nu'] * 
                          params['pc']['epsilon_pcc'] * 
                          pc_activity.reshape(-1, 1))
                    
                    # Store data
                    trajectories[step] = position
                    actions[step] = 0
                    rewards[step] = re
                    td_errors[step] = td_error
                    
                    previous_g = g
                    t += params['env']['dt']
                    step += 1

                # Trim datasets to actual size used
                actual_steps = min(step, max_steps - 1)  # Ensure we don't exceed array bounds
                for name, data in [
                    ('trajectories', trajectories[:actual_steps]),
                    ('actions', actions[:actual_steps]),
                    ('rewards', rewards[:actual_steps]),
                    ('td_errors', td_errors[:actual_steps])
                ]:
                    trial_group.create_dataset(name, data=data)
                
                # Store metadata (removed platform position)
                trial_group.attrs['latency'] = t

                # After trial ends, save final weights
                trial_group.create_dataset('final_actor_weights', data=aw)
                trial_group.create_dataset('final_critic_weights', data=cw)

                # Save visualization data if needed
                if store_viz_data:
                    trial_group.create_dataset('history_ua', data=np.array(history_ua))
                    trial_group.create_dataset('history_input_pc', data=np.array(history_input_pc))
                    trial_group.create_dataset('history_input_noise', data=np.array(history_input_noise))

# Example usage
def initialize_parameters():
    """Initialize all parameters for the experiment from YAML config file."""
    import yaml
    import numpy as np
    
    # Load base parameters from YAML
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Extract dt for convenience
    dt = params['dt']
    
    # Add dt to env params
    params['env']['dt'] = dt
    
    ###################
    # Dynamic Parameter Computations
    ###################
    
    # Initialize place cell centers
    num_pc = params['pc']['num_cells']
    angles = np.random.uniform(0, 2*np.pi, num_pc)
    radii = np.sqrt(np.random.uniform(0, 1, num_pc)) * params['env']['arena_radius']
    radii = np.sort(radii)[::-1]
    params['pc']['centres'] = np.vstack([
        np.cos(angles) * radii,
        np.sin(angles) * radii
    ])
    
    # Initialize actor network parameters
    num_actions = params['actor']['num_actions']
    angles = np.linspace(0, 2*np.pi, num_actions, endpoint=False)
    params['actor']['directions'] = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Compute coupled weights
    angle_diffs = angles[:, np.newaxis] - angles
    w_min = params['actor']['w_min']
    w_max = params['actor']['w_max']
    xi_0 = params['actor']['xi_0']
    params['actor']['coupled_aw'] = (w_min/num_actions + 
                                   w_max * np.exp(xi_0 * np.cos(angle_diffs)) / 
                                   np.sum(np.exp(xi_0 * np.cos(angle_diffs)), axis=1)[:, np.newaxis])
    
    # Initialize weights
    params['actor']['init_weights'] = np.random.normal(
        params['actor']['init_weights_mean'],
        params['actor']['init_weights_std'],
        (num_pc, num_actions)
    )
    
    # Set critic time constants based on dt
    params['critic']['tau_m'] = dt - params['critic']['tau_m_offset']
    params['critic']['tau_s'] = dt - params['critic']['tau_s_offset']
    params['critic']['init_weights'] = np.zeros((num_pc, 1))
    
    # Clean up temporary parameters
    del params['actor']['w_min']
    del params['actor']['w_max']
    del params['actor']['xi_0']
    del params['actor']['init_weights_mean']
    del params['actor']['init_weights_std']
    del params['critic']['tau_m_offset']
    del params['critic']['tau_s_offset']
    del params['dt']
    
    return params

if __name__ == '__main__':
    # Initialize parameters and run experiment
    params = initialize_parameters()
    # Get features from params
    features = params.pop('features')  # Remove features from params dict and store separately
    experiment(params, features, 'results.h5')
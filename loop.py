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
from functions import calculate_place_cell_activity

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
    dv = params['critic']['nu'] * params['learning']['epsilon_pcc'] * (g - previous_g) / params['env']['dt']
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
           (dt/tau_s) * params['learning']['epsilon_pca'] * input_pc +
           (dt/tau_s) * input_noise)
    # plt.plot((dt/tau_s) * input_noise,'y')
    # plt.plot((dt/tau_s) * params['learning']['epsilon_pca'] * input_pc,'g')
    # plt.plot((params['actor']['epsilon_aa'] * dt/tau_s) * np.dot(params['actor']['coupled_aw'], activation_function(ua, params)),'r')
    # plt.plot(va * (1 - dt/tau_s),'b')
    #plt.plot(activation_function(ua, params),'r')
    #plt.plot(np.dot(params['actor']['coupled_aw'], activation_function(ua, params)),'g')
    #plt.plot(ua,'k')
    
    return dua, dva

def critic_activity_double_dyn(uc, vc, input_pc, params):
    """Update critic network activity using double dynamics."""
    dt = params['env']['dt']
    tau_m = params['critic']['tau_m']
    tau_s = params['critic']['tau_s']
    
    # Update membrane potential
    duc = (dt / tau_m) * (vc - uc) + uc
    
    # Update synaptic current
    dvc = vc * (1 - dt/tau_s) + (params['learning']['epsilon_pcc'] * dt/tau_s) * input_pc
    
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
                              td_error * params['learning']['epsilon_pca'] * 
                              np.outer(pc_activity, ua))
                    
                    cw += (params['learning']['critic_lr'] * params['env']['dt'] * 
                          td_error * params['critic']['nu'] * 
                          params['learning']['epsilon_pcc'] * 
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
                              td_error * params['learning']['epsilon_pca'] * 
                              np.outer(pc_activity, ua))
                    
                    cw += (params['learning']['critic_lr'] * params['env']['dt'] * 
                          td_error * params['critic']['nu'] * 
                          params['learning']['epsilon_pcc'] * 
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
    """Initialize all parameters for the experiment.
    
    Parameters are organized in categories:
    - Environment: Physical parameters of the arena and simulation
    - Place Cells: Parameters for place cell encoding
    - Actor: Parameters for the actor network
    - Critic: Parameters for the critic network
    - Reward: Parameters for reward computation
    - Learning: Learning rates and related parameters
    - Noise: Parameters for exploration noise
    """
    # Basic parameters
    dt = 0.1  # Time step (s)
    
    ###################
    # Environment Parameters
    ###################
    env_params = {
        'arena_radius': 100,    # Radius of circular arena (cm)
        'goal_radius': 5,       # Radius of goal zone (cm)
        'speed': 30,            # Movement speed (cm/s)
        'max_trial_time': 120,  # Maximum trial duration (s)
        'reward_window': 2,   # Time window for reward computation after finding goal (s)
        'dt': dt,               # Time step (s)
        # Starting positions (cm) - East, North, West, South
        'start_positions': [[95, 0], [0, 95], [-95, 0], [0, -95]],
        # Platform positions (cm)
        'platform_positions': [[30, 0], [0, 30], [-30, 0], [0, -30], 
                             [50, 50], [-50, 50], [50, -50], [-50, -50]]
    }
    
    ###################
    # Place Cell Parameters
    ###################
    num_pc = 500  # Number of place cells
    
    # Initialize place cell centers
    angles = np.random.uniform(0, 2*np.pi, num_pc)
    radii = np.sqrt(np.random.uniform(0, 1, num_pc)) * env_params['arena_radius']
    radii = np.sort(radii)[::-1]
    centres = np.vstack([
        np.cos(angles) * radii,
        np.sin(angles) * radii
    ])
    
    pc_params = {
        'num_cells': num_pc,      # Number of place cells
        'centres': centres,        # Place cell centers (cm)
        'sigma': 30,              # Place field width (cm)
        'amplitude': 1,           # Maximum firing rate (Hz)
        'epsilon_pc': 1         # Place cell input weight
    }
    
    ###################
    # Actor Parameters
    ###################
    num_actions = 180  # Number of action cells (one per 2 degrees)
    
    # Initialize coupled weights for actor network
    angles = np.linspace(0, 2*np.pi, num_actions, endpoint=False)
    directions = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Compute coupled weights
    w_min = -1  # inhibition
    w_max = 1   # excitation
    xi_0 = 1    # scale factor
    angle_diffs = angles[:, np.newaxis] - angles
    coupled_aw = (w_min/num_actions + 
                 w_max * np.exp(xi_0 * np.cos(angle_diffs)) / 
                 np.sum(np.exp(xi_0 * np.cos(angle_diffs)), axis=1)[:, np.newaxis])
    
    actor_params = {
        'num_actions': num_actions,  # Number of action cells
        'directions': directions,    # Action directions
        'coupled_aw': coupled_aw,    # Coupled weights matrix
        'tau_m': 0.15,              # Membrane time constant (s)
        'tau_s': 0.3,               # Synaptic time constant (s)
        'rho': 0.5,                 # Activation function gain
        'beta': 5,                  # Activation function steepness
        'h': 1,                     # Activation function threshold
        'epsilon_aa': 1,          # Action-to-action coupling weight
        'init_weights': np.random.normal(0.005, 0.0001, (num_pc, num_actions))  # Initial weights
    }
    
    ###################
    # Critic Parameters
    ###################
    critic_params = {
        'tau_m': dt - 0.001,        # Membrane time constant (s)
        'tau_s': dt - 0.001,        # Synaptic time constant (s)
        'nu': 0.99,                  # Value scaling factor
        'v0': 0.0,                  # Baseline value
        'init_weights': np.zeros((num_pc, 1))  # Initial weights
    }
    
    ###################
    # Reward Parameters
    ###################
    reward_params = {
        'goal_reward': 1.0,     # Reward for reaching goal
        'wall_penalty': -0.01,   # Penalty for hitting wall
        'tau_ra': 1.5,          # Fast reward time constant (s)
        'tau_rb': 1.1,          # Slow reward time constant (s)
        'tau_r': 5              # Reward discount time (s)
    }
    
    ###################
    # Learning Parameters
    ###################
    learning_params = {
        'actor_lr': 0.01,       # Actor learning rate
        'critic_lr': 0.1,       # Critic learning rate
        'epsilon_pcc': 0.1,     # Place cell to critic weight
        'epsilon_pca': 0.1      # Place cell to actor weight
    }
    
    ###################
    # Noise Parameters
    ###################
    noise_params = {
        'tau': 0.3,             # Noise time constant (s)
        'mu': 10,               # Noise center deviation
        'sigma': 1,             # Noise distribution width
        'rho': 2,               # Noise amplitude
        'width': 0.3            # Noise angular width
    }
    
    # Combine all parameters
    params = {
        'env': env_params,
        'pc': pc_params,
        'actor': actor_params,
        'critic': critic_params,
        'reward': reward_params,
        'learning': learning_params,
        'noise': noise_params
    }
    
    return params

def test_activation_function():
    """Test the activation function."""
    params = initialize_parameters()  # Get full parameters
    
    # Test 1: Check if activation is bounded by rho
    ua = np.random.randn(10)
    act = activation_function(ua, params)
    assert np.all(act >= 0), "Activation should be non-negative"
    assert np.all(act <= params['actor']['rho']), f"Activation should not exceed rho ({params['actor']['rho']})"
    
    # Test 2: Check if activation is monotonic
    ua1 = np.zeros(10)
    ua2 = ua1 + 1
    act1 = activation_function(ua1, params)
    act2 = activation_function(ua2, params)
    assert np.all(act2 >= act1), "Activation should be monotonically increasing"
    
    # Test 3: Check if activation is rho/2 at h (inflection point)
    ua = np.full(10, params['actor']['h'])
    act = activation_function(ua, params)
    assert np.allclose(act, params['actor']['rho']/2), "Activation should be rho/2 at h"
    
    print("All activation function tests passed!")

def test_reward_function():
    """Test the reward function."""
    params = initialize_parameters()
    
    # Test goal reward buildup
    ra = rb = 0
    for _ in range(10):  # Let reward build up
        ra, rb, re, found_goal = reward_function(0, 0, 0, 0, False, ra, rb, params)
    assert found_goal == True, "Should find goal when at goal location"
    assert re > 0, "Reward should be positive at goal after buildup"
    
    # Test wall penalty
    ra = rb = 0
    for _ in range(10):  # Let penalty build up
        ra, rb, re, found_goal = reward_function(50, 50, 0, 0, True, ra, rb, params)
    assert found_goal == False, "Should not find goal when hitting wall"
    assert re < 0, "Reward should be negative when hitting wall"
    
    # Test no reward away from goal
    ra = rb = 0
    for _ in range(10):
        ra, rb, re, found_goal = reward_function(50, 50, 0, 0, False, ra, rb, params)
    assert found_goal == False, "Should not find goal when away from it"
    assert abs(re) < 1e-6, "Reward should be close to zero away from goal"
    
    print("All reward function tests passed!")

def test_place_cells():
    """Test place cell activity."""
    params = initialize_parameters()
    
    # Test peak activity at center
    pos = np.array([0., 0.])
    pc_act = placecells(pos, params)
    assert np.all(pc_act >= 0), "Place cell activity should be non-negative"
    assert np.all(pc_act <= params['pc']['amplitude']), f"Place cell activity should not exceed maximum ({params['pc']['amplitude']})"
    
    # Test activity decay with distance from a specific place cell
    test_center = params['pc']['centres'][:, 0]  # Take first place cell center
    pos1 = test_center  # At center
    pos2 = test_center + np.array([30., 30.])  # 30√2 cm away from center
    pc_act1 = placecells(pos1, params)
    pc_act2 = placecells(pos2, params)
    assert pc_act1[0] > pc_act2[0], "Activity should decay with distance from place cell center"
    
    # Test if at least some place cells are active at any position
    random_pos = np.random.uniform(-80, 80, 2)
    pc_act = placecells(random_pos, params)
    assert np.any(pc_act > 0.1), "Some place cells should be active at any position"
    
    print("All place cell tests passed!")

def test_position_update():
    """Test position updates and wall collisions."""
    params = initialize_parameters()
    
    # Test straight movement
    pos = np.array([0., 0.])
    new_pos, hit_wall = update_position(pos, (0, np.array([1, 0])), params)  # Move east
    assert new_pos[0] > 0 and abs(new_pos[1]) < 1e-6, "Should move east"
    assert not hit_wall, "Should not hit wall in center"
    
    # Test wall collision
    pos = np.array([99., 0.])  # Very close to wall
    new_pos, hit_wall = update_position(pos, (0, np.array([1, 0])), params)  # Move east
    assert hit_wall, "Should hit wall when moving toward it"
    assert np.linalg.norm(new_pos) <= params['env']['arena_radius'], "Should not exceed arena radius"
    assert np.isclose(np.linalg.norm(new_pos), params['env']['arena_radius'], rtol=1e-5), "Should be exactly at arena radius after collision"
    
    # Test diagonal movement
    pos = np.array([0., 0.])
    new_pos, hit_wall = update_position(pos, (0, np.array([np.sqrt(2)/2, np.sqrt(2)/2])), params)  # Move northeast
    assert new_pos[0] > 0 and new_pos[1] > 0, "Should move northeast"
    assert abs(new_pos[0] - new_pos[1]) < 1e-6, "Should move at 45 degrees"
    
    # Test multiple steps
    pos = np.array([0., 0.])
    for _ in range(5):
        pos, hit_wall = update_position(pos, (0, np.array([1, 0])), params)  # Move east repeatedly
    assert pos[0] > 0 and abs(pos[1]) < 1e-6, "Should keep moving east"
    
    print("All position update tests passed!")

if __name__ == '__main__':
    # Run tests
    test_activation_function()
    test_reward_function()
    test_place_cells()
    test_position_update()
    
    print("\nAll tests passed! Starting experiment...\n")
    
    # Initialize parameters and run experiment
    params = initialize_parameters()
    features = {
        'num_rats': 20,
        'num_trials': 20
    }
    experiment(params, features, 'results.h5')
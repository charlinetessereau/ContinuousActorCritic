# Continuous Actor-Critic Learning with Place Cells

This repository implements a continuous actor-critic reinforcement learning model for spatial navigation using place cells. The model learns to navigate in a circular arena to find a hidden platform, similar to the Morris Water Maze task.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ContinuousActorCritic.git
cd ContinuousActorCritic
```

2. Install required packages:
```bash
pip install numpy scipy h5py matplotlib seaborn jupyter
```

## Project Structure

- `loop.py`: Core implementation of the actor-critic model
  - Place cell encoding
  - Actor network for action selection
  - Critic network for value estimation
  - Reward dynamics for goal finding
  
- `analysis.py`: Analysis and visualization tools
  - Learning curve plotting
  - Trajectory visualization
  - Parameter comparison tools

## Quick Start

1. Run a basic experiment:
```python
from loop import initialize_parameters, experiment
from analysis import plot_learning_curve

# Run experiment
params = initialize_parameters()
features = {'num_rats': 20, 'num_trials': 20}
experiment(params, features, 'results.h5')

# Plot results
plot_learning_curve('results.h5')
```

2. Or use the Jupyter notebook for interactive exploration:
```bash
jupyter notebook README.ipynb
```

## Model Parameters

The model's parameters are organized in categories:

### Environment
- Arena radius: 100 cm
- Goal radius: 5 cm
- Movement speed: 30 cm/s
- Trial duration: 120 s

### Place Cells
- Number: 500
- Field width: 30 cm
- Maximum firing rate: 1 Hz

### Actor Network
- Action cells: 180 (2° resolution)
- Learning rate: 0.01
- Noise parameters for exploration

### Critic Network
- Learning rate: 0.1
- Value scaling: 0.1
- Temporal discount: 4 s

### Reward
- Goal reward: 1.0
- Wall penalty: -0.1
- Reward time constants: 1.5s, 1.1s

## Results

The model learns to:
1. Navigate efficiently to the platform
2. Avoid walls
3. Develop direct trajectories to the goal
4. Adapt to different platform locations

## Contributing

Feel free to open issues or submit pull requests. Areas for improvement include:
- Additional analysis tools
- Parameter optimization
- Performance improvements
- Documentation enhancements 

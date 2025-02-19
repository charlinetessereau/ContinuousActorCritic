# Continuous Actor-Critic Learning with Place Cells

This repository implements a continuous actor-critic reinforcement learning model for spatial navigation using place cells, as described in Chapter 5 of my [PhD thesis](https://eprints.nottingham.ac.uk/67019/) ([Tessereau, 2021](https://eprints.nottingham.ac.uk/67019/)). This is a rate network implementation of the spiking version in [Frémaux et al., 2013](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003024). 

The model produces an agent which learns to navigate in a circular arena to find a goal location, similar to the Morris Water Maze task. 

## Brief descripton of the model: 
The position of the agent is encoded in a layer of units mimiking hippocampal place cells. This is given as a feedforward input to a critic cell and an actor network. The former is in charge of learning the value of the locations, and the latter of learning the policy, which determines which is the best direction to choose at every position step.  
As this is a continuous RL implementation, the critic and actor activities evolve accoridng to a second order ODE which mimics the characteristic membrane potential response to an electrical input. If you want to know more, check out the references or reach out to me! 

## Installation

1. Clone the repository:
```bash
git clone https://github.com/charlinetessereau/ContinuousActorCritic.git
cd ContinuousActorCritic
```

2. Either create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate actor-critic
```
3. Or, create a venv environment:

```bash
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

## Project Structure

- `main.py`: Core implementation of the actor-critic model
  - Place cell encoding
  - Actor network for action selection
  - Critic network for value estimation
  - Reward dynamics for goal finding
  
- `analysis.py`: Analysis and visualization tools
  - Learning curve plotting
  - Trajectory visualization
  - Parameter comparison tools

## Quick Start

1. Run the experiment:
```python
from main import initialize_parameters, experiment

# Initialize parameters and run experiment
params = initialize_parameters()
features = params.pop('features')  # Remove features from params dict
experiment(params, features, 'results.h5')
```

2. Visualize learning through latencies:
```python
from analysis.learning_plots import plot_learning_curve

# Plot learning curve
plot_learning_curve('results.h5', metric='latency', save_path=True)
```

We can verify that the agent learns by looking at the average time (number of steps) that they need to reach the goal with repeated exposure to the same goal location:

<img src="assets/latency.png" width="400">

This plots shows the average latency and standard deviation (error bars) to reach the goal accros 20 independent runs. We can see that it decreases with trials, which is a proxy for learning. 

3. Examine the value function:
```python
from analysis.value_plots import plot_value_comparison

# Plot value function after learning
plot_value_comparison('results.h5', rat_idx=0, trial_idx1=19, save_path=True)
```

<img src="assets/value_function.png" width="400">

We can see that the value function after learning peaks at the reward location. In this code, it was initialised at 0, such that all locations have a null value function before learning. 

4. Compare decision-making dynamics before and after learning:
```python
from analysis.visualize import setup_figure, setup_lines, load_trial_data
import matplotlib.animation as animation

# Setup visualization
fig, axes = setup_figure()
lines = setup_lines(axes)

# Generate early learning video
data = load_trial_data('results.h5', rat_index=0, trial_index=0)
anim = animation.FuncAnimation(fig, update_plot, frames=len(data['history_ua']), 
                             fargs=(data, lines, axes), interval=50)
anim.save('early_learning.mp4', writer='ffmpeg', fps=30)

# Generate late learning video
data = load_trial_data('results.h5', rat_index=0, trial_index=19)
anim = animation.FuncAnimation(fig, update_plot, frames=len(data['history_ua']), 
                             fargs=(data, lines, axes), interval=50)
anim.save('late_learning.mp4', writer='ffmpeg', fps=30)
```

<p align="center">
<img src="assets/early_learning.gif" width="400"> <img src="assets/late_learning.gif" width="400">
</p>

The videos illustrates the actor dynamics (left and top right hand side in polar coordinates) and the resulting trajectory (right hand side - bottom). The lines on the left hand side plot and the polar plot show the different components that feed the actor dynamics: the input from the place cells (learned via the TD error) and the noise input (varying with the degree of learning). The vertical bar and the compass bar in the polar plot shows the residual decision on direction. At every timestep, the agent (red dot in the bottom right video) moves according to this compass. 

# Key observations: 
- Before learning, the noise level is higher, this is an exploratory noise, which leads the agent to explore its environment. After the agent finds the platform of after a limiter number of timestep, the agent is palced on the platform to mimic real experimental conditions.
- After learning, the place cell input is stronger, as the weights have been shaped through TD error to reflect the direction which leads to locations with higher values. This leads to a more targetted trajectory towards the goal location.
- As the dynamics are quite complex, the trajectories often can get stuck in some local limit cyles, leading to a high standard deviation accross independent runs. Hyperparameters fit would help optimise the dymanics!
- Random exploration doesnt reflect biological conditions, and I have some ideas on how to improve exploration to make it more biologically realistic - reach out if you're interested in collaborating! 

Note: If videos don't play directly, you can find them in the `assets` folder:
- [Early Learning Video](assets/animation_2_0.mp4)
- [Late Learning Video](assets/animation_2_19.mp4)

## Model Parameters

The model's parameters can be adjusted in `params.yaml`. Key parameters include:

- **Environment**:
  - Arena radius: 100 cm
  - Goal radius: 5 cm
  - Movement speed: 30 cm/s
  - Trial duration: 120 s

- **Place Cells**:
  - Number of cells: 500
  - Field width (σ): 30 cm
  - Maximum firing rate: 1 Hz

- **Actor Network**:
  - Action cells: 180 (one per 2 degrees)
  - Learning rate: 0.01
  - Noise decay time: 0.3 s

- **Critic Network**:
  - Learning rate: 0.1
  - Value scaling: 0.99

Feel free to experiment with different values to explore their impact on learning dynamics.

## Results

The model learns to navigate efficiently to the platform.

## Contributing

Feel free to open issues or submit pull requests. Areas for improvement include:
- Additional analysis tools
- Parameter optimization
- Performance improvements
- Documentation enhancements

## References

1. Tessereau, Charline. "Reinforcement Learning Approaches to Rapid Hippocampal Place Learning." Diss. University of Nottingham, 2021.

2. Frémaux, Nicolas, Henning Sprekeler, and Wulfram Gerstner. "Reinforcement learning using a continuous time actor-critic framework with spiking neurons." PLoS computational biology 9.4 (2013): e1003024. 

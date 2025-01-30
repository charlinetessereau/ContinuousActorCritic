"""Main analysis script with high-level functions."""
from analysis.core import load_experiment_data
from analysis.trajectory_plots import plot_trajectory_comparison
from analysis.value_plots import plot_value_comparison
from analysis.learning_plots import plot_learning_curve

output_dir='/Users/ctessereau/Documents/ContinuousActorCriticResults'
def analyze_experiment(filename, output_dir=None):
    """Run complete analysis pipeline on experiment data."""
    # Load data
    parameters, features = load_experiment_data(filename)
    
    # Generate plots
    if output_dir:
        plot_trajectory_comparison(filename, save_path=f"{output_dir}/trajectories.png")
        plot_value_comparison(filename, save_path=f"{output_dir}/value_functions.png")
        plot_learning_curve(filename, save_path=f"{output_dir}/learning_curve.png")
    else:
        plot_trajectory_comparison(filename)
        plot_value_comparison(filename)
        plot_learning_curve(filename) 
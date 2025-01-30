"""Analysis package for water maze experiments."""
from .core import load_experiment_data
from .trajectory_plots import plot_trajectory_comparison
from .value_plots import plot_value_comparison
from .learning_plots import plot_learning_curve

__all__ = [
    'load_experiment_data',
    'plot_trajectory_comparison',
    'plot_value_comparison',
    'plot_learning_curve',
] 
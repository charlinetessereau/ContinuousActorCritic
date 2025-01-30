from analysis import plot_trajectory_comparison, plot_value_comparison

# For individual plots
plot_trajectory_comparison("experiment.h5")
plot_value_comparison("experiment.h5")

# Or for complete analysis
from analysis.main import analyze_experiment
analyze_experiment("experiment.h5", output_dir="results") 
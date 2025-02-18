from analysis import plot_trajectory_comparison, plot_value_comparison,plot_learning_curve

# For individual plots
plot_trajectory_comparison('results.h5', rat_idx=2, trial_idx1=0, trial_idx2=-1,save_path=True)
plot_value_comparison('results.h5', rat_idx=2, trial_idx1=0, trial_idx2=-1,save_path=True)
plot_learning_curve('results.h5', metric='latency', save_path=True)

# Or for complete analysis
from analysis.main import analyze_experiment
analyze_experiment("results.h5", output_dir="results") 
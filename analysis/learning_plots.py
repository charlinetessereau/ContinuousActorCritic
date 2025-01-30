"""Functions for plotting learning curves and performance metrics."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from analysis.core import load_experiment_data
import h5py

def plot_learning_curve(filename, metric='latency', save_path=None): # filename='results.h5' filename='results_timereward2.h5'
    """Plot learning curve showing performance metrics across trials.
    
    Args:
        filename: Path to the data file
        metric: Performance metric to plot ('latency' by default)
        save_path: Optional path to save the figure
    """
    sns.set_style("whitegrid")
    
    with h5py.File(filename, 'r') as f:
        results = f['results']
        num_rats = len(results.keys())
        num_trials = len(results[f'rat_0'].keys())
        
        # Collect data for all rats and trials
        data = np.zeros((num_rats, num_trials))
        for rat_idx in range(num_rats):
            rat_group = results[f'rat_{rat_idx}']
            for trial_idx in range(num_trials):
                trial_group = rat_group[f'trial_{trial_idx}']
                if metric == 'latency':
                    data[rat_idx, trial_idx] = trial_group.attrs['latency']
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
    
    # Compute statistics
    mean_values = np.mean(data, axis=0)
    sem_values = np.std(data, axis=0) / np.sqrt(data.shape[0])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot mean with error bars
    trials = np.arange(1, num_trials + 1)
    ax.plot(trials, mean_values, '-', color='darkgreen', linewidth=2, label=f'Mean {metric.title()}')
    ax.errorbar(trials, mean_values, yerr=sem_values, fmt='o', color='black', 
                capsize=5, capthick=1, elinewidth=1)
    
    # Customize plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Trial Number', fontsize=14)
    ax.set_ylabel(f'{metric.title()} (s)', fontsize=14)
    ax.set_title('Learning Performance', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path+f'/{metric}.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show() 
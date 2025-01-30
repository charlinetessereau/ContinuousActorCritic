"""Core functionality for data loading and processing."""
import h5py
import numpy as np
from functions import calculate_place_cell_activity
import os

def load_experiment_data(filename):
    """Load experiment data from HDF5 file.
    
    Args:
        filename (str): Path to the HDF5 file
        
    Returns:
        tuple: (parameters, features) dictionaries
    """
    parameters = {}
    features = {}
    
    try:
        with h5py.File(filename, 'r') as f:
            # Load parameters
            if 'parameters' in f:
                param_group = f['parameters']
                for category in param_group.keys():
                    parameters[category] = {}
                    for key, value in param_group[category].attrs.items():
                        parameters[category][key] = value
                    # Load any dataset values
                    for key in param_group[category].keys():
                        parameters[category][key] = param_group[category][key][:]
            else:
                print("Warning: No parameters group found in file")
                
            # Load features
            if 'features' in f:
                for key, value in f['features'].attrs.items():
                    features[key] = value
                
            # If no parameters/features found, try to infer from data structure
            if not parameters and not features:
                print("Warning: No explicit parameters or features found. Inferring from data structure...")
                rat_groups = [key for key in f.keys() if key.startswith('rat_')]
                features['num_rats'] = len(rat_groups)
                if rat_groups:
                    trial_groups = [key for key in f[rat_groups[0]].keys() if key.startswith('trial_')]
                    features['num_trials'] = len(trial_groups)
                
    except Exception as e:
        print(f"Error loading file {filename}: {str(e)}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Absolute path attempted: {os.path.abspath(filename)}")
        raise
    
    return parameters, features 
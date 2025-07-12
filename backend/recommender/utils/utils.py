import numpy as np
import json
from datetime import datetime
import os

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_experiment_results(model_info, metrics, filepath="experiment_results.json"):
    """
    Save experiment results to a JSON file
    
    Args:
        model_info: Dictionary containing model configuration
        metrics: Dictionary containing evaluation metrics
        filepath: Path to save the results
    """
    # create  results dictionary
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "configuration": model_info,
        "metrics": metrics
    }
    
    # only create directory if filepath contains a directory path
    directory = os.path.dirname(filepath)
    if directory:  # Check if directory is not empty
        os.makedirs(directory, exist_ok=True)
    
    # if the file exists, load existing results and append
    existing_results = []
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            try:
                existing_results = json.load(f)
                if not isinstance(existing_results, list):
                    existing_results = [existing_results]
            except json.JSONDecodeError:
                existing_results = []
    
    # append new results
    existing_results.append(results)
    
    # save the updated results
    with open(filepath, 'w') as f:
        json.dump(existing_results, f, indent=2, cls=NumpyEncoder)
    
    print(f"Experiment results saved to {filepath}")

def format_time(seconds):
    """
    Format time in seconds to a readable string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        return f"{seconds:.1f}s"
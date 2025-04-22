import wandb
import pandas as pd
import os
import argparse

# Define sweep path (adjust entity/project if needed)
SWEEP_PATH = "polargeese/PolarBERT-sweeps/2l72klhg"
OUTPUT_DIR = "tables" # Path relative to workspace root
OUTPUT_FILENAME = "sweep_results_pretrain_kaggle_350k_time_offset.csv"
METRIC_NAME = "val/full_loss" # The metric to retrieve

def fetch_sweep_data(sweep_path, metric_name):
    """Fetches run data (config and metric) for a given W&B sweep."""
    print(f"Fetching data for sweep: {sweep_path}")
    api = wandb.Api()
    try:
        sweep = api.sweep(sweep_path)
    except wandb.errors.CommError as e:
        print(f"Error fetching sweep: {e}")
        print(f"Please ensure the sweep path '{sweep_path}' is correct and you have permissions.")
        return None

    runs_data = []
    for run in sweep.runs:
        run_dict = {}
        # Get relevant config parameters (add more if needed)
        config_keys = [
            'max_epochs', 'mask_prob', 'logical_batch_size', 'max_lr', 
            'weight_decay', 'gradient_clip_val', 'one_minus_adam_beta1', 
            'one_minus_adam_beta2', 'adam_eps', 'pct_start', 'div_factor', 
            'final_div_factor' 
        ]
        for key in config_keys:
             run_dict[key] = run.config.get(key)
        
        # Get the summary metric
        summary = run.summary
        metric_value = summary.get(metric_name)
        # Handle cases where metric might be missing or nested (e.g., _step)
        if metric_value is None:
             # Try finding the metric without potential nesting like _step, _runtime etc.
             metric_value = run.summary._json_dict.get(metric_name)

        if metric_value is not None:
            run_dict[metric_name] = metric_value
            run_dict['run_id'] = run.id
            run_dict['run_name'] = run.name
            runs_data.append(run_dict)
        else:
             print(f"Warning: Metric '{metric_name}' not found for run {run.id} ({run.name}). Skipping run.")


    if not runs_data:
         print("No runs found with the specified metric for this sweep.")
         return None

    print(f"Successfully fetched data for {len(runs_data)} runs.")
    return pd.DataFrame(runs_data)

def save_results_to_csv(df, output_dir, output_filename):
    """Saves the DataFrame to a CSV file, creating the directory if needed."""
    # Construct path relative to workspace root
    # Assuming script is run from workspace root
    output_path = os.path.join(output_dir, output_filename)
    
    # Create output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Directory '{output_dir}' ensured.")
    except OSError as e:
        print(f"Error creating directory {output_dir}: {e}")
        return

    # Save DataFrame to CSV
    try:
        df.to_csv(output_path, index=False)
        print(f"Successfully saved sweep results to {output_path}")
    except Exception as e:
        print(f"Error saving DataFrame to CSV: {e}")

if __name__ == "__main__":
    
    # --- Configuration ---
    sweep_path = SWEEP_PATH
    output_dir = OUTPUT_DIR
    output_filename = OUTPUT_FILENAME
    metric_name = METRIC_NAME
    # --- End Configuration ---

    # Fetch data
    sweep_df = fetch_sweep_data(sweep_path, metric_name)

    # Save data if fetch was successful
    if sweep_df is not None and not sweep_df.empty:
        save_results_to_csv(sweep_df, output_dir, output_filename)
    else:
        print("No data fetched or DataFrame is empty. Skipping save.") 
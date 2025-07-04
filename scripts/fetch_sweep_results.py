import wandb
import pandas as pd
import os
import argparse
from typing import List

def fetch_sweep_data(sweep_path: str, metric_names: List[str]) -> pd.DataFrame:
    """Fetches run data (config and metrics) for a given W&B sweep."""
    print(f"Fetching data for sweep: {sweep_path}")
    print(f"Requested metrics: {metric_names}")
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
            if key in run.config:
                run_dict[key] = run.config[key]
        
        # Get the summary metrics
        summary = run.summary
        
        for metric_name in metric_names:
            metric_value = summary.get(metric_name)
            # Handle cases where metric might be missing or nested (e.g., _step)
            if metric_value is None:
                 # Try finding the metric without potential nesting like _step, _runtime etc.
                 metric_value = run.summary._json_dict.get(metric_name)

            if metric_value is not None:
                run_dict[metric_name] = metric_value
            else:
                # Set to None/NaN for missing metrics to maintain consistent columns
                run_dict[metric_name] = None
                print(f"Warning: Metric '{metric_name}' not found for run {run.id} ({run.name})")

        # Include all runs regardless of whether metrics were found
        run_dict['run_id'] = run.id
        run_dict['run_name'] = run.name
        runs_data.append(run_dict)

    if not runs_data:
         print("No runs found for this sweep.")
         return None

    print(f"Successfully fetched data for {len(runs_data)} runs.")
    return pd.DataFrame(runs_data)

def save_results_to_csv(df: pd.DataFrame, output_dir: str, output_filename: str):
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

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fetch and save W&B sweep results.')
    parser.add_argument('sweep_path', type=str,
                      help='Path to the W&B sweep (e.g., "polargeese/PolarBERT-sweeps/2l72klhg")')
    parser.add_argument('output_filename', type=str,
                      help='Name of the output CSV file')
    parser.add_argument('--output-dir', type=str, default="../../tables",
                      help='Directory to save the output file (default: "tables")')
    parser.add_argument('--metrics', type=str, nargs='+', default=["val/full_loss"],
                      help='Names of the metrics to retrieve (default: ["val/full_loss"]). '
                           'Can specify multiple metrics, e.g., --metrics val/full_loss val2/full_loss val/charge_loss')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Fetch data
    sweep_df = fetch_sweep_data(args.sweep_path, args.metrics)

    # Save data if fetch was successful
    if sweep_df is not None and not sweep_df.empty:
        save_results_to_csv(sweep_df, args.output_dir, args.output_filename)
    else:
        print("No data fetched or DataFrame is empty. Skipping save.") 
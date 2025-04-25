import pandas as pd
import argparse
import numpy as np
import logging
import yaml # Added for YAML handling
import os

# Define the columns assumed to be hyperparameters
HYPERPARAMETER_COLS = [
    'max_epochs', 'mask_prob', 'logical_batch_size', 'max_lr', 
    'weight_decay', 'gradient_clip_val', 'one_minus_adam_beta1', 
    'one_minus_adam_beta2', 'adam_eps', 'pct_start', 'div_factor', 
    'final_div_factor', 'random_time_offset'
]

# Parameters that should be integers in the final config
INTEGER_PARAMS = ['max_epochs', 'logical_batch_size']

def generate_tuned_config(input_config_path, output_config_path, median_hyperparams):
    """Loads an input YAML config, updates it with median hyperparameters, and saves to output path."""
    try:
        with open(input_config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Input config file not found at {input_config_path}")
        return False
    except yaml.YAMLError as e:
        print(f"Error parsing input config file {input_config_path}: {e}")
        return False
    except Exception as e:
        print(f"Error reading input config file {input_config_path}: {e}")
        return False

    print(f"\nUpdating config based on {input_config_path}...")
    
    # Verify required config sections exist
    if 'training' not in config:
        raise KeyError("Config file missing required 'training' section")
    if 'model' not in config:
        raise KeyError("Config file missing required 'model' section")
    if 'checkpoint' not in config['training']:
        raise KeyError("Config file missing required 'training.checkpoint' section")
        
    # Update hyperparameters
    param_map = {
        'one_minus_adam_beta1': 'adam_beta1',
        'one_minus_adam_beta2': 'adam_beta2',
    }
    value_transform = {
        'adam_beta1': lambda x: 1.0 - x,
        'adam_beta2': lambda x: 1.0 - x,
    }
    
    for param, median_val in median_hyperparams.items():
        target_param = param_map.get(param, param)
        final_val = value_transform.get(target_param, lambda x: x)(median_val)
        
        # Convert numpy types to standard Python types AND enforce integer types
        if target_param in INTEGER_PARAMS:
             if isinstance(final_val, (np.integer, np.floating)):
                  final_val = int(round(final_val)) # Round before casting to int
             elif isinstance(final_val, float):
                  final_val = int(round(final_val))
             else:
                  final_val = int(final_val) # Try casting directly
        elif isinstance(final_val, np.integer):
            final_val = int(final_val)
        elif isinstance(final_val, np.floating):
            final_val = float(final_val)
            
        config['training'][target_param] = final_val
        print(f"  Set training.{target_param} = {final_val:.6g}" if isinstance(final_val, float) else f"  Set training.{target_param} = {final_val}")

    # Update project
    original_project = config['training'].get('project', 'Unknown')
    new_project = original_project.replace('-sweeps', '-results') # Simple replacement logic
    if "results" not in new_project:
         new_project += "-results" # Append if not present
    config['training']['project'] = new_project
    print(f"  Set training.project = {new_project}")

    # Update model name
    original_model_name = config['model'].get('model_name', 'UnknownModel')
    new_model_name = original_model_name.replace('untuned', 'tuned')
    if "tuned" not in new_model_name:
         new_model_name += "-tuned"
    config['model']['model_name'] = new_model_name
    print(f"  Set model.model_name = {new_model_name}")

    # Update checkpoint settings
    original_dirpath = config['training']['checkpoint'].get('dirpath', 'checkpoints/unknown')
    new_dirpath = original_dirpath.replace('-sweep', '-tuned')
    if "/results/" not in new_dirpath:
         new_dirpath = new_dirpath.replace("checkpoints/", "checkpoints/results/")
    if "-tuned" not in new_dirpath:
         new_dirpath += "-tuned"
    
    # Check if directory already exists
    # if os.path.exists(new_dirpath):
    #     raise FileExistsError(f"Checkpoint directory already exists: {new_dirpath}")
         
    config['training']['checkpoint']['dirpath'] = new_dirpath
    config['training']['checkpoint']['save_top_k'] = 1
    config['training']['checkpoint']['save_last'] = True
    config['training']['checkpoint']['save_final'] = True
    print(f"  Set checkpoint.dirpath = {new_dirpath}")
    print(f"  Set checkpoint.save_top_k = 1")
    print(f"  Set checkpoint.save_last = True")
    print(f"  Set checkpoint.save_final = True")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_config_path)
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            print(f"Error creating directory for output config {output_dir}: {e}")
            return False
            
    # Save the updated config
    try:
        with open(output_config_path, 'w') as f:
            # Use a representer to handle potential remaining numpy types if needed,
            # but explicit conversion above should be sufficient.
            # yaml.add_representer(np.float64, lambda dumper, data: dumper.represent_float(float(data)))
            # yaml.add_representer(np.int64, lambda dumper, data: dumper.represent_int(int(data)))
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"\nSuccessfully generated tuned config: {output_config_path}")
        return True
    except Exception as e:
        print(f"Error writing tuned config file {output_config_path}: {e}")
        return False

def analyze_sweep(csv_path, metric_col, top_k=None, quantile=None, input_config=None, output_config=None):
    """Analyzes sweep results and optionally generates a tuned config file."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if metric_col not in df.columns:
        print(f"Error: Metric column '{metric_col}' not found in the CSV.")
        return

    # Drop rows with missing metric values
    df = df.dropna(subset=[metric_col])
    if df.empty:
        print("No runs with valid metric values found.")
        return

    # Sort by metric (ascending)
    df_sorted = df.sort_values(by=metric_col, ascending=True)

    if top_k is not None:
        print(f"Selecting top {top_k} runs based on '{metric_col}'.")
        top_runs_df = df_sorted.head(top_k)
        if len(top_runs_df) < top_k:
            logging.warning(f"Warning: Found only {len(top_runs_df)} runs, less than the requested {top_k}.")
    elif quantile is not None:
        loss_threshold = df_sorted[metric_col].quantile(quantile)
        print(f"Selecting runs in the top {quantile*100:.1f}% quantile (<= {loss_threshold:.4f}) based on '{metric_col}'.")
        top_runs_df = df_sorted[df_sorted[metric_col] <= loss_threshold]
    else:
        logging.error("Neither top_k nor quantile specified, using default top_k=5")
        top_runs_df = df_sorted.head(5)

    if top_runs_df.empty:
        print("No runs selected based on the criteria.")
        return

    print(f"Selected {len(top_runs_df)} runs for analysis.")

    # Calculate median hyperparameters
    median_hyperparams = {}
    available_hyperparams = [col for col in HYPERPARAMETER_COLS if col in top_runs_df.columns]
    
    for param in available_hyperparams:
        # Ensure column is numeric before calculating median
        if pd.api.types.is_numeric_dtype(top_runs_df[param]):
            median_hyperparams[param] = top_runs_df[param].median()
        else:
            # Handle non-numeric potentially (e.g., if constant, report it)
            unique_vals = top_runs_df[param].unique()
            if len(unique_vals) == 1:
                 median_hyperparams[param] = unique_vals[0]
            else:
                 logging.error(f"Error: Hyperparameter '{param}' is non-numeric and not constant. Cannot calculate median.")

    print("\nMedian Hyperparameters of Selected Runs:")
    if median_hyperparams:
        # Pretty print
        max_key_len = max(len(k) for k in median_hyperparams.keys())
        for key, value in median_hyperparams.items():
            if isinstance(value, float):
                print(f"  {key:<{max_key_len}} : {value:.6g}")
            else:
                print(f"  {key:<{max_key_len}} : {value}")
    else:
        print("  No median hyperparameters could be calculated.")
        return # Cannot generate config without hyperparameters

    # Generate tuned config if paths provided
    if input_config and output_config:
        # Call the generation function which now handles exceptions internally
        generate_tuned_config(input_config, output_config, median_hyperparams)
    else:
        print("\nSkipping tuned config generation: --input_config or --output_config not provided.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze W&B sweep results from a CSV file and optionally generate a tuned config.")
    parser.add_argument("--csv_file", type=str, required=True, 
                        help="Path to the sweep results CSV file.")
    parser.add_argument("--metric_col", type=str, default="val/full_loss", 
                        help="Name of the metric column to minimize.")
    parser.add_argument("--input_config", type=str, 
                        help="(Optional) Path to the input (untuned) YAML config file.")
    parser.add_argument("--output_config", type=str, 
                        help="(Optional) Path to save the generated tuned YAML config file. Requires --input_config.")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--top_k", type=int, 
                       help="Analyze the top K runs based on the metric.")
    group.add_argument("--quantile", type=float, 
                       help="Analyze runs within the top Q quantile based on the metric (e.g., 0.05 for top 5%%).")

    args = parser.parse_args()

    if args.output_config and not args.input_config:
         parser.error("--output_config requires --input_config.")

    top_k_to_use = args.top_k
    quantile_to_use = args.quantile
    if args.top_k is None and args.quantile is None:
        top_k_to_use = 5 # Default to top_k=5
        print("Neither --top_k nor --quantile specified, using default --top_k=5")

    analyze_sweep(args.csv_file, args.metric_col, 
                  top_k=top_k_to_use, quantile=quantile_to_use, 
                  input_config=args.input_config, output_config=args.output_config) 
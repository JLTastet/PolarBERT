#!/usr/bin/env python3
"""
Script to analyze dataset scaling experiments by processing hyperparameter sweeps.
For each sweep directory matching the pattern '2-{X}M_on_{Y}M-lr_scan':
1. Downloads sweep results from W&B
2. Selects the best checkpoint based on validation loss
3. Copies the best checkpoint to a 'best_checkpoint' subdirectory
4. Computes test loss for the best checkpoint
5. Generates a summary table
"""

import argparse
import os
import re
import shutil
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml

# Import functions from existing scripts
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fetch_sweep_results import fetch_sweep_data, save_results_to_csv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_sweep_path_from_slurm(slurm_file: Path) -> Optional[str]:
    """Extract W&B sweep path from slurm.sh file."""
    try:
        with open(slurm_file, 'r') as f:
            content = f.read()
        
        # Look for wandb agent command
        match = re.search(r'wandb agent.*?([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)', content)
        if match:
            return match.group(1)
        return None
    except Exception as e:
        logger.error(f"Error reading {slurm_file}: {e}")
        return None


def parse_sweep_directory_name(dir_name: str) -> Optional[Tuple[int, int]]:
    """Parse sweep directory name to extract fine-tuning and pretraining event counts."""
    match = re.match(r'2-(\d+)M_on_(\d+)M-lr_scan', dir_name)
    if match:
        finetune_events_M = int(match.group(1))
        pretrain_events_M = int(match.group(2))
        return finetune_events_M, pretrain_events_M
    return None


def get_finetune_events_from_config(config_file: Path) -> Optional[int]:
    """Extract training events from base_config.yaml."""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('data', {}).get('train_events')
    except Exception as e:
        logger.error(f"Error reading config {config_file}: {e}")
        return None


def find_best_checkpoint(sweep_results_df: pd.DataFrame, checkpoints_dir: Path) -> Optional[Tuple[str, str, float, float]]:
    """
    Find the best checkpoint based on validation loss.
    Returns: (run_name, run_id, max_lr, val_loss) or None if not found.
    """
    if sweep_results_df.empty or 'val/loss' not in sweep_results_df.columns:
        logger.error("No valid sweep results or missing val/loss column")
        return None
    
    # Find the run with minimum validation loss
    best_idx = sweep_results_df['val/loss'].idxmin()
    best_run = sweep_results_df.loc[best_idx]
    
    run_name = best_run['run_name']
    run_id = best_run['run_id']
    max_lr = best_run.get('max_lr', None)
    val_loss = best_run['val/loss']
    
    # Look for the checkpoint directory
    run_checkpoint_dir = checkpoints_dir / run_name
    if not run_checkpoint_dir.exists():
        logger.error(f"Checkpoint directory not found: {run_checkpoint_dir}")
        return None
    
    # Look for last.ckpt file
    checkpoint_file = run_checkpoint_dir / "last.ckpt"
    if not checkpoint_file.exists():
        logger.error(f"Checkpoint file not found: {checkpoint_file}")
        return None
    
    logger.info(f"Best run: {run_name} (ID: {run_id}, val_loss: {val_loss:.6f}, max_lr: {max_lr})")
    return run_name, run_id, max_lr, val_loss


def copy_best_checkpoint(sweep_dir: Path, run_name: str, run_id: str) -> Optional[Path]:
    """Copy the best checkpoint to best_checkpoint subdirectory."""
    checkpoints_dir = sweep_dir / "checkpoints"
    best_checkpoint_dir = sweep_dir / "best_checkpoint"
    
    source_dir = checkpoints_dir / run_name
    if not source_dir.exists():
        logger.error(f"Source checkpoint directory not found: {source_dir}")
        return None
    
    # Remove existing directory if it exists
    if best_checkpoint_dir.exists():
        shutil.rmtree(best_checkpoint_dir)
    
    try:
        shutil.copytree(source_dir, best_checkpoint_dir)
        logger.info(f"Copied best checkpoint to {best_checkpoint_dir}")
        
        # Create metadata file to track which run this is
        metadata = {
            'run_name': run_name,
            'run_id': run_id,
            'copied_at': str(pd.Timestamp.now())
        }
        with open(best_checkpoint_dir / "run_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return best_checkpoint_dir / "last.ckpt"
    except Exception as e:
        logger.error(f"Error copying checkpoint: {e}")
        return None


def compute_test_loss_for_checkpoint(config_file: Path, checkpoint_file: Path, test_dir: str, device: str = 'auto') -> Optional[Dict]:
    """Compute test loss for a checkpoint and return results."""
    try:
        # Import required modules locally to avoid import issues when script is not run from src/polarbert
        import torch
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from polarbert.utils.config import load_and_process_config
        from polarbert.utils.data import default_transform
        from polarbert.pretraining import MODEL_CLASSES
        from polarbert.finetuning import DirectionalHead
        from polarbert.icecube_dataset import IceCubeDataset
        from torch.utils.data import DataLoader
        import numpy as np
        from tqdm import tqdm
        
        # Setup device (use GPU if available, unless explicitly specified)
        if device == 'auto':
            torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            torch_device = torch.device(device if torch.cuda.is_available() and device.startswith('cuda') else 'cpu')
        logger.info(f"Using device: {torch_device}")
        
        # Load config
        config = load_and_process_config(str(config_file))
        
        # Load model (reimplemented from compute_test_loss.py)
        checkpoint = torch.load(str(checkpoint_file), map_location=torch_device, weights_only=True)
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Ensure config has required directional section for finetuning
        if 'directional' not in config.get('model', {}):
            config.setdefault('model', {})['directional'] = {'hidden_size': 1024}
            logger.warning("Config missing 'model.directional' section, using default hidden_size=1024")
        
        model = DirectionalHead(config)
        model.load_state_dict(state_dict)
        model.to(torch_device)
        model.eval()
        
        # Create test dataloader (reimplemented from compute_test_loss.py)
        target_transform = DirectionalHead.target_transform_kaggle
        
        full_dataset = IceCubeDataset(
            data_dir=test_dir,
            batch_size=1000,
            transform=default_transform,
            target_transform=target_transform
        )
        
        if full_dataset.num_events == 0:
            raise ValueError(f"Dataset at {test_dir} contains no events")
        
        test_loader = DataLoader(
            full_dataset,
            batch_size=None,
            num_workers=config['data'].get('num_workers', 0),
            pin_memory=config['data'].get('pin_memory', False),
            persistent_workers=config['data'].get('persistent_workers', False)
        )
        
        # Evaluate model (reimplemented from compute_test_loss.py)
        model.eval()
        losses = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating", leave=False):
                (x, l), (y, c) = batch
                x = {k: v.to(torch_device) if isinstance(v, torch.Tensor) else v for k, v in x.items()} if isinstance(x, dict) else x.to(torch_device)
                l = l.to(torch_device)
                y = y.to(torch_device)
                c = c.to(torch_device)
                
                loss = model.shared_step(((x, l), (y, c)), 0)
                losses.append(loss.item())
        
        results = {
            'test/loss': np.mean(losses),
            'uncertainty/test/loss': np.std(losses) / np.sqrt(len(losses) - 1) if len(losses) > 1 else 0.0,
        }
        
        # Save results to JSON file next to checkpoint
        results_path = checkpoint_file.parent / "test_loss.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Test loss: {results['test/loss']:.6f} ± {results['uncertainty/test/loss']:.6f}")
        return results
        
    except Exception as e:
        logger.error(f"Error computing test loss: {e}")
        return None


def process_sweep_directory(sweep_dir: Path, test_dir: str, device: str = 'auto', skip_unfinished: bool = True, force_reevaluate: bool = False) -> Optional[Dict]:
    """
    Process a single sweep directory.
    Returns summary data or None if processing failed.
    """
    logger.info(f"Processing sweep directory: {sweep_dir.name}")
    
    # Parse directory name
    parsed = parse_sweep_directory_name(sweep_dir.name)
    if not parsed:
        logger.warning(f"Skipping directory {sweep_dir.name} - doesn't match expected pattern")
        return None
    
    finetune_events_M, pretrain_events_M = parsed
    
    # Check required files
    slurm_file = sweep_dir / "slurm.sh"
    config_file = sweep_dir / "base_config.yaml"
    checkpoints_dir = sweep_dir / "checkpoints"
    
    if not all(f.exists() for f in [slurm_file, config_file]):
        logger.error(f"Missing required files in {sweep_dir}")
        return None
    
    if not checkpoints_dir.exists():
        logger.warning(f"No checkpoints directory in {sweep_dir} - sweep may not be complete")
        return None
    
    # Extract sweep path
    sweep_path = extract_sweep_path_from_slurm(slurm_file)
    if not sweep_path:
        logger.error(f"Could not extract sweep path from {slurm_file}")
        return None
    
    logger.info(f"Found sweep path: {sweep_path}")
    
    # Check if sweep is finished and download results
    sweep_results_file = sweep_dir / "sweep_results.csv"
    sweep_state = "UNKNOWN"  # Default value
    try:
        import wandb
        api = wandb.Api()
        sweep = api.sweep(sweep_path)
        
        # Check if sweep is finished
        sweep_state = sweep.state
        logger.info(f"Sweep state: {sweep_state}")
        
        if sweep_state.upper() != 'FINISHED' and skip_unfinished:
            logger.warning(f"Sweep {sweep_path} is not finished (state: {sweep_state}). Skipping test evaluation.")
            # Still download current results for inspection, but don't evaluate on test set
            sweep_df = fetch_sweep_data(sweep_path, ["val/loss"])
            if sweep_df is not None and not sweep_df.empty:
                sweep_df.to_csv(sweep_results_file, index=False)
                logger.info(f"Saved current sweep results to {sweep_results_file}")
            return None
        elif sweep_state.upper() != 'FINISHED':
            logger.warning(f"Sweep {sweep_path} is not finished (state: {sweep_state}), but proceeding with test evaluation as requested.")
        else:
            logger.info(f"Sweep {sweep_path} is finished. Proceeding with full processing including test evaluation.")
        
        # Sweep is finished (or forced), proceed with full processing
        sweep_df = fetch_sweep_data(sweep_path, ["val/loss"])
        if sweep_df is None or sweep_df.empty:
            logger.error(f"No sweep data retrieved for {sweep_path}")
            return None
        
        # Save results
        sweep_df.to_csv(sweep_results_file, index=False)
        logger.info(f"Saved sweep results to {sweep_results_file}")
        
    except Exception as e:
        logger.error(f"Error fetching sweep data: {e}")
        return None
    
    # Find best checkpoint
    best_result = find_best_checkpoint(sweep_df, checkpoints_dir)
    if not best_result:
        logger.error(f"Could not find best checkpoint for {sweep_dir}")
        # For unfinished sweeps, provide more context
        if sweep_state.upper() != 'FINISHED':
            logger.info(f"This is an unfinished sweep (state: {sweep_state}). The best run may not have completed yet or may not have checkpoints available.")
        return None
    
    run_name, run_id, max_lr, val_loss = best_result
    
    # Get actual finetune events from config
    actual_finetune_events = get_finetune_events_from_config(config_file)
    if actual_finetune_events:
        actual_finetune_events_M = actual_finetune_events / 1_000_000
    else:
        actual_finetune_events_M = finetune_events_M  # fallback to parsed value
    
    # Check if test loss has already been computed for this specific run
    best_checkpoint_dir = sweep_dir / "best_checkpoint"
    test_loss_file = best_checkpoint_dir / "test_loss.json"
    metadata_file = best_checkpoint_dir / "run_metadata.json"
    
    # Check if we already have results for this specific run
    has_existing_results = False
    if test_loss_file.exists() and metadata_file.exists() and not force_reevaluate:
        try:
            # Check if the existing results are for the same run
            with open(metadata_file, 'r') as f:
                existing_metadata = json.load(f)
            
            if existing_metadata.get('run_id') == run_id:
                # Same run, load existing results
                with open(test_loss_file, 'r') as f:
                    test_results = json.load(f)
                test_loss = test_results['test/loss']
                test_loss_uncertainty = test_results['uncertainty/test/loss']
                logger.info(f"Test loss results already exist for run {run_id}. Loading existing results.")
                logger.info(f"Loaded existing test loss: {test_loss:.6f} ± {test_loss_uncertainty:.6f}")
                has_existing_results = True
            else:
                logger.info(f"Existing results are for different run ({existing_metadata.get('run_id')} vs {run_id}). Will re-compute.")
                has_existing_results = False
        except Exception as e:
            logger.warning(f"Could not load existing metadata or test results: {e}. Will re-compute.")
            has_existing_results = False
    
    if not has_existing_results:
        # Copy best checkpoint (this will overwrite existing best_checkpoint directory)
        best_checkpoint_file = copy_best_checkpoint(sweep_dir, run_name, run_id)
        if not best_checkpoint_file:
            logger.error(f"Could not copy best checkpoint for {sweep_dir}")
            return None
        
        if force_reevaluate:
            logger.info(f"Forcing re-evaluation of test loss.")
        
        # Compute test loss
        test_results = compute_test_loss_for_checkpoint(config_file, best_checkpoint_file, test_dir, device)
        if not test_results:
            logger.error(f"Could not compute test loss for {sweep_dir}")
            return None
        
        test_loss = test_results['test/loss']
        test_loss_uncertainty = test_results['uncertainty/test/loss']
    
    # Return summary data
    return {
        'sweep_name': sweep_dir.name,
        'finetune_events_M': actual_finetune_events_M,
        'pretrain_events_M': pretrain_events_M,
        'run_name': run_name,
        'run_id': run_id,
        'max_lr': max_lr,
        'val_loss': val_loss,
        'test_loss': test_loss,
        'test_loss_uncertainty': test_loss_uncertainty,
        'sweep_state': sweep_state
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze dataset scaling experiments')
    parser.add_argument('--experiment-dir', type=str, required=True,
                       help='Path to experiment directory containing sweep subdirectories')
    parser.add_argument('--test-dir', type=str, required=True,
                       help='Path to test dataset directory')
    parser.add_argument('--pattern', type=str, default=r'2-\d+M_on_\d+M-lr_scan',
                       help='Regex pattern for sweep directory names (default: 2-\\d+M_on_\\d+M-lr_scan)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use for evaluation (default: auto - use GPU if available)')
    parser.add_argument('--skip-unfinished', action='store_true', default=True,
                       help='Skip test evaluation for unfinished sweeps (default: True)')
    parser.add_argument('--include-unfinished', action='store_true',
                       help='Include test evaluation for unfinished sweeps (overrides --skip-unfinished)')
    parser.add_argument('--evaluate-unfinished', action='store_true',
                       help='Evaluate best checkpoints from unfinished sweeps (alias for --include-unfinished)')
    parser.add_argument('--force-reevaluate', action='store_true',
                       help='Force re-evaluation of test loss even if results already exist')
    
    args = parser.parse_args()
    
    # Handle the skip/include unfinished logic
    skip_unfinished = args.skip_unfinished and not args.include_unfinished and not args.evaluate_unfinished
    
    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        logger.error(f"Experiment directory not found: {experiment_dir}")
        return
    
    # Find all matching sweep directories
    pattern = re.compile(args.pattern)
    sweep_dirs = [d for d in experiment_dir.iterdir() 
                  if d.is_dir() and pattern.match(d.name)]
    
    if not sweep_dirs:
        logger.error(f"No directories matching pattern '{args.pattern}' found in {experiment_dir}")
        return
    
    logger.info(f"Found {len(sweep_dirs)} sweep directories to process")
    
    # Process each sweep directory
    summary_data = []
    for sweep_dir in sorted(sweep_dirs):
        try:
            result = process_sweep_directory(sweep_dir, args.test_dir, args.device, skip_unfinished, args.force_reevaluate)
            if result:
                summary_data.append(result)
            else:
                logger.warning(f"Failed to process {sweep_dir.name}")
        except Exception as e:
            logger.error(f"Error processing {sweep_dir.name}: {e}")
            continue
    
    if not summary_data:
        logger.error("No sweep directories were successfully processed")
        return
    
    # Create summary table
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values(['pretrain_events_M', 'finetune_events_M'])
    
    # Save summary table
    summary_file = experiment_dir / "dataset_scaling_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Saved summary table to {summary_file}")
    
    # Print summary
    print(f"\nDataset Scaling Summary:")
    print(f"Processed {len(summary_data)} sweeps successfully")
    print(f"Summary saved to: {summary_file}")
    
    # Display key results
    print(f"\nKey Results:")
    for _, row in summary_df.iterrows():
        state_indicator = "" if row['sweep_state'] == 'FINISHED' else f" [{row['sweep_state']}]"
        print(f"{row['sweep_name']}{state_indicator}: val_loss={row['val_loss']:.6f}, test_loss={row['test_loss']:.6f}±{row['test_loss_uncertainty']:.6f}")
    
    # Show summary of sweep states
    if 'sweep_state' in summary_df.columns:
        state_counts = summary_df['sweep_state'].value_counts()
        print(f"\nSweep States Summary:")
        for state, count in state_counts.items():
            print(f"  {state}: {count} sweeps")


if __name__ == '__main__':
    main()

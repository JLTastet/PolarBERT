#!/usr/bin/env python3
"""
Script to compute test loss for PolarBERT models (pretraining or finetuning).
Supports only the Kaggle dataset for now.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import json

from polarbert.utils.config import load_and_process_config
from polarbert.utils.data import default_transform
from polarbert.pretraining import MODEL_CLASSES
from polarbert.finetuning import DirectionalHead
from polarbert.icecube_dataset import IceCubeDataset
from torch.utils.data import DataLoader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path, device):
    """Load checkpoint and return state dict."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    return checkpoint.get('state_dict', checkpoint)


def load_model(config, checkpoint_path, device, model_type):
    """Load model based on type."""
    state_dict = load_checkpoint(checkpoint_path, device)
    
    if model_type == 'pretraining':
        # Determine model class from config or default to 'flash'
        arch_type = config.get('model', {}).get('type', 'flash')
        if arch_type not in MODEL_CLASSES:
            logger.warning(f"Unknown model type '{arch_type}', defaulting to 'flash'")
            arch_type = 'flash'
        
        ModelClass, _ = MODEL_CLASSES[arch_type]
        model = ModelClass(config)
        logger.info(f"Loaded {arch_type} pretraining model")
    else:  # finetuning
        # Ensure config has required directional section for finetuning
        if 'directional' not in config.get('model', {}):
            config.setdefault('model', {})['directional'] = {'hidden_size': 1024}
            logger.warning("Config missing 'model.directional' section, using default hidden_size=1024")
        
        model = DirectionalHead(config)
        logger.info("Loaded finetuning model")
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def move_to_device(data, device):
    """Recursively move data to device, handling dicts and tensors."""
    if isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data


def evaluate_model(model, dataloader, device, model_type):
    """
    Evaluate model on the given dataloader.
    Returns loss results dict with means and errors.
    """
    model.eval()
    if model_type == 'pretraining':
        dom_losses = []
        charge_losses = []
        full_losses = []
    else:
        losses = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            # Move batch to device
            (x, l), (y, c) = batch
            x = move_to_device(x, device)
            l = l.to(device)
            y = y.to(device)
            c = c.to(device)
            
            # Compute loss based on model type
            if model_type == 'pretraining':
                dom_loss, charge_loss = model.shared_step(((x, l), (y, c)))
                full_loss = dom_loss + model.lambda_charge * charge_loss
                dom_losses.append(dom_loss.item())
                charge_losses.append(charge_loss.item())
                full_losses.append(full_loss.item())
            else:  # finetuning
                loss = model.shared_step(((x, l), (y, c)), 0)
                losses.append(loss.item())
    
    if model_type == 'pretraining':
        results = {
            'test/dom_loss': np.mean(dom_losses),
            'test/charge_loss': np.mean(charge_losses),
            'test/full_loss': np.mean(full_losses),
            'uncertainty/test/dom_loss': np.std(dom_losses) / np.sqrt(len(dom_losses) - 1) if len(dom_losses) > 1 else 0.0,
            'uncertainty/test/charge_loss': np.std(charge_losses) / np.sqrt(len(charge_losses) - 1) if len(charge_losses) > 1 else 0.0,
            'uncertainty/test/full_loss': np.std(full_losses) / np.sqrt(len(full_losses) - 1) if len(full_losses) > 1 else 0.0,
        }
    else:  # finetuning
        results = {
            'test/loss': np.mean(losses),
            'uncertainty/test/loss': np.std(losses) / np.sqrt(len(losses) - 1) if len(losses) > 1 else 0.0,
        }
    
    return results


def create_test_dataloader(test_dir, config, batch_size, test_events=None):
    """Create a dataloader for the test dataset."""
    # Use Kaggle target transform (works for both pretraining and finetuning)
    target_transform = DirectionalHead.target_transform_kaggle
    
    # Create a full dataset to check the total number of events
    full_dataset = IceCubeDataset(
        data_dir=test_dir,
        batch_size=batch_size,
        transform=default_transform,
        target_transform=target_transform
    )
    total_events = full_dataset.num_events
    
    # Check for empty dataset
    if total_events == 0:
        raise ValueError(f"Dataset at {test_dir} contains no events")
    
    # Determine actual number of events to use
    if test_events is not None:
        actual_events = min(test_events, total_events)
        if test_events > total_events:
            logger.warning(f"Requested {test_events} test events, but dataset only has {total_events} events. Using all {total_events} events.")
        else:
            logger.info(f"Using {actual_events} out of {total_events} available test events.")
        
        # Create sliced dataset using the actual number of events
        test_dataset = full_dataset.slice(0, actual_events)
    else:
        # Use full dataset
        test_dataset = full_dataset
        logger.info(f"Using all {total_events} available test events.")
    
    # Create DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=None,  # Dataset handles batching internally
        num_workers=config['data'].get('num_workers', 0),
        pin_memory=config['data'].get('pin_memory', False),
        persistent_workers=config['data'].get('persistent_workers', False)
    )
    
    return test_loader


def main():
    parser = argparse.ArgumentParser(description='Compute test loss for PolarBERT models')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to model config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test-dir', type=str, required=True,
                       help='Path to test dataset directory')
    parser.add_argument('--model-type', type=str, required=True, 
                       choices=['pretraining', 'finetuning'],
                       help='Model type: pretraining or finetuning')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size for evaluation (default: 1000)')
    parser.add_argument('--test-events', type=int, default=None,
                       help='Maximum number of test events to use (default: use all)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Validate paths
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    test_dir = Path(args.test_dir)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    # Load config
    logger.info(f"Loading config from {config_path}")
    config = load_and_process_config(str(config_path))
    
    # Load model
    logger.info(f"Loading {args.model_type} model from {checkpoint_path}")
    model = load_model(config, str(checkpoint_path), device, args.model_type)
    
    # Create test dataloader
    logger.info(f"Creating test dataloader from {test_dir}")
    test_loader = create_test_dataloader(str(test_dir), config, args.batch_size, args.test_events)
    
    # Evaluate model
    logger.info("Computing test loss...")
    results = evaluate_model(model, test_loader, device, args.model_type)
    
    # Save results to JSON file next to checkpoint
    results_path = checkpoint_path.parent / "test_loss.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved test loss results to {results_path}")
    
    # Print results
    print(f"\nTest Loss Results:")
    print(f"Model: {config.get('model', {}).get('model_name', 'unknown')} ({args.model_type})")
    
    if args.model_type == 'pretraining':
        print(f"DOM Loss: {results['test/dom_loss']:.6f} ± {results['uncertainty/test/dom_loss']:.6f}")
        print(f"Charge Loss: {results['test/charge_loss']:.6f} ± {results['uncertainty/test/charge_loss']:.6f}")
        print(f"Full Loss: {results['test/full_loss']:.6f} ± {results['uncertainty/test/full_loss']:.6f}")
    else:
        print(f"Loss: {results['test/loss']:.6f} ± {results['uncertainty/test/loss']:.6f}")
    
    print(f"Results saved to: {results_path}")


if __name__ == '__main__':
    main()

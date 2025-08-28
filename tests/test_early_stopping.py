#!/usr/bin/env python3
"""
test_early_stopping.py
=======================
Simple test script to verify early stopping functionality.
This script creates a mock training scenario to test early stopping logic.
"""

import torch
import numpy as np
from types import SimpleNamespace

# Mock the trainer's early stopping functionality for testing
class MockTrainer:
    def __init__(self, patience=5, min_delta=0.01, mode='max'):
        # Early stopping parameters
        self.early_stopping_patience = patience
        self.early_stopping_min_delta = min_delta
        self.early_stopping_mode = mode
        
        # Initialize early stopping tracking variables
        self.best_metric = None
        self.patience_counter = 0
        self.early_stopped = False

    def _check_early_stopping(self, val_metric):
        """
        Check if early stopping criteria are met.
        
        Parameters
        ----------
        val_metric : float
            The current validation metric value.
            
        Returns
        -------
        bool
            True if training should stop, False otherwise.
        """
        if self.early_stopping_patience is None:
            return False
            
        if self.best_metric is None:
            self.best_metric = val_metric
            self.patience_counter = 0
            return False
            
        # Check if current metric is better than best
        improved = False
        if self.early_stopping_mode == 'max':
            # For metrics like accuracy, AUC, etc. where higher is better
            improved = val_metric > (self.best_metric + self.early_stopping_min_delta)
        else:  # mode == 'min'
            # For metrics like loss where lower is better
            improved = val_metric < (self.best_metric - self.early_stopping_min_delta)
            
        if improved:
            self.best_metric = val_metric
            self.patience_counter = 0
            print(f"Validation metric improved to {val_metric:.6f}")
        else:
            self.patience_counter += 1
            print(f"No improvement in validation metric. Patience: {self.patience_counter}/{self.early_stopping_patience}")
            
        if self.patience_counter >= self.early_stopping_patience:
            print(f"Early stopping triggered after {self.patience_counter} epochs without improvement")
            return True
            
        return False

def simulate_training_with_early_stopping():
    """Simulate a training run that should trigger early stopping."""
    print("=== Test 1: Early stopping should trigger (max mode) ===")
    
    # Create mock trainer with early stopping
    trainer = MockTrainer(patience=3, min_delta=0.01, mode='max')
    
    # Simulate validation metrics that improve initially then plateau
    mock_metrics = [0.75, 0.80, 0.85, 0.86, 0.855, 0.854, 0.853, 0.852]
    
    for epoch, metric in enumerate(mock_metrics, 1):
        print(f"\nEpoch {epoch}: Validation metric = {metric:.3f}")
        should_stop = trainer._check_early_stopping(metric)
        
        if should_stop:
            print(f"Training stopped early at epoch {epoch}")
            break
    else:
        print("Training completed without early stopping")
    
    print(f"Best metric achieved: {trainer.best_metric:.6f}")

def simulate_training_without_early_stopping():
    """Simulate a training run that should NOT trigger early stopping."""
    print("\n\n=== Test 2: Early stopping should NOT trigger (continuous improvement) ===")
    
    # Create mock trainer with early stopping
    trainer = MockTrainer(patience=3, min_delta=0.005, mode='max')
    
    # Simulate validation metrics that keep improving
    mock_metrics = [0.70, 0.73, 0.76, 0.80, 0.83, 0.85, 0.87, 0.89]
    
    for epoch, metric in enumerate(mock_metrics, 1):
        print(f"\nEpoch {epoch}: Validation metric = {metric:.3f}")
        should_stop = trainer._check_early_stopping(metric)
        
        if should_stop:
            print(f"Training stopped early at epoch {epoch}")
            break
    else:
        print("Training completed without early stopping")
    
    print(f"Best metric achieved: {trainer.best_metric:.6f}")

def simulate_loss_based_early_stopping():
    """Simulate early stopping with loss (min mode)."""
    print("\n\n=== Test 3: Early stopping with loss metric (min mode) ===")
    
    # Create mock trainer for loss minimization
    trainer = MockTrainer(patience=4, min_delta=0.01, mode='min')
    
    # Simulate loss that decreases then plateaus
    mock_losses = [1.2, 0.8, 0.5, 0.3, 0.29, 0.295, 0.298, 0.301]
    
    for epoch, loss in enumerate(mock_losses, 1):
        print(f"\nEpoch {epoch}: Validation loss = {loss:.3f}")
        should_stop = trainer._check_early_stopping(loss)
        
        if should_stop:
            print(f"Training stopped early at epoch {epoch}")
            break
    else:
        print("Training completed without early stopping")
    
    print(f"Best loss achieved: {trainer.best_metric:.6f}")

if __name__ == "__main__":
    print("Testing Early Stopping Functionality")
    print("=" * 50)
    
    # Run different test scenarios
    simulate_training_with_early_stopping()
    simulate_training_without_early_stopping()
    simulate_loss_based_early_stopping()
    
    print("\n" + "=" * 50)
    print("Early stopping tests completed!")
    print("\nTo use early stopping in your BEND training:")
    print("1. Add early stopping parameters to your config YAML file")
    print("2. Set early_stopping_patience to desired number of epochs")
    print("3. Set early_stopping_mode to 'max' for metrics or 'min' for loss")
    print("4. Run training normally - early stopping will activate automatically")

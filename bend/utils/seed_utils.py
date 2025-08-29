"""
seed_utils.py
=============
Utility functions for setting random seeds to ensure reproducibility.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 0):
    """
    Set random seeds for reproducibility across all libraries and frameworks.
    
    This function sets seeds for:
    - Python's built-in random module
    - NumPy
    - PyTorch (CPU and GPU)
    - CUDA operations
    
    Parameters
    ----------
    seed : int, optional
        The random seed to use. Default is 0.
    """
    # Set Python's built-in random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch seeds
    torch.manual_seed(seed)
    
    # Set CUDA seeds for reproducibility on GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        
        # Additional CUDA settings for reproducibility
        # These settings may reduce performance but increase reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variable for CUDA operations
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed set to: {seed}")


def set_seed_worker(worker_id: int, base_seed: int = 0):
    """
    Set random seed for DataLoader workers to ensure reproducible data loading.
    
    This function is designed to be used with PyTorch DataLoader's worker_init_fn.
    Each worker gets a unique but deterministic seed based on the worker_id.
    
    Parameters
    ----------
    worker_id : int
        The worker ID provided by PyTorch DataLoader
    base_seed : int, optional
        The base random seed to use. Default is 0.
    """
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

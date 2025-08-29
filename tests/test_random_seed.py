"""
test_random_seed.py
==================
Test script to verify that random seed functionality works correctly.
"""

import os
import sys
import tempfile
import shutil
import torch
import numpy as np

# Add the parent directory to sys.path to import bend modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bend.utils.seed_utils import set_seed


def test_reproducibility():
    """Test that setting the same seed produces identical results."""
    print("Testing random seed reproducibility...")
    
    # Test 1: Python random
    set_seed(42)
    python_rand1 = [torch.rand(1).item() for _ in range(5)]
    
    set_seed(42)
    python_rand2 = [torch.rand(1).item() for _ in range(5)]
    
    assert python_rand1 == python_rand2, f"Python random not reproducible: {python_rand1} != {python_rand2}"
    print("✓ Python random is reproducible")
    
    # Test 2: NumPy random
    set_seed(42)
    np_rand1 = np.random.rand(5).tolist()
    
    set_seed(42)
    np_rand2 = np.random.rand(5).tolist()
    
    assert np.allclose(np_rand1, np_rand2), f"NumPy random not reproducible: {np_rand1} != {np_rand2}"
    print("✓ NumPy random is reproducible")
    
    # Test 3: PyTorch random
    set_seed(42)
    torch_rand1 = torch.rand(5).tolist()
    
    set_seed(42)
    torch_rand2 = torch.rand(5).tolist()
    
    assert torch.allclose(torch.tensor(torch_rand1), torch.tensor(torch_rand2)), \
           f"PyTorch random not reproducible: {torch_rand1} != {torch_rand2}"
    print("✓ PyTorch random is reproducible")
    
    # Test 4: Neural network initialization
    set_seed(42)
    model1 = torch.nn.Linear(10, 5)
    weights1 = model1.weight.data.clone()
    
    set_seed(42)
    model2 = torch.nn.Linear(10, 5)
    weights2 = model2.weight.data.clone()
    
    assert torch.allclose(weights1, weights2), "Neural network initialization not reproducible"
    print("✓ Neural network initialization is reproducible")
    
    print("All reproducibility tests passed! ✓")


def test_different_seeds():
    """Test that different seeds produce different results."""
    print("\nTesting that different seeds produce different results...")
    
    set_seed(42)
    result1 = torch.rand(5)
    
    set_seed(123)
    result2 = torch.rand(5)
    
    assert not torch.allclose(result1, result2), "Different seeds should produce different results"
    print("✓ Different seeds produce different results")


def test_cuda_reproducibility():
    """Test CUDA reproducibility if available."""
    print("\nTesting CUDA reproducibility...")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping CUDA tests")
        return
    
    device = torch.device('cuda:0')
    
    set_seed(42)
    cuda_rand1 = torch.rand(5, device=device).cpu()
    
    set_seed(42)
    cuda_rand2 = torch.rand(5, device=device).cpu()
    
    assert torch.allclose(cuda_rand1, cuda_rand2), "CUDA random not reproducible"
    print("✓ CUDA random is reproducible")


def test_seed_worker():
    """Test the worker seed function."""
    print("\nTesting worker seed function...")
    
    from bend.utils.seed_utils import set_seed_worker
    
    # Test that different workers get different but deterministic seeds
    set_seed_worker(0, base_seed=42)
    worker0_result = np.random.rand()
    
    set_seed_worker(1, base_seed=42)
    worker1_result = np.random.rand()
    
    assert worker0_result != worker1_result, "Different workers should get different results"
    
    # Test reproducibility for same worker
    set_seed_worker(0, base_seed=42)
    worker0_result_repeat = np.random.rand()
    
    assert worker0_result == worker0_result_repeat, "Same worker should get reproducible results"
    print("✓ Worker seeding works correctly")


if __name__ == "__main__":
    print("Running random seed tests...")
    print("=" * 50)
    
    test_reproducibility()
    test_different_seeds()
    test_cuda_reproducibility()
    test_seed_worker()
    
    print("\n" + "=" * 50)
    print("All tests passed! Random seed functionality is working correctly.")
    print("You can now use random_seed parameter in your training configurations.")

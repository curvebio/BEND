#!/usr/bin/env python
"""
Test script to verify GPU device selection functionality
"""

import torch
from bend.utils.embedders import get_device, OneHotEmbedder

def test_device_selection():
    """Test device selection function"""
    print("Testing device selection...")
    
    # Test auto selection
    device_auto = get_device()
    print(f"Auto-selected device: {device_auto}")
    
    if torch.cuda.is_available():
        # Test specific GPU selection
        device_0 = get_device(0)
        print(f"GPU 0 device: {device_0}")
        
        # Test with invalid GPU (should fall back gracefully)
        device_invalid = get_device(99)  # Assuming GPU 99 doesn't exist
        print(f"Invalid GPU (99) device: {device_invalid}")
        
        # Test OneHot embedder with device selection
        print("\nTesting OneHot embedder with device selection...")
        embedder_auto = OneHotEmbedder()
        embedder_gpu0 = OneHotEmbedder(device_id=0)
        
        print(f"Auto embedder device: {embedder_auto.device}")
        print(f"GPU 0 embedder device: {embedder_gpu0.device}")
        
        # Test embedding
        test_seq = "ATCGATCG"
        embedding = embedder_gpu0.embed([test_seq])
        print(f"Successfully embedded sequence on device: {embedder_gpu0.device}")
        print(f"Embedding shape: {embedding[0].shape}")
    else:
        print("CUDA not available, testing CPU-only functionality")
        device_cpu = get_device(0)  # Should fallback to CPU
        print(f"Requested GPU 0, got: {device_cpu}")

if __name__ == "__main__":
    test_device_selection()

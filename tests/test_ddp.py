#!/usr/bin/env python3
"""
test_ddp.py
-----------
Simple test script to verify DDP functionality.
This script creates a minimal model and tests DDP training.
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp


def setup_ddp(rank, world_size, port=12355):
    """Initialize DDP process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


class SimpleModel(nn.Module):
    """Simple test model."""
    def __init__(self, input_size=10, hidden_size=20, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def ddp_worker(rank, world_size, port=12355):
    """DDP worker function."""
    print(f"Rank {rank} starting...")
    
    try:
        # Setup DDP
        setup_ddp(rank, world_size, port)
        device = torch.device(f'cuda:{rank}')
        
        # Create model
        model = SimpleModel().to(device)
        model = DDP(model, device_ids=[rank])
        
        # Create dummy data
        batch_size = 32
        input_size = 10
        X = torch.randn(batch_size * 10, input_size)
        y = torch.randn(batch_size * 10, 1)
        dataset = TensorDataset(X, y)
        
        # Create distributed sampler
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Training loop
        model.train()
        for epoch in range(3):
            sampler.set_epoch(epoch)  # Important for proper shuffling
            epoch_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Average loss across all processes
            avg_loss = epoch_loss / len(dataloader)
            if dist.is_initialized():
                loss_tensor = torch.tensor(avg_loss, device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                avg_loss = loss_tensor.item() / world_size
            
            if rank == 0:  # Only print on main process
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
                
        print(f"Rank {rank} training completed successfully!")
        
    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        raise e
    finally:
        cleanup_ddp()


def test_ddp():
    """Test DDP functionality."""
    if not torch.cuda.is_available():
        print("CUDA not available. Cannot test DDP.")
        return
        
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print(f"Only {world_size} GPU available. DDP needs at least 2 GPUs for testing.")
        print("Running on single GPU instead...")
        # Single GPU test
        device = torch.device('cuda:0')
        model = SimpleModel().to(device)
        print("Single GPU test completed successfully!")
        return
    
    print(f"Testing DDP with {world_size} GPUs...")
    
    # Set multiprocessing start method to spawn for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    
    # Launch DDP processes using spawn method
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=ddp_worker, args=(rank, world_size))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    print("DDP test completed successfully!")


if __name__ == "__main__":
    print("Testing DDP setup...")
    test_ddp()

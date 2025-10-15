"""
Example of using Pipelayer with PyTorch FSDP (Fully Sharded Data Parallel).

This example demonstrates:
1. Wrapping a model with FSDP
2. Saving FSDP model checkpoints in chunked format
3. Loading FSDP checkpoints with pipelined streaming
4. Training with FSDP + pipelayer

Usage:
    # Single GPU (testing):
    python fsdp_example.py
    
    # Multi-GPU:
    torchrun --nproc_per_node=2 fsdp_example.py
"""

import os
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload

from pipelayer import (
    save_fsdp_model_chunked,
    FSDPPipelinedStateLoader,
    FSDPPipelayerModelWrapper,
)


class SimpleTransformerBlock(nn.Module):
    """A simple transformer-like block for testing."""
    
    def __init__(self, d_model: int = 512, nhead: int = 8, dim_feedforward: int = 2048):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # Self attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed forward
        ff_out = self.linear2(self.activation(self.linear1(x)))
        x = self.norm2(x + ff_out)
        return x


class SimpleModel(nn.Module):
    """A simple model with multiple transformer blocks."""
    
    def __init__(self, d_model: int = 512, num_layers: int = 4):
        super().__init__()
        self.embedding = nn.Linear(d_model, d_model)
        self.layers = nn.ModuleList([
            SimpleTransformerBlock(d_model) for _ in range(num_layers)
        ])
        self.output = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output(x)
        return x


def setup_distributed():
    """Initialize distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        # Single GPU mode
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_distributed(world_size):
    """Clean up distributed training."""
    if world_size > 1:
        dist.destroy_process_group()


def demo_fsdp_save_and_load():
    """Demonstrate FSDP checkpoint saving and loading with pipelayer."""
    
    # Setup distributed environment
    rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    
    if rank == 0:
        print(f"Running on {world_size} GPU(s)")
        print(f"Device: {device}")
    
    # Create model
    model = SimpleModel(d_model=512, num_layers=4)
    
    # Wrap with FSDP
    if world_size > 1 and torch.cuda.is_available():
        model = FSDP(
            model,
            device_id=local_rank,
            # Optional: CPU offload for large models
            # cpu_offload=CPUOffload(offload_params=True),
        )
        if rank == 0:
            print("Model wrapped with FSDP")
    else:
        model = model.to(device)
        if rank == 0:
            print("Running without FSDP (single GPU mode)")
    
    # Create optimizer (must be created after FSDP wrapping)
    optimizer = Adam(model.parameters(), lr=1e-3)
    
    # Simulate some training to create non-zero state
    if rank == 0:
        print("\n--- Simulating training ---")
    
    model.train()
    for step in range(3):
        # Create dummy batch
        batch = torch.randn(4, 32, 512, device=device)
        
        # Forward pass
        output = model(batch)
        loss = output.mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if rank == 0:
            print(f"Step {step + 1}, Loss: {loss.item():.4f}")
    
    # Save checkpoint in chunked format
    if rank == 0:
        print("\n--- Saving FSDP checkpoint ---")
    
    save_dir = "./fsdp_chkpt"
    num_chunks = save_fsdp_model_chunked(
        model=model,
        optimizer=optimizer,
        save_dir=save_dir,
        target_chunk_bytes=10 * 1024 * 1024,  # 10MB chunks for demo
        rank=rank,
    )
    
    if rank == 0:
        print(f"Saved {num_chunks} chunks to {save_dir}")
    
    # Synchronize before loading
    if world_size > 1:
        dist.barrier()
    
    # Create new model for loading
    if rank == 0:
        print("\n--- Loading checkpoint with pipelined loader ---")
    
    new_model = SimpleModel(d_model=512, num_layers=4)
    
    # Wrap with FSDP
    if world_size > 1 and torch.cuda.is_available():
        new_model = FSDP(new_model, device_id=local_rank)
    else:
        new_model = new_model.to(device)
    
    new_optimizer = Adam(new_model.parameters(), lr=1e-3)
    
    # Use pipelined loader
    loader = FSDPPipelinedStateLoader(
        model=new_model,
        optimizer=new_optimizer,
        chkpt_dir=save_dir,
        device=device,
        rank=rank,
        world_size=world_size,
    )
    
    if rank == 0:
        print("Waiting for chunks to load...")
    
    # Wait for all chunks
    for i in range(loader.num_chunks):
        loader.wait_for_chunk(i)
    
    # Finalize FSDP loading (important!)
    if isinstance(new_model, FSDP):
        loader.finalize_fsdp_loading()
    
    loader.stop()
    
    if rank == 0:
        print("Checkpoint loaded successfully!")
    
    # Test loaded model
    if rank == 0:
        print("\n--- Testing loaded model ---")
    
    new_model.eval()
    with torch.no_grad():
        test_batch = torch.randn(4, 32, 512, device=device)
        test_output = new_model(test_batch)
        if rank == 0:
            print(f"Test output shape: {test_output.shape}")
            print(f"Test output mean: {test_output.mean().item():.4f}")
    
    # Demo with wrapper
    if rank == 0:
        print("\n--- Using FSDPPipelayerModelWrapper ---")
    
    wrapped_model = SimpleModel(d_model=512, num_layers=4)
    if world_size > 1 and torch.cuda.is_available():
        wrapped_model = FSDP(wrapped_model, device_id=local_rank)
    else:
        wrapped_model = wrapped_model.to(device)
    
    wrapped_optimizer = Adam(wrapped_model.parameters(), lr=1e-3)
    
    wrapper = FSDPPipelayerModelWrapper(
        model=wrapped_model,
        optimizer=wrapped_optimizer,
        chkpt_dir=save_dir,
        load_checkpoint=True,
        device=device,
        rank=rank,
        world_size=world_size,
    )
    
    # Can use wrapper directly like a regular model
    # It will automatically switch to normal forward after loading
    if rank == 0:
        print("Wrapper created, forward pass will use pipelined loading initially")
    
    with torch.no_grad():
        test_batch = torch.randn(4, 32, 512, device=device)
        # This forward may wait for chunks if they're not all loaded yet
        test_output = wrapper(test_batch)
        if rank == 0:
            print(f"Wrapper output shape: {test_output.shape}")
    
    wrapper.cleanup()
    
    # Clean up
    cleanup_distributed(world_size)
    
    if rank == 0:
        print("\n--- Demo completed successfully! ---")
        # Clean up checkpoint directory
        import shutil
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        print(f"Cleaned up {save_dir}")


if __name__ == "__main__":
    demo_fsdp_save_and_load()

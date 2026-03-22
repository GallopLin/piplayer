"""
Tests for FSDP checkpointing functionality.

Note: These tests are designed to work in single-GPU mode.
Multi-GPU tests should be run with torchrun separately.
"""

import os
import shutil
import time

import torch
import torch.nn as nn
from torch.optim import Adam

# Try importing pytest, but don't require it
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Minimal pytest.mark.skipif replacement
    class _SkipIfMark:
        def __init__(self, condition, reason):
            self.condition = condition
            self.reason = reason
        def __call__(self, func):
            if self.condition:
                return lambda *args, **kwargs: None
            return func
    class _PytestMock:
        class mark:
            @staticmethod
            def skipif(condition, reason):
                return _SkipIfMark(condition, reason)
    pytest = _PytestMock()

# Try importing FSDP, skip tests if not available
try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False

from pipelayer.fsdp_checkpointing import (
    save_fsdp_model_chunked,
    FSDPPipelinedStateLoader,
)
from pipelayer.fsdp_wrapper import FSDPPipelayerModelWrapper


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_dim=16, hidden_dim=32, output_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def test_fsdp_save_and_load_single_gpu(tmp_path):
    """Test FSDP checkpoint saving and loading on single GPU (no actual FSDP)."""
    
    # Create a simple model
    model = SimpleModel()
    optimizer = Adam(model.parameters(), lr=1e-3)
    
    # Train for a few steps to create non-zero optimizer state
    model.train()
    for _ in range(3):
        x = torch.randn(4, 16)
        y = model(x)
        loss = y.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Save checkpoint
    save_dir = tmp_path / "fsdp_test"
    save_dir_str = str(save_dir)
    
    num_chunks = save_fsdp_model_chunked(
        model=model,
        optimizer=optimizer,
        save_dir=save_dir_str,
        target_chunk_bytes=1024,  # Small chunks for testing
        rank=0,
    )
    
    assert num_chunks >= 1
    assert os.path.exists(os.path.join(save_dir_str, "metadata.json"))
    assert os.path.exists(os.path.join(save_dir_str, "optimizer_param_groups.pt"))
    
    # Check metadata
    import json
    with open(os.path.join(save_dir_str, "metadata.json"), "r") as f:
        metadata = json.load(f)
    assert "fsdp_enabled" in metadata
    assert metadata["fsdp_enabled"] == False  # Not actually FSDP in this test
    
    # Save original state for comparison
    orig_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    
    # Zero out model
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    
    # Load checkpoint
    new_model = SimpleModel()
    new_optimizer = Adam(new_model.parameters(), lr=1e-3)
    
    loader = FSDPPipelinedStateLoader(
        model=new_model,
        optimizer=new_optimizer,
        chkpt_dir=save_dir_str,
        device="cpu",
        rank=0,
        world_size=1,
    )
    
    # Wait for all chunks
    deadline = time.time() + 10
    while time.time() < deadline:
        if all(loader.is_chunk_loaded(i) for i in range(loader.num_chunks)):
            break
        time.sleep(0.05)
    
    assert all(loader.is_chunk_loaded(i) for i in range(loader.num_chunks))
    loader.stop()
    
    # Compare states
    for k, v in new_model.state_dict().items():
        if isinstance(v, torch.Tensor) and isinstance(orig_state.get(k), torch.Tensor):
            assert torch.allclose(v.cpu(), orig_state[k].cpu(), atol=1e-6, rtol=1e-5), f"Mismatch in {k}"


def test_fsdp_wrapper_single_gpu(tmp_path):
    """Test FSDPPipelayerModelWrapper on single GPU."""
    
    # Create and train a model
    model = SimpleModel()
    optimizer = Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for _ in range(3):
        x = torch.randn(4, 16)
        y = model(x)
        loss = y.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Save checkpoint
    save_dir = tmp_path / "wrapper_test"
    save_dir_str = str(save_dir)
    
    save_fsdp_model_chunked(
        model=model,
        optimizer=optimizer,
        save_dir=save_dir_str,
        target_chunk_bytes=1024,
        rank=0,
    )
    
    # Create new model and load with wrapper
    new_model = SimpleModel()
    new_optimizer = Adam(new_model.parameters(), lr=1e-3)
    
    wrapper = FSDPPipelayerModelWrapper(
        model=new_model,
        optimizer=new_optimizer,
        chkpt_dir=save_dir_str,
        load_checkpoint=True,
        device="cpu",
        rank=0,
        world_size=1,
    )
    
    # Wait a bit for loading
    time.sleep(1.0)
    
    # Test forward pass
    wrapper.eval()
    with torch.no_grad():
        x = torch.randn(4, 16)
        y = wrapper(x)
        assert y.shape == (4, 8)
    
    wrapper.cleanup()


def test_fsdp_save_non_rank0(tmp_path):
    """Test that non-rank-0 processes don't save."""
    
    model = SimpleModel()
    optimizer = Adam(model.parameters(), lr=1e-3)
    
    save_dir = tmp_path / "non_rank0_test"
    save_dir_str = str(save_dir)
    
    # Call with rank != 0
    num_chunks = save_fsdp_model_chunked(
        model=model,
        optimizer=optimizer,
        save_dir=save_dir_str,
        target_chunk_bytes=1024,
        rank=1,  # Not rank 0
    )
    
    # Should return 0 and not create directory
    assert num_chunks == 0
    assert not os.path.exists(save_dir_str)


@pytest.mark.skipif(not FSDP_AVAILABLE, reason="FSDP not available")
def test_fsdp_import():
    """Test that FSDP imports work correctly."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    assert FSDP is not None


if __name__ == "__main__":
    # Run basic tests
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        from pathlib import Path
        tmp_path = Path(tmpdir)
        
        print("Running test_fsdp_save_and_load_single_gpu...")
        test_fsdp_save_and_load_single_gpu(tmp_path)
        print("✓ Passed")
        
        print("\nRunning test_fsdp_wrapper_single_gpu...")
        test_fsdp_wrapper_single_gpu(tmp_path)
        print("✓ Passed")
        
        print("\nRunning test_fsdp_save_non_rank0...")
        test_fsdp_save_non_rank0(tmp_path)
        print("✓ Passed")
        
        if FSDP_AVAILABLE:
            print("\nRunning test_fsdp_import...")
            test_fsdp_import()
            print("✓ Passed")
        
        print("\n✅ All tests passed!")

"""
FSDP-aware model wrapper for Pipelayer.

This module provides a wrapper for FSDP models that integrates with
pipelayer's pipelined checkpoint loading mechanism.
"""

from __future__ import annotations

import atexit
import threading
import time
from typing import Any, List, Tuple, Optional, Union

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from .fsdp_checkpointing import FSDPPipelinedStateLoader


class FSDPPipelayerModelWrapper(nn.Module):
    """
    Wrapper for FSDP models with pipelined checkpoint loading.
    
    This wrapper extends the concept of PipelayerModelWrapper to FSDP models,
    enabling checkpoint loading while training in multi-GPU scenarios.
    
    Key features:
    - Load FSDP checkpoints in chunks while training
    - Automatically switch to normal forward after loading completes
    - Preserve pipelayer's core philosophy for FSDP models
    """
    
    def __init__(
        self,
        model: Union[nn.Module, FSDP],
        optimizer: torch.optim.Optimizer,
        chkpt_dir: Optional[str] = None,
        load_checkpoint: bool = False,
        device: Optional[str] = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        """
        Initialize FSDP-aware Pipelayer wrapper.
        
        Args:
            model: FSDP-wrapped model or regular model
            optimizer: Optimizer (should be created after FSDP wrapping)
            chkpt_dir: Checkpoint directory
            load_checkpoint: Whether to load checkpoint on initialization
            device: Device string (default: cuda:rank if available)
            rank: Current process rank
            world_size: Total number of processes
        """
        super().__init__()
        
        self.rank = rank
        self.world_size = world_size
        self.is_fsdp = isinstance(model, FSDP)
        
        # Set device
        if device is None:
            if torch.cuda.is_available() and world_size > 1:
                device = f"cuda:{rank}"
            elif torch.cuda.is_available():
                device = "cuda:0"
            else:
                device = "cpu"
        self.device = torch.device(device)
        
        # Store model and optimizer
        self.original_model = model
        if not self.is_fsdp:
            self.original_model = self.original_model.to(self.device)
        self.optimizer = optimizer
        self.loader: Optional[FSDPPipelinedStateLoader] = None
        self.chkpt_dir = chkpt_dir
        
        # State management
        self._loading_complete = threading.Event()
        self._use_original_forward = not load_checkpoint
        self._partition_model_blocks()
        
        # Register cleanup
        atexit.register(self.cleanup)
        self._cleaned = False
        
        # Initialize loader if needed
        if load_checkpoint:
            if not chkpt_dir:
                raise ValueError("chkpt_dir must be provided when load_checkpoint is True")
            self._initialize_loader()
            self._start_loading_monitor()
    
    def _initialize_loader(self) -> None:
        """Initialize FSDP-aware loader."""
        self.loader = FSDPPipelinedStateLoader(
            model=self.original_model,
            optimizer=self.optimizer,
            chkpt_dir=self.chkpt_dir,
            device=str(self.device),
            rank=self.rank,
            world_size=self.world_size,
        )
    
    def _partition_model_blocks(self) -> None:
        """
        Partition model into blocks for pipelined forward.
        
        For FSDP models, we use a simplified partitioning strategy
        since FSDP already handles parameter sharding.
        """
        if self._use_original_forward:
            return
        
        self.compute_blocks: List[Tuple[str, nn.Module]] = []
        
        # Get the base model (unwrap FSDP if needed)
        base_model = self.original_model
        if self.is_fsdp:
            # Access the wrapped module
            while hasattr(base_model, 'module') or hasattr(base_model, '_fsdp_wrapped_module'):
                if hasattr(base_model, '_fsdp_wrapped_module'):
                    base_model = base_model._fsdp_wrapped_module
                elif hasattr(base_model, 'module'):
                    base_model = base_model.module
                else:
                    break
        
        # Collect leaf modules
        for name, module in base_model.named_modules():
            if len(list(module.children())) == 0:
                self.compute_blocks.append((name, module))
    
    def _start_loading_monitor(self) -> None:
        """Start background thread to monitor loading completion."""
        
        def monitor_loading() -> None:
            try:
                while self.loader and not self._loading_complete.is_set():
                    # Check if all chunks are loaded
                    all_loaded = True
                    for i in range(self.loader.num_chunks):
                        if not self._check_chunk_loaded(i):
                            all_loaded = False
                            break
                    
                    if all_loaded:
                        # All chunks loaded, finalize FSDP loading
                        if self.is_fsdp:
                            self.loader.finalize_fsdp_loading()
                        
                        if self.rank == 0:
                            print("[FSDP PipeLayer] All chunks loaded, switching to original forward")
                        self._switch_to_original_forward()
                        break
                    
                    time.sleep(0.1)
            except Exception as e:
                if self.rank == 0:
                    print(f"[FSDP PipeLayer] Loading monitor error: {e}")
        
        monitor_thread = threading.Thread(target=monitor_loading, daemon=True)
        monitor_thread.start()
    
    def _check_chunk_loaded(self, chunk_idx: int) -> bool:
        """Check if a chunk is loaded (non-blocking)."""
        if self.loader is None:
            return False
        return self.loader.is_chunk_loaded(chunk_idx)
    
    def _switch_to_original_forward(self) -> None:
        """Switch to using original model's forward method."""
        self._use_original_forward = True
        self._loading_complete.set()
        
        # Clean up loader
        if self.loader:
            self.loader.stop()
            self.loader = None
        
        # Clean up compute blocks
        if hasattr(self, "compute_blocks"):
            del self.compute_blocks
        
        # Force garbage collection
        import gc
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
    
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Forward pass with intelligent routing.
        
        Routes to original model forward if loading is complete,
        otherwise uses pipelined forward that waits for chunks.
        """
        if self._use_original_forward:
            return self.original_model(*args, **kwargs)
        
        return self._pipelined_forward(*args, **kwargs)
    
    def _pipelined_forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Pipelined forward pass that waits for required chunks.
        
        This is a simplified implementation for FSDP models.
        In practice, FSDP's communication patterns make fine-grained
        chunk waiting less beneficial, so we wait for all chunks.
        """
        # For FSDP models, wait for all chunks before forward
        # This is because FSDP needs complete parameter shards
        assert self.loader is not None, "Loader must be initialized for pipelined forward"
        
        for i in range(self.loader.num_chunks):
            self.loader.wait_for_chunk(i)
        
        # Once all chunks are ready, use original forward
        return self.original_model(*args, **kwargs)
    
    def force_switch_to_original(self) -> None:
        """Manually force switch to original forward."""
        if self.rank == 0:
            print("[FSDP PipeLayer] Manually switching to original forward")
        self._switch_to_original_forward()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if not getattr(self, "_cleaned", False):
            self._loading_complete.set()
            if hasattr(self, "loader") and self.loader is not None:
                self.loader.stop()
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            self._cleaned = True
    
    def __del__(self) -> None:
        if not getattr(self, "_cleaned", False):
            try:
                self.cleanup()
            except Exception:
                pass

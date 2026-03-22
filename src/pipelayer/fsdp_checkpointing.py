"""
FSDP-aware checkpointing utilities for Pipelayer.

This module extends the core pipelayer checkpointing to support PyTorch FSDP
(Fully Sharded Data Parallel) for single-machine, multi-GPU scenarios.

Key features:
- Save FSDP model checkpoints in chunked format
- Load FSDP model checkpoints with pipelined streaming
- Preserve pipelayer's core philosophy: load while training
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import threading
from collections import defaultdict
from typing import Dict, Any, List, Optional, Union

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.fsdp.api import FullOptimStateDictConfig

from .checkpointing import _flush_chunk, PipelinedStateLoader


def save_fsdp_model_chunked(
    model: Union[nn.Module, FSDP],
    optimizer: torch.optim.Optimizer,
    save_dir: str,
    target_chunk_bytes: int = 50 * 1024 * 1024,
    rank: int = 0,
) -> int:
    """
    Save FSDP model and optimizer state in chunked format.
    
    This function handles FSDP models by first gathering the full state dict
    on rank 0, then chunking it similar to the original save_model_chunked.
    
    Args:
        model: FSDP-wrapped model or regular model
        optimizer: Optimizer (should be created after FSDP wrapping)
        save_dir: Directory to save chunks
        target_chunk_bytes: Target size for each chunk (default 50MB)
        rank: Current process rank (only rank 0 saves)
    
    Returns:
        Number of chunks created (only on rank 0, others return 0)
    """
    # Only rank 0 performs saving
    if rank != 0:
        # Other ranks need to participate in state dict gathering but don't save
        if isinstance(model, FSDP):
            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
            ):
                _ = model.state_dict()
                _ = FSDP.optim_state_dict(model, optimizer)
        return 0
    
    # Rank 0: gather full state dict and save
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # Gather full state dict from all ranks
    if isinstance(model, FSDP):
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            model_state_dict = model.state_dict()
            optim_state_dict = FSDP.optim_state_dict(model, optimizer)
    else:
        # Regular model (non-FSDP)
        model_state_dict = model.state_dict()
        optim_state_dict = optimizer.state_dict()
    
    # Build parameter mappings
    param_name_to_param = {}
    id_to_name = {}
    
    # For FSDP models, we need to handle the wrapped model
    if isinstance(model, FSDP):
        # FSDP wraps the original model, access it via module attribute
        original_model = model
        while hasattr(original_model, 'module') or hasattr(original_model, '_fsdp_wrapped_module'):
            if hasattr(original_model, '_fsdp_wrapped_module'):
                original_model = original_model._fsdp_wrapped_module
            elif hasattr(original_model, 'module'):
                original_model = original_model.module
            else:
                break
        
        # Use the full state dict keys
        for name in model_state_dict.keys():
            if name in model_state_dict and isinstance(model_state_dict[name], torch.Tensor):
                # Create a dummy parameter for mapping
                param_name_to_param[name] = model_state_dict[name]
                id_to_name[id(model_state_dict[name])] = name
    else:
        for name, param in model.named_parameters():
            param_name_to_param[name] = param
            id_to_name[id(param)] = name
    
    # Build param -> optim state mapping
    param_to_optim_state: Dict[Any, Dict[str, Any]] = {}
    
    # FSDP optimizer state dict has different structure
    if isinstance(model, FSDP):
        # FSDP optim state dict uses parameter names directly
        for name, state in optim_state_dict.get("state", {}).items():
            if name in param_name_to_param:
                param_to_optim_state[param_name_to_param[name]] = state
    else:
        # Regular optimizer state dict uses parameter IDs
        for group in optim_state_dict.get("param_groups", []):
            for param_id in group.get("params", []):
                if param_id in optim_state_dict.get("state", {}) and param_id in id_to_name:
                    name = id_to_name[param_id]
                    param = param_name_to_param[name]
                    param_to_optim_state[param] = optim_state_dict["state"][param_id]
    
    # Group parameters by prefix
    groups: Dict[str, List[str]] = defaultdict(list)
    for key in model_state_dict.keys():
        prefix = key.split(".", 1)[0] if "." in key else "__root__"
        groups[prefix].append(key)
    
    chunk_metadata: Dict[int, Dict[str, Any]] = {}
    chunk_idx = 0
    
    def tensor_size_bytes(tensor: torch.Tensor) -> int:
        return int(tensor.numel() * tensor.element_size())
    
    # Chunk parameters by group
    for group_name, state_keys_in_group in groups.items():
        cur_chunk_keys: List[str] = []
        cur_chunk_bytes = 0
        
        for key in state_keys_in_group:
            t = model_state_dict[key]
            if not isinstance(t, torch.Tensor):
                continue
            b = tensor_size_bytes(t)
            
            # Flush if adding this would exceed target and we have content
            if cur_chunk_bytes + b > target_chunk_bytes and cur_chunk_keys:
                chunk_idx = _flush_chunk(
                    save_dir,
                    chunk_idx,
                    cur_chunk_keys,
                    model_state_dict,
                    param_name_to_param,
                    param_to_optim_state,
                    chunk_metadata,
                    group_name,
                )
                cur_chunk_keys = []
                cur_chunk_bytes = 0
            
            cur_chunk_keys.append(key)
            cur_chunk_bytes += b
            
            # If single param exceeds target, flush immediately
            if cur_chunk_bytes >= target_chunk_bytes:
                chunk_idx = _flush_chunk(
                    save_dir,
                    chunk_idx,
                    cur_chunk_keys,
                    model_state_dict,
                    param_name_to_param,
                    param_to_optim_state,
                    chunk_metadata,
                    group_name,
                )
                cur_chunk_keys = []
                cur_chunk_bytes = 0
        
        # Flush remaining
        if cur_chunk_keys:
            chunk_idx = _flush_chunk(
                save_dir,
                chunk_idx,
                cur_chunk_keys,
                model_state_dict,
                param_name_to_param,
                param_to_optim_state,
                chunk_metadata,
                group_name,
            )
    
    # Save optimizer param_groups
    new_param_groups = []
    if isinstance(model, FSDP):
        # FSDP optimizer uses parameter names
        for group in optim_state_dict.get("param_groups", []):
            new_group = dict(group)
            # Params should already be names in FSDP case
            if "params" in new_group and isinstance(new_group["params"], list):
                new_param_groups.append(new_group)
    else:
        # Regular optimizer: convert IDs to names
        for group in optim_state_dict.get("param_groups", []):
            new_group = dict(group)
            new_group["params"] = []
            for param_id in group.get("params", []):
                if param_id in id_to_name:
                    new_group["params"].append(id_to_name[param_id])
            new_param_groups.append(new_group)
    
    torch.save(new_param_groups, os.path.join(save_dir, "optimizer_param_groups.pt"))
    
    # Save metadata
    metadata = {
        "num_chunks": chunk_idx,
        "chunks": chunk_metadata,
        "fsdp_enabled": isinstance(model, FSDP),
    }
    with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    
    total_params = sum(p.numel() for p in model_state_dict.values() if isinstance(p, torch.Tensor))
    print(
        f"[FSDP Saver] Model ({total_params/1e6:.2f}M params) and Optimizer "
        f"chunked into {chunk_idx} parts -> {save_dir}"
    )
    return chunk_idx


class FSDPPipelinedStateLoader(PipelinedStateLoader):
    """
    FSDP-aware pipelined state loader.
    
    This loader extends PipelinedStateLoader to handle FSDP models correctly.
    It loads the full state dict in chunks and uses FSDP's state dict loading
    mechanisms to distribute parameters across ranks.
    """
    
    def __init__(
        self,
        model: Union[nn.Module, FSDP],
        optimizer: torch.optim.Optimizer,
        chkpt_dir: str,
        device: str = "cuda:0",
        host_queue_maxsize: int = 32,
        num_copy_workers: Optional[int] = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        """
        Initialize FSDP-aware pipelined state loader.
        
        Args:
            model: FSDP-wrapped model
            optimizer: Optimizer (created after FSDP wrapping)
            chkpt_dir: Checkpoint directory
            device: Device string
            host_queue_maxsize: Max size of host buffer queue
            num_copy_workers: Number of copy worker threads
            rank: Current process rank
            world_size: Total number of processes
        """
        self.rank = rank
        self.world_size = world_size
        self.is_fsdp = isinstance(model, FSDP)
        
        # For FSDP models, we need to handle state dict loading differently
        # We'll collect the full state dict in chunks, then load it all at once
        # using FSDP's set_state_dict method
        if self.is_fsdp and rank == 0:
            # Only rank 0 loads from disk
            self.accumulated_model_state = {}
            self.accumulated_optim_state = {}
        
        # Initialize parent class
        # Note: For FSDP, all ranks need to participate in the loading process
        super().__init__(
            model=model,
            optimizer=optimizer,
            chkpt_dir=chkpt_dir,
            device=device,
            host_queue_maxsize=host_queue_maxsize,
            num_copy_workers=num_copy_workers,
        )
    
    def _apply_model_chunk(self, host_model_chunk: Dict[str, torch.Tensor]) -> None:
        """
        Apply model chunk to the model.
        
        For FSDP models, we accumulate chunks and apply them all at once
        after all chunks are loaded, using FSDP's state dict loading.
        """
        if self.is_fsdp and self.rank == 0:
            # Accumulate state dict for later application
            for name, tensor in host_model_chunk.items():
                if isinstance(tensor, torch.Tensor):
                    self.accumulated_model_state[name] = tensor.to(
                        self.device, non_blocking=self.device.type == "cuda"
                    )
        else:
            # For non-FSDP or non-rank-0, use parent implementation
            super()._apply_model_chunk(host_model_chunk)
    
    def _apply_optimizer_chunk(self, host_optim_chunk: Dict[str, Dict[str, Any]]) -> None:
        """
        Apply optimizer chunk to the optimizer.
        
        For FSDP models, we accumulate chunks and apply them all at once.
        """
        if self.is_fsdp and self.rank == 0:
            # Accumulate optimizer state for later application
            for param_name, state_dict in host_optim_chunk.items():
                new_state: Dict[str, Any] = {}
                for state_name, value in state_dict.items():
                    if isinstance(value, torch.Tensor):
                        new_state[state_name] = value.to(
                            self.device, non_blocking=self.device.type == "cuda"
                        )
                    else:
                        new_state[state_name] = value
                self.accumulated_optim_state[param_name] = new_state
        else:
            # For non-FSDP, use parent implementation
            super()._apply_optimizer_chunk(host_optim_chunk)
    
    def finalize_fsdp_loading(self) -> None:
        """
        Finalize FSDP model and optimizer loading.
        
        This method should be called after all chunks are loaded.
        It applies the accumulated state dict to the FSDP model.
        """
        if not self.is_fsdp:
            return
        
        if self.rank == 0:
            # Rank 0: Load the accumulated state dict into the model
            with FSDP.state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
                FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=False),
            ):
                # Load model state
                self.model.load_state_dict(self.accumulated_model_state)
                
                # Load optimizer state
                if self.accumulated_optim_state:
                    optim_state_dict = {
                        "state": self.accumulated_optim_state,
                        "param_groups": self.optimizer.param_groups,
                    }
                    FSDP.optim_state_dict_to_load(
                        self.model, self.optimizer, optim_state_dict
                    )
            
            # Clean up accumulated state to free memory
            self.accumulated_model_state = {}
            self.accumulated_optim_state = {}
        else:
            # Other ranks: participate in state dict loading
            with FSDP.state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
                FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=False),
            ):
                # Load empty state dict (data will be broadcast from rank 0)
                self.model.load_state_dict({})
                FSDP.optim_state_dict_to_load(self.model, self.optimizer, {})

import os
import shutil
import time

import torch
import torch.nn as nn
from torch.optim import Adam

from pipelayer.checkpointing import save_model_chunked, PipelinedStateLoader


def test_chunk_save_and_cpu_load(tmp_path):
    # Small toy model
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
    opt = Adam(model.parameters(), lr=1e-3)

    # Save checkpoint
    save_dir = tmp_path / "chkpt"
    save_dir_str = str(save_dir)
    chunks = save_model_chunked(model, opt, save_dir=save_dir_str, target_chunk_bytes=1024)  # force multiple chunks
    assert chunks >= 1
    assert os.path.exists(os.path.join(save_dir_str, "metadata.json"))
    assert os.path.exists(os.path.join(save_dir_str, "optimizer_param_groups.pt"))

    # Keep a copy of original state dict to compare
    orig_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # Zero out model
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

    # Load via loader on CPU
    loader = PipelinedStateLoader(model, opt, chkpt_dir=save_dir_str, device="cpu", host_queue_maxsize=4, num_copy_workers=1)

    # Wait until all chunks loaded
    deadline = time.time() + 10
    while time.time() < deadline:
        if all(loader.is_chunk_loaded(i) for i in range(loader.num_chunks)):
            break
        time.sleep(0.05)

    assert all(loader.is_chunk_loaded(i) for i in range(loader.num_chunks)), "Chunks not loaded in time"
    loader.stop()

    # Compare a few tensors
    for k, v in model.state_dict().items():
        if isinstance(v, torch.Tensor) and isinstance(orig_state.get(k), torch.Tensor):
            assert torch.allclose(v.cpu(), orig_state[k].cpu(), atol=1e-6, rtol=1e-5)
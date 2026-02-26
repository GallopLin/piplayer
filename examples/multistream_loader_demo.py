"""Minimal demo for MultiStreamStateLoader.

This script is safe to run even when checkpoint files are missing; it will
print a hint and exit early.
"""

import os
import torch
import torch.nn as nn
from torch.optim import Adam

from pipelayer.checkpointing import MultiStreamStateLoader


def main():
    chkpt_dir = os.environ.get("MULTISTREAM_CHKPT_DIR", "./chkpt")
    metadata_file = os.environ.get("MULTISTREAM_METADATA", "checkpoint.chk.metadata.json")
    checkpoint_file = os.environ.get("MULTISTREAM_FILE", "checkpoint.chk")
    lib_path = os.environ.get("MULTISTREAM_LIB", "./libtest_ssd.so")

    metadata_path = metadata_file if os.path.isabs(metadata_file) else os.path.join(chkpt_dir, metadata_file)
    checkpoint_path = checkpoint_file if os.path.isabs(checkpoint_file) else os.path.join(chkpt_dir, checkpoint_file)

    if not os.path.exists(metadata_path) or not os.path.exists(checkpoint_path):
        print("[Demo] Missing checkpoint files. Set MULTISTREAM_CHKPT_DIR/MULTISTREAM_METADATA/MULTISTREAM_FILE to run.")
        return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
    optimizer = Adam(model.parameters(), lr=1e-3)

    loader = MultiStreamStateLoader(
        model=model,
        optimizer=optimizer,
        chkpt_dir=chkpt_dir,
        lib_path=lib_path,
        metadata_file=metadata_file,
        checkpoint_file=checkpoint_file,
        device=device,
    )

    for i in range(loader.num_chunks):
        loader.wait_for_chunk(i)

    loader.stop()
    print("[Demo] Multistream checkpoint loaded successfully.")


if __name__ == "__main__":
    main()

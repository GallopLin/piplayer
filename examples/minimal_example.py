import torch
import torch.nn as nn
from torch.optim import Adam

from pipelayer.checkpointing import save_model_chunked, PipelinedStateLoader
from pipelayer.wrapper import PipelayerModelWrapper


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10))
    model.to(device)
    opt = Adam(model.parameters(), lr=1e-3)

    # Save chunked checkpoint
    num_chunks = save_model_chunked(model, opt, save_dir="./chkpt", target_chunk_bytes=50 * 1024 * 1024)
    print("Saved chunks:", num_chunks)

    # Zero model to demonstrate load
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

    # Load with loader
    loader = PipelinedStateLoader(model, opt, chkpt_dir="./chkpt", device=device)
    for i in range(loader.num_chunks):
        loader.wait_for_chunk(i)
    loader.stop()

    # Or use wrapper
    wrapped = PipelayerModelWrapper(
        model=nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10)),
        optimizer=Adam(model.parameters(), lr=1e-3),
        chkpt_dir="./chkpt",
        load_checkpoint=True,
        device=device,
    )

    x = torch.randn(4, 128, device=device)
    y = wrapped(x)
    print("Output shape:", y.shape)


if __name__ == "__main__":
    main()
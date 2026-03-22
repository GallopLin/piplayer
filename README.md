# Pipelayer

Pipelayer 提供以下能力：
- 将 PyTorch 模型与优化器状态分块保存（默认 50MB 为目标 chunk 大小）。
- 使用独立的 IO 线程与复制工作线程，从 CPU pinned 内存向设备侧（CUDA 或 CPU 回退）进行非阻塞状态加载。
- 提供 PipelayerModelWrapper 在加载过程中分块等待、逐步可用，加载完成后自动切换到原始 forward。
- **NEW**: 支持 PyTorch FSDP（Fully Sharded Data Parallel）单机多卡训练场景。

特性亮点：
- 面向大模型训练检查点的快启、流水化 IO/拷贝。
- 优化器状态与参数名对齐，便于恢复优化器。
- 事件与流的正确同步，避免 CUDA 隐式同步导致的性能回退。
- CPU 回退路径（无 CUDA 环境也可运行和测试）。
- **FSDP 支持**：在单机多卡场景下保存和加载 FSDP 模型检查点，同时保留 pipelayer 核心哲学。

## 安装

该包依赖 PyTorch，请先安装 torch（CPU 或 CUDA 版本，视环境而定）。例如在 CI 或本地 CPU 环境：

```bash
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

> 若你已安装 GPU 版本的 PyTorch，请按官方指引安装对应版本。

## 快速上手

```python
import torch
import torch.nn as nn
from torch.optim import Adam
from pipelayer.checkpointing import save_model_chunked, PipelinedStateLoader
from pipelayer.wrapper import PipelayerModelWrapper

# 1) 定义模型与优化器
model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10))
opt = Adam(model.parameters(), lr=1e-3)

# 2) 保存分块检查点
num_chunks = save_model_chunked(model, opt, save_dir="./chkpt", target_chunk_bytes=50 * 1024 * 1024)
print("Chunks:", num_chunks)

# 3) 恢复时的两种方式：

# 3.a) 直接使用 PipelinedStateLoader 恢复到现有模型与优化器（支持 CPU、CUDA）
loader = PipelinedStateLoader(model, opt, chkpt_dir="./chkpt", device="cuda:0" if torch.cuda.is_available() else "cpu")
# 等待所有 chunk 加载完成（也可以在你的 forward 中逐块等待）
for i in range(loader.num_chunks):
    if loader.device.type == "cuda":
        # CUDA: 在当前流上与事件同步
        loader.wait_for_chunk(i)
    else:
        # CPU: 同步事件
        while not loader.is_chunk_loaded(i):
            pass
loader.stop()

# 3.b) 使用 PipelayerModelWrapper，在推理/训练期间渐进可用，加载完成后自动切换到原始 forward
wrapped = PipelayerModelWrapper(model=nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10)),
                                optimizer=Adam(model.parameters(), lr=1e-3),
                                chkpt_dir="./chkpt",
                                load_checkpoint=True,
                                device="cuda:0" if torch.cuda.is_available() else "cpu")

x = torch.randn(4, 128)
y = wrapped(x)  # 加载过程中会按需等待参数所在 chunk
```

## FSDP 支持（单机多卡）

Pipelayer 现已支持 PyTorch FSDP，允许在单机多卡场景下使用流水线加载检查点。

### FSDP 快速上手

```python
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from pipelayer import save_fsdp_model_chunked, FSDPPipelinedStateLoader, FSDPPipelayerModelWrapper

# 初始化分布式环境
dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ.get("LOCAL_RANK", 0))

# 创建模型并用 FSDP 包装
model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10))
model = FSDP(model, device_id=local_rank)

# 创建优化器（必须在 FSDP 包装后）
optimizer = Adam(model.parameters(), lr=1e-3)

# 保存 FSDP 检查点（仅 rank 0 保存）
num_chunks = save_fsdp_model_chunked(
    model=model,
    optimizer=optimizer,
    save_dir="./fsdp_chkpt",
    target_chunk_bytes=50 * 1024 * 1024,
    rank=rank,
)

# 加载检查点方式 1：使用 FSDPPipelinedStateLoader
loader = FSDPPipelinedStateLoader(
    model=model,
    optimizer=optimizer,
    chkpt_dir="./fsdp_chkpt",
    device=f"cuda:{local_rank}",
    rank=rank,
    world_size=world_size,
)

# 等待所有 chunk 加载完成
for i in range(loader.num_chunks):
    loader.wait_for_chunk(i)

# 完成 FSDP 加载（重要！）
loader.finalize_fsdp_loading()
loader.stop()

# 加载检查点方式 2：使用 FSDPPipelayerModelWrapper
wrapper = FSDPPipelayerModelWrapper(
    model=model,
    optimizer=optimizer,
    chkpt_dir="./fsdp_chkpt",
    load_checkpoint=True,
    device=f"cuda:{local_rank}",
    rank=rank,
    world_size=world_size,
)

# 可以直接使用 wrapper，它会自动处理加载和切换
x = torch.randn(4, 128, device=f"cuda:{local_rank}")
y = wrapper(x)
```

### 运行 FSDP 示例

```bash
# 单 GPU 测试（不使用实际 FSDP）
python examples/fsdp_example.py

# 多 GPU（例如 2 卡）
torchrun --nproc_per_node=2 examples/fsdp_example.py
```

### FSDP 设计要点

1. **保存时**：
   - 所有 rank 参与状态字典收集
   - 仅 rank 0 实际保存文件到磁盘
   - 使用 FSDP 的 `FULL_STATE_DICT` 模式收集完整参数

2. **加载时**：
   - 仅 rank 0 从磁盘读取并累积 chunks
   - 所有 rank 参与 FSDP 的 `load_state_dict` 调用
   - 参数会自动从 rank 0 广播到其他 ranks
   - 必须调用 `finalize_fsdp_loading()` 完成加载

3. **核心哲学保留**：
   - 检查点以 chunks 形式流式加载
   - 训练可以在加载过程中开始（等待必要的 chunks）
   - 加载完成后自动切换到正常训练模式

## 设计说明

- 分块保存：
  - 先按第一层 prefix（例如 `encoder`, `decoder`）进行分组，保持原语义。
  - 每个组内按 `target_chunk_bytes` 尺寸阈值切分子 chunk。
  - 同时保存优化器 `param_groups`（将 `id` 替换为 `name`）和各 chunk 对应的参数名列表的 `metadata.json`。

- 流水加载：
  - IO 线程顺序读取 chunk，映射到 CPU，pin_memory 后提交至队列。
  - 多个 copy worker 使用独立 CUDA stream（或 CPU 回退路径）将张量拷贝/设置到目标设备上，更新模型参数和优化器 state。
  - 使用每个 chunk 对应的事件进行就绪同步，前向执行可按需等待。

## 多流检查点加载（multistream）

当检查点使用 multistream 格式保存（单个文件中包含多个 checkpoint），可以使用新的 `MultiStreamStateLoader` 进行流水加载：

```python
from pipelayer.checkpointing import MultiStreamStateLoader

loader = MultiStreamStateLoader(
  model,
  optimizer,
  chkpt_dir="/path/to/chkpt_dir",
  lib_path="/path/to/libtest_ssd.so",
  metadata_file="checkpoint.chk.metadata.json",
  checkpoint_file="checkpoint.chk",
  device="cuda:0",
  load_grad=False,
)

for i in range(loader.num_chunks):
  loader.wait_for_chunk(i)
loader.stop()
```

说明：
- `metadata_file` 来自 multistream 保存端导出的元数据文件（包含 stream offsets 和 layer group 切分信息）。
- `checkpoint_file` 为 multistream 的 checkpoint 文件。
- 若要使用 `PipelayerModelWrapper`，可传入 `loader_cls=MultiStreamStateLoader` 及 `loader_kwargs`。

- 重要修复：
  - 原实现将数据复制到 `model.state_dict()[name]` 上不会更新模型，应使用 `named_parameters` 和 `named_buffers` 获取原位引用，进行 `param.data.copy_(...)`。

## 兼容性
- Python 3.9+
- torch >= 1.13
- 支持 CUDA / CPU 回退

## 测试

```bash
// 检查点恢复
python3.9 /home/linzhicheng/download/pipelayer/examples/run_clm_pipelayer2.py \
    --model_name_or_path facebook/opt-1.3b \
    --output_dir /home/linzhicheng/download/ckpt \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 1 \
    --use_pipelayer \
    --pipelayer_loader multistream \
    --multistream_lib_path /home/linzhicheng/code/pccheck/checkpoint_eval/pccheck/libtest_ssd.so \
    --multistream_checkpoint_file /home/linzhicheng/multistream_checkpoint.chk \
    --multistream_metadata_file /home/linzhicheng/multistream_checkpoint.chk.metadata.json \
    --resume_from_checkpoint dummy_resume/step_1 \
    --overwrite_output_dir \
    --max_train_steps 5 

// 检查点保存
python3.9 /home/linzhicheng/code/transformers/examples/pytorch/language-modeling/run_clm_multistream.py \
    --model_name_or_path facebook/opt-1.3b \
    --output_dir /home/linzhicheng/download/ckpt \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --per_device_train_batch_size 1 \
    --max_async 2 \
    --num_threads 2 \
    --num_layer_groups 6 \
    --cfreq 50 \
    --bench_total_steps 100 \
    --c_lib_path /home/linzhicheng/code/pccheck/checkpoint_eval/pccheck/libtest_ssd.so \
    --multistream_checkpoint_file multistream_test.chk \
    --overwrite_output_dir \
    --multistream_metadata_file multistream_test.chk.metadata.json
```

在 CI 中我们安装 CPU 版本的 PyTorch 以保障环境一致性。

## 许可

本项目基于 MIT 许可证发布。详见 [LICENSE](LICENSE)。

## 致谢

- 初始实现与需求来自 GallopLin。
# Pipelayer FSDP 扩展文档

## 概述

本文档详细说明了 Pipelayer 对 PyTorch FSDP（Fully Sharded Data Parallel）的支持实现。FSDP 扩展保留了 Pipelayer 的核心思想（边加载边训练），并将其扩展到单机多卡场景。

## 核心设计理念

### 保留的核心思想
1. **流水线加载**：检查点以分块方式流式加载，无需等待全部加载完成即可开始训练
2. **异步 IO**：独立的 IO 线程负责从磁盘读取，多个拷贝工作线程负责将数据传输到设备
3. **自动切换**：加载完成后自动切换到原始 forward 函数，不影响正常训练

### FSDP 特有设计
1. **分布式协调**：
   - 保存时：所有 rank 参与状态字典收集，但只有 rank 0 实际保存文件
   - 加载时：只有 rank 0 从磁盘读取，然后通过 FSDP API 广播到其他 ranks

2. **状态字典处理**：
   - 使用 FSDP 的 `FULL_STATE_DICT` 模式收集完整参数
   - 优化器状态也通过 FSDP API 正确处理

3. **内存效率**：
   - 累积式加载：rank 0 逐块累积状态字典，全部加载后一次性应用
   - 其他 ranks 不需要加载数据，通过 FSDP 的状态字典加载机制接收数据

## API 文档

### 1. save_fsdp_model_chunked

保存 FSDP 模型和优化器状态为分块格式。

```python
def save_fsdp_model_chunked(
    model: Union[nn.Module, FSDP],
    optimizer: torch.optim.Optimizer,
    save_dir: str,
    target_chunk_bytes: int = 50 * 1024 * 1024,
    rank: int = 0,
) -> int
```

**参数**：
- `model`: FSDP 包装的模型或普通模型
- `optimizer`: 优化器（必须在 FSDP 包装后创建）
- `save_dir`: 保存目录
- `target_chunk_bytes`: 每个块的目标大小（默认 50MB）
- `rank`: 当前进程的 rank（只有 rank 0 会保存文件）

**返回值**：
- 创建的块数量（rank 0），其他 ranks 返回 0

**使用示例**：
```python
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from pipelayer import save_fsdp_model_chunked

# 初始化分布式
dist.init_process_group("nccl")
rank = dist.get_rank()
local_rank = int(os.environ["LOCAL_RANK"])

# 创建并包装模型
model = MyModel()
model = FSDP(model, device_id=local_rank)
optimizer = torch.optim.Adam(model.parameters())

# 保存（所有 ranks 调用，但只有 rank 0 实际保存）
num_chunks = save_fsdp_model_chunked(
    model=model,
    optimizer=optimizer,
    save_dir="./checkpoint",
    rank=rank,
)
```

### 2. FSDPPipelinedStateLoader

FSDP 感知的流水线状态加载器。

```python
class FSDPPipelinedStateLoader(PipelinedStateLoader):
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
    )
```

**参数**：
- `model`: FSDP 包装的模型
- `optimizer`: 优化器
- `chkpt_dir`: 检查点目录
- `device`: 设备字符串
- `host_queue_maxsize`: 主机缓冲队列的最大大小
- `num_copy_workers`: 拷贝工作线程数量
- `rank`: 当前进程 rank
- `world_size`: 总进程数

**重要方法**：
- `wait_for_chunk(chunk_idx)`: 等待特定块加载完成
- `finalize_fsdp_loading()`: 完成 FSDP 加载（必须调用！）
- `stop()`: 停止加载器并清理资源

**使用示例**：
```python
from pipelayer import FSDPPipelinedStateLoader

loader = FSDPPipelinedStateLoader(
    model=model,
    optimizer=optimizer,
    chkpt_dir="./checkpoint",
    device=f"cuda:{local_rank}",
    rank=rank,
    world_size=world_size,
)

# 等待所有块加载
for i in range(loader.num_chunks):
    loader.wait_for_chunk(i)

# 重要：完成 FSDP 加载
loader.finalize_fsdp_loading()
loader.stop()
```

### 3. FSDPPipelayerModelWrapper

FSDP 模型的 Pipelayer 包装器，提供自动加载和切换功能。

```python
class FSDPPipelayerModelWrapper(nn.Module):
    def __init__(
        self,
        model: Union[nn.Module, FSDP],
        optimizer: torch.optim.Optimizer,
        chkpt_dir: Optional[str] = None,
        load_checkpoint: bool = False,
        device: Optional[str] = None,
        rank: int = 0,
        world_size: int = 1,
    )
```

**特性**：
- 自动管理检查点加载
- 加载期间使用流水线 forward
- 加载完成后自动切换到原始 forward
- 自动资源清理

**使用示例**：
```python
from pipelayer import FSDPPipelayerModelWrapper

wrapper = FSDPPipelayerModelWrapper(
    model=model,
    optimizer=optimizer,
    chkpt_dir="./checkpoint",
    load_checkpoint=True,
    device=f"cuda:{local_rank}",
    rank=rank,
    world_size=world_size,
)

# 可以直接使用，内部自动处理加载
output = wrapper(input_tensor)

# 可选：手动强制切换到原始 forward
wrapper.force_switch_to_original()

# 清理
wrapper.cleanup()
```

## 使用指南

### 完整训练流程示例

```python
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import Adam

from pipelayer import (
    save_fsdp_model_chunked,
    FSDPPipelayerModelWrapper,
)

def setup_distributed():
    """初始化分布式环境"""
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def main():
    rank, world_size, local_rank = setup_distributed()
    
    # 1. 创建模型并用 FSDP 包装
    model = MyLargeModel()
    model = FSDP(model, device_id=local_rank)
    
    # 2. 创建优化器（必须在 FSDP 包装后）
    optimizer = Adam(model.parameters(), lr=1e-3)
    
    # 3. 训练循环
    for epoch in range(num_epochs):
        for batch in dataloader:
            output = model(batch)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # 4. 每个 epoch 保存检查点
        save_fsdp_model_chunked(
            model=model,
            optimizer=optimizer,
            save_dir=f"./checkpoints/epoch_{epoch}",
            rank=rank,
        )
    
    # 5. 恢复训练（使用 wrapper 方式）
    new_model = MyLargeModel()
    new_model = FSDP(new_model, device_id=local_rank)
    new_optimizer = Adam(new_model.parameters(), lr=1e-3)
    
    wrapper = FSDPPipelayerModelWrapper(
        model=new_model,
        optimizer=new_optimizer,
        chkpt_dir="./checkpoints/epoch_0",
        load_checkpoint=True,
        device=f"cuda:{local_rank}",
        rank=rank,
        world_size=world_size,
    )
    
    # 6. 继续训练（可以在加载过程中开始）
    for batch in dataloader:
        output = wrapper(batch)  # 自动等待需要的参数
        # ... 训练代码
    
    # 清理
    wrapper.cleanup()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

### 运行多 GPU 训练

```bash
# 使用 torchrun（推荐）
torchrun --nproc_per_node=4 train.py

# 或使用 torch.distributed.launch
python -m torch.distributed.launch --nproc_per_node=4 train.py
```

## 注意事项

### 必须遵守的规则

1. **优化器创建顺序**：
   ```python
   # ✓ 正确
   model = FSDP(model)
   optimizer = Adam(model.parameters())
   
   # ✗ 错误
   optimizer = Adam(model.parameters())
   model = FSDP(model)  # 优化器不会正确跟踪 FSDP 参数
   ```

2. **finalize_fsdp_loading 调用**：
   使用 `FSDPPipelinedStateLoader` 时必须调用：
   ```python
   loader = FSDPPipelinedStateLoader(...)
   # 等待所有块
   for i in range(loader.num_chunks):
       loader.wait_for_chunk(i)
   # 必须调用！
   loader.finalize_fsdp_loading()
   ```

3. **所有 ranks 参与**：
   保存和加载时，所有 ranks 都必须调用相应函数：
   ```python
   # 所有 ranks 都调用，即使只有 rank 0 实际保存
   save_fsdp_model_chunked(model, optimizer, "./chkpt", rank=rank)
   ```

### 性能优化建议

1. **chunk 大小调整**：
   - 较大的模型：使用较大的 `target_chunk_bytes`（如 100-200MB）
   - 较小的模型：使用较小的值（如 10-50MB）

2. **copy workers 数量**：
   ```python
   loader = FSDPPipelinedStateLoader(
       ...,
       num_copy_workers=2,  # 可以尝试 1-4
   )
   ```

3. **队列大小**：
   ```python
   loader = FSDPPipelinedStateLoader(
       ...,
       host_queue_maxsize=64,  # 增加以减少 IO 等待
   )
   ```

## 测试

### 运行测试

```bash
# 单 GPU 测试（不实际使用 FSDP）
cd tests
python test_fsdp.py

# 多 GPU 测试
torchrun --nproc_per_node=2 test_fsdp_multi_gpu.py
```

### 运行示例

```bash
# 单 GPU 模式（测试用）
python examples/fsdp_example.py

# 多 GPU 模式
torchrun --nproc_per_node=2 examples/fsdp_example.py
```

## 故障排查

### 常见问题

1. **Q: 加载时卡住不动**
   - A: 确保所有 ranks 都调用了加载函数
   - A: 检查是否调用了 `finalize_fsdp_loading()`

2. **Q: 优化器状态不正确**
   - A: 确保优化器是在 FSDP 包装后创建的
   - A: 检查所有 ranks 是否使用相同的随机种子

3. **Q: 内存不足**
   - A: 减小 `target_chunk_bytes`
   - A: 减小 `host_queue_maxsize`
   - A: 考虑使用 FSDP 的 CPU offload 功能

4. **Q: 保存的文件为空**
   - A: 确保在 rank 0 上调用保存函数
   - A: 检查是否有足够的磁盘空间

## 与标准 Pipelayer 的区别

| 特性 | 标准 Pipelayer | FSDP Pipelayer |
|------|---------------|----------------|
| 适用场景 | 单 GPU | 多 GPU（单机） |
| 状态字典 | 直接访问 | 通过 FSDP API |
| 保存操作 | 单进程 | 多进程协调 |
| 加载操作 | 直接写入 | 累积后批量应用 |
| 广播需求 | 无 | rank 0 广播到其他 ranks |

## 扩展阅读

- [PyTorch FSDP 官方文档](https://pytorch.org/docs/stable/fsdp.html)
- [Pipelayer 原理和设计](../README.md)
- [分布式训练最佳实践](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

## 许可证

本扩展遵循与 Pipelayer 相同的 MIT 许可证。

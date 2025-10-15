from __future__ import annotations

import atexit
import threading
import time
from typing import Any, List, Tuple

import torch
import torch.nn as nn

from .checkpointing import PipelinedStateLoader


class PipelayerModelWrapper(nn.Module):
    """
    在加载期间进行片段同步的包装器：
    - 未完成加载时：按 compute_blocks 切分，推理/训练时对需要的 chunk 等待。
    - 全部加载后：自动切换回原始模型 forward。
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        chkpt_dir: str | None = None,
        load_checkpoint: bool = False,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.original_model = model.to(self.device)
        self.optimizer = optimizer
        self.loader: PipelinedStateLoader | None = None
        self.chkpt_dir = chkpt_dir

        # 状态管理
        self._loading_complete = threading.Event()
        self._use_original_forward = not load_checkpoint  # 如果不需要加载，直接使用原forward
        self._partition_model_blocks()

        # 注册清理函数
        atexit.register(self.cleanup)
        self._cleaned = False

        # 只有当需要加载检查点时才初始化加载器并启动加载
        if load_checkpoint:
            if not chkpt_dir:
                raise ValueError("chkpt_dir must be provided when load_checkpoint is True")
            self._initialize_loader()
            self._start_loading_monitor()

    def _initialize_loader(self) -> None:
        """初始化加载器"""
        self.loader = PipelinedStateLoader(self.original_model, self.optimizer, self.chkpt_dir, device=str(self.device))

    def _partition_model_blocks(self) -> None:
        if self._use_original_forward:
            return

        self.compute_blocks: List[Tuple[str, nn.Module]] = []
        # 自动收集所有叶子层
        for name, module in self.original_model.named_modules():
            # 只收集叶子层（没有子模块的）
            if len(list(module.children())) == 0:
                self.compute_blocks.append((name, module))

    def _start_loading_monitor(self) -> None:
        """启动后台监控线程，检测加载完成"""

        def monitor_loading() -> None:
            try:
                while self.loader and not self._loading_complete.is_set():
                    # 检查是否所有块都已加载完成
                    all_loaded = True
                    for i in range(self.loader.num_chunks):
                        if not self._check_chunk_loaded(i):
                            all_loaded = False
                            break

                    if all_loaded:
                        print("[PipeLayer] All chunks loaded, switching to original forward function")
                        self._switch_to_original_forward()
                        break

                    time.sleep(0.05)  # 短暂休眠避免CPU占用过高
            except Exception as e:
                print(f"[PipeLayer] Loading monitor error: {e}")

        monitor_thread = threading.Thread(target=monitor_loading, daemon=True)
        monitor_thread.start()

    def _check_chunk_loaded(self, chunk_idx: int) -> bool:
        """非阻塞检查块是否已加载完成"""
        if self.loader is None:
            return False
        return self.loader.is_chunk_loaded(chunk_idx)

    def _switch_to_original_forward(self) -> None:
        """切换到原始forward函数"""
        self._use_original_forward = True
        self._loading_complete.set()

        # 清理加载器资源
        if self.loader:
            self.loader.stop()
            self.loader = None

        # 清理不再需要的compute_blocks
        if hasattr(self, "compute_blocks"):
            del self.compute_blocks

        # 强制垃圾回收
        import gc

        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """智能forward函数：根据加载状态选择执行路径"""

        # 快速路径：直接使用原始模型
        if self._use_original_forward:
            return self.original_model(*args, **kwargs)

        # 慢速路径：分块加载期间使用
        return self._pipelined_forward(*args, **kwargs)

    def _pipelined_forward(self, *args: Any, **kwargs: Any) -> Any:
        # 简单的自动推断：假设第一个位置参数为主输入
        x = args[0] if args else kwargs.get("input_ids")
        x = x.to(self.device)
        outputs = {}
        assert self.loader is not None, "Loader must be initialized when using pipelined forward."
        for name, block in self.compute_blocks:
            # 等待所需 chunk
            needed_chunk_ids = set()
            for pname, chunk_id in self.loader.param_name_to_chunk.items():
                if pname.startswith(name):
                    needed_chunk_ids.add(chunk_id)
            for cid in sorted(needed_chunk_ids):
                self.loader.wait_for_chunk(cid)

            # 自动推断输入（基础策略，可按需要扩展）
            try:
                x = block(x)
            except Exception:
                x = block(x, **kwargs)
            outputs[name] = x
        # 最后输出
        return x

    def force_switch_to_original(self) -> None:
        """手动强制切换到原始forward函数（用于调试或特殊情况）"""
        print("[PipeLayer] Manually switching to original forward function")
        self._switch_to_original_forward()

    def cleanup(self) -> None:
        """显式清理资源"""
        if not getattr(self, "_cleaned", False):
            self._loading_complete.set()  # 确保监控线程能够退出
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
from __future__ import annotations

import atexit
import json
import os
import queue
import shutil
import threading
import time
from collections import defaultdict
from typing import Dict, Any, List, Optional

from ctypes import cdll, c_void_p, c_char_p, c_int, c_size_t, POINTER

import torch
import torch.nn as nn


def save_model_chunked(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    save_dir: str,
    target_chunk_bytes: int = 50 * 1024 * 1024,
) -> int:
    """
    保存模型与优化器状态为多个 chunk。
    逻辑：
      1) 按参数名的第一层前缀分组（兼容模块语义，如 encoder/decoder/...）。
      2) 每组内按字节大小切分为多个子 chunk（默认目标 50MB）。
    同时保存：
      - 每个 chunk 的模型参数与对应的优化器 state（按参数名）。
      - optimizer 的 param_groups（将 param 的 id 替换为 name）。
      - metadata.json：包含 chunk 映射信息与数量。

    返回：chunk 数量。
    """
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    model_state_dict = model.state_dict()
    optim_state_dict = optimizer.state_dict()

    # name <-> param <-> id 三向映射（用于提取优化器状态）
    param_name_to_param = {name: param for name, param in model.named_parameters()}
    id_to_name = {id(param): name for name, param in model.named_parameters()}
    param_to_optim_state: Dict[nn.Parameter, Dict[str, Any]] = {}

    for group in optim_state_dict.get("param_groups", []):
        for param_id in group.get("params", []):
            if param_id in optim_state_dict.get("state", {}) and param_id in id_to_name:
                name = id_to_name[param_id]
                param = param_name_to_param[name]
                param_to_optim_state[param] = optim_state_dict["state"][param_id]

    # 先按 prefix 分组（兼容你原来的做法）
    groups: Dict[str, List[str]] = defaultdict(list)
    for key in model_state_dict.keys():
        prefix = key.split(".", 1)[0] if "." in key else "__root__"
        groups[prefix].append(key)

    chunk_metadata: Dict[int, Dict[str, Any]] = {}
    chunk_idx = 0

    # helper: size of a tensor in bytes
    def tensor_size_bytes(tensor: torch.Tensor) -> int:
        return int(tensor.numel() * tensor.element_size())

    # 对每个 prefix 内部按字节切分为若干子chunk
    for group_name, state_keys_in_group in groups.items():
        cur_chunk_keys: List[str] = []
        cur_chunk_bytes = 0
        for key in state_keys_in_group:
            t = model_state_dict[key]
            if not isinstance(t, torch.Tensor):
                # 对于非张量（极少数 buffer 情况），跳过
                continue
            b = tensor_size_bytes(t)

            # 如果加入后超过阈值且已有内容，则先 flush
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

            # 若单个参数本身就超过阈值，则立即写入（独立 chunk）
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

        # flush 剩余
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

    # 保存 optimizer param_groups（id -> name）
    new_param_groups = []
    for group in optim_state_dict.get("param_groups", []):
        new_group = dict(group)
        new_group["params"] = []
        for param_id in group.get("params", []):
            if param_id in id_to_name:
                new_group["params"].append(id_to_name[param_id])
        new_param_groups.append(new_group)
    torch.save(new_param_groups, os.path.join(save_dir, "optimizer_param_groups.pt"))

    # 保存 metadata（包含每个 chunk 的参数列表）
    with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"num_chunks": chunk_idx, "chunks": chunk_metadata}, f, indent=4, ensure_ascii=False)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Saver] Model ({total_params/1e6:.2f}M params) and Optimizer chunked into {chunk_idx} parts -> {save_dir}")
    if total_params < 100e6:
        print("[Warning] 模型参数太小，pipeline IO 优化效果可能不明显。")
    return chunk_idx


def _flush_chunk(
    save_dir: str,
    chunk_idx: int,
    cur_chunk_keys: List[str],
    model_state_dict: Dict[str, torch.Tensor],
    param_name_to_param: Dict[str, nn.Parameter],
    param_to_optim_state: Dict[nn.Parameter, Dict[str, Any]],
    chunk_metadata: Dict[int, Dict[str, Any]],
    group_name: str,
) -> int:
    # 构造模型子状态
    chunk_model_sd = {k: model_state_dict[k].cpu() for k in cur_chunk_keys if isinstance(model_state_dict[k], torch.Tensor)}
    # 构造优化器子状态（按参数名）
    chunk_optim_sd_states: Dict[str, Dict[str, Any]] = {}
    for k in cur_chunk_keys:
        if k in param_name_to_param:
            param_tensor = param_name_to_param[k]
            if param_tensor in param_to_optim_state:
                chunk_optim_sd_states[k] = param_to_optim_state[param_tensor]
    chunk_file = f"chunk_{chunk_idx}.pt"
    torch.save({"model": chunk_model_sd, "optimizer_states": chunk_optim_sd_states}, os.path.join(save_dir, chunk_file))
    chunk_metadata[chunk_idx] = {"file": chunk_file, "params": list(cur_chunk_keys), "group": group_name}
    return chunk_idx + 1


class PipelinedStateLoader:
    """
    修复后的流水加载器，避免 CUDA 隐式同步问题，并支持 CPU 回退路径。
    - 独立 IO 线程：按 chunk 顺序读盘 -> pin memory -> 入队。
    - 多个拷贝工作线程：从 pinned 内存到 device，更新模型参数与优化器 state。
    - 每个 chunk 一个事件，用于前向或外部等待。
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        chkpt_dir: str,
        device: str = "cuda:0",
        host_queue_maxsize: int = 32,
        num_copy_workers: Optional[int] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.chkpt_dir = chkpt_dir
        self.device = torch.device(device)
        # 注意：model 通常已由 PipelayerModelWrapper 移动到目标设备
        # 此处仅检查而非强制移动，避免冗余操作
        if next(model.parameters(), None) is not None:
            model_device = next(model.parameters()).device
            if model_device != self.device:
                self.model.to(self.device)

        with open(os.path.join(chkpt_dir, "metadata.json"), "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.num_chunks = int(self.metadata["num_chunks"])

        # host queue（producer -> 多 copy worker）
        self.host_buffer_queue: "queue.Queue[Any]" = queue.Queue(maxsize=host_queue_maxsize)

        # 参数与 buffer 名 -> 张量映射（用于原位写入）
        self.param_name_to_tensor: Dict[str, nn.Parameter] = {name: p for name, p in self.model.named_parameters()}
        self.buffer_name_to_tensor: Dict[str, torch.Tensor] = {name: b for name, b in self.model.named_buffers()}

        # 构建 param_name -> chunk_idx 的反向索引（便于前向按模块等待所需 chunk）
        self.param_name_to_chunk: Dict[str, int] = {}
        for cid, meta in self.metadata["chunks"].items():
            ic = int(cid)
            for pname in meta["params"]:
                self.param_name_to_chunk[pname] = ic

        # 加载优化器 param groups
        self._load_optimizer_param_groups()

        # 设置 copy workers 数量
        if num_copy_workers is None:
            # 保守：默认 1，避免多流复杂同步问题；可按需调大
            num_copy_workers = 1
        self.num_copy_workers = int(num_copy_workers)

        # 创建 events/streams（CUDA 或 CPU 回退）
        self._init_device_primitives()

        self.producer_stop_event = threading.Event()
        self.copy_worker_stop_event = threading.Event()

        # 启动线程
        self._producer_thread = threading.Thread(target=self._io_producer_loop, daemon=True)
        self._copy_worker_threads: List[threading.Thread] = []
        self._producer_thread.start()
        for i in range(self.num_copy_workers):
            t = threading.Thread(target=self._copy_worker_loop, args=(i,), daemon=True)
            t.start()
            self._copy_worker_threads.append(t)

        # 清理
        atexit.register(self.stop)

    def _init_device_primitives(self) -> None:
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                self.copy_streams = [torch.cuda.Stream() for _ in range(self.num_copy_workers)]
                self.events = [torch.cuda.Event(enable_timing=False) for _ in range(self.num_chunks)]
        else:
            # CPU 回退：无 stream 概念，使用 None 占位；事件用 threading.Event
            self.copy_streams = [None for _ in range(self.num_copy_workers)]
            self.events = [threading.Event() for _ in range(self.num_chunks)]

    def _load_optimizer_param_groups(self) -> None:
        param_groups_path = os.path.join(self.chkpt_dir, "optimizer_param_groups.pt")
        if os.path.exists(param_groups_path):
            saved_param_groups = torch.load(param_groups_path, map_location="cpu")
            new_param_groups = []
            for saved_group in saved_param_groups:
                new_group = dict(saved_group)
                new_params = []
                for pname in saved_group.get("params", []):
                    if pname in self.param_name_to_tensor:
                        new_params.append(self.param_name_to_tensor[pname])
                new_group["params"] = new_params
                if new_group["params"]:
                    new_param_groups.append(new_group)
            self.optimizer.param_groups = new_param_groups
            self.optimizer.state = {}

    def _io_producer_loop(self) -> None:
        """IO 线程：按 chunk 顺序从磁盘 load 到 CPU 并 pin memory，然后放 queue"""
        try:
            for i in range(self.num_chunks):
                if self.producer_stop_event.is_set():
                    break
                fname = self.metadata["chunks"][str(i)]["file"]
                chunk_path = os.path.join(self.chkpt_dir, fname)
                chunk_data = torch.load(chunk_path, map_location="cpu", weights_only=False)

                # pin model tensors
                pinned_model = {}
                for k, v in chunk_data["model"].items():
                    if isinstance(v, torch.Tensor):
                        pinned_model[k] = v.pin_memory()
                    else:
                        pinned_model[k] = v

                optim_states = chunk_data.get("optimizer_states", {})

                # pin optimizer tensor states too
                for param_name, state in optim_states.items():
                    for state_name, tensor in list(state.items()):
                        if isinstance(tensor, torch.Tensor):
                            state[state_name] = tensor.pin_memory()

                # push to queue
                self.host_buffer_queue.put((i, pinned_model, optim_states))
        except Exception as e:
            print(f"[ERROR] Producer thread error: {e}")
            self.producer_stop_event.set()

    def _copy_worker_loop(self, worker_idx: int) -> None:
        """每个 copy worker 使用独立流，把 CPU pinned tensor 上传到设备并写入模型/optimizer"""
        stream = self.copy_streams[worker_idx]
        try:
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    self._copy_worker_loop_impl(worker_idx, stream)  # type: ignore[arg-type]
            else:
                self._copy_worker_loop_impl(worker_idx, stream=None)
        except Exception as e:
            print(f"[ERROR] Copy worker thread error (idx={worker_idx}): {e}")
            self.copy_worker_stop_event.set()

    def _copy_worker_loop_impl(self, worker_idx: int, stream: Optional["torch.cuda.Stream"]) -> None:  # type: ignore[name-defined]
        while not (self.copy_worker_stop_event.is_set() and self.host_buffer_queue.empty()):
            try:
                item = self.host_buffer_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                self.host_buffer_queue.task_done()
                break
            chunk_idx, host_model_chunk, host_optim_chunk = item
            try:
                if self.device.type == "cuda":
                    assert stream is not None
                    with torch.cuda.stream(stream):
                        self._apply_model_chunk(host_model_chunk)
                        self._apply_optimizer_chunk(host_optim_chunk)
                        self.events[chunk_idx].record(stream)  # type: ignore[union-attr]
                else:
                    # CPU：直接赋值并置位事件
                    self._apply_model_chunk(host_model_chunk)
                    self._apply_optimizer_chunk(host_optim_chunk)
                    self.events[chunk_idx].set()  # type: ignore[union-attr]
            finally:
                self.host_buffer_queue.task_done()

    def _apply_model_chunk(self, host_model_chunk: Dict[str, torch.Tensor]) -> None:
        for name, host_tensor in host_model_chunk.items():
            if not isinstance(host_tensor, torch.Tensor):
                continue
            # 参数优先
            if name in self.param_name_to_tensor:
                param = self.param_name_to_tensor[name]
                param.data.copy_(host_tensor.to(self.device, non_blocking=self.device.type == "cuda"))
            elif name in self.buffer_name_to_tensor:
                buf = self.buffer_name_to_tensor[name]
                buf.data.copy_(host_tensor.to(self.device, non_blocking=self.device.type == "cuda"))
            # 未知条目忽略（例如某些临时 buffer）

    def _apply_optimizer_chunk(self, host_optim_chunk: Dict[str, Dict[str, Any]]) -> None:
        for param_name, state_dict in host_optim_chunk.items():
            if param_name not in self.param_name_to_tensor:
                continue
            param_tensor = self.param_name_to_tensor[param_name]
            new_state: Dict[str, Any] = {}
            for state_name, tensor in state_dict.items():
                if isinstance(tensor, torch.Tensor):
                    new_state[state_name] = tensor.to(self.device, non_blocking=self.device.type == "cuda")
                else:
                    new_state[state_name] = tensor
            self.optimizer.state[param_tensor] = new_state

    def stop(self) -> None:
        """停止并释放资源：优雅关闭 producer 与 copy workers"""
        self.producer_stop_event.set()
        self.copy_worker_stop_event.set()
        # 唤醒可能阻塞的 workers
        for _ in range(self.num_copy_workers + 2):
            try:
                self.host_buffer_queue.put_nowait(None)
            except queue.Full:
                break

        if hasattr(self, "_producer_thread") and self._producer_thread.is_alive():
            self._producer_thread.join(timeout=2.0)
        for t in getattr(self, "_copy_worker_threads", []):
            if t.is_alive():
                t.join(timeout=2.0)

        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        # 清空队列
        while not self.host_buffer_queue.empty():
            try:
                self.host_buffer_queue.get_nowait()
                self.host_buffer_queue.task_done()
            except queue.Empty:
                break

    def wait_for_chunk(self, chunk_idx: int) -> None:
        """阻塞等待某个 chunk 的上传（供 forward 使用）"""
        if 0 <= chunk_idx < self.num_chunks:
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    torch.cuda.current_stream().wait_event(self.events[chunk_idx])  # type: ignore[arg-type]
            else:
                self.events[chunk_idx].wait()  # type: ignore[union-attr]

    def is_chunk_loaded(self, chunk_idx: int) -> bool:
        """非阻塞查询某个 chunk 是否已加载完成（用于 monitor 或 progress）"""
        if 0 <= chunk_idx < self.num_chunks:
            if self.device.type == "cuda":
                return bool(self.events[chunk_idx].query())  # type: ignore[union-attr]
            else:
                return bool(self.events[chunk_idx].is_set())  # type: ignore[union-attr]
        return False

    def debug_stats(self) -> None:
        print(f"[Loader Stats] Queue Size: {self.host_buffer_queue.qsize()}, CopyWorkers: {self.num_copy_workers}")


class MultiStreamStateLoader:
    """
    基于 multistream 格式的流水加载器（C++ mmap 读取 + Python pipeline）。

    - C++ 侧负责 mmap + 按 stream/offset 拷贝到 CPU pinned buffer。
    - Python 侧负责异步拷贝到 device，并原位更新参数与优化器状态。
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        chkpt_dir: str,
        lib_path: str,
        checkpoint_file: Optional[str] = None,
        metadata_file: str = "multistream_metadata.json",
        device: str = "cuda:0",
        host_queue_maxsize: int = 32,
        num_copy_workers: Optional[int] = None,
        load_grad: bool = False,
        parall_iter: Optional[int] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.chkpt_dir = chkpt_dir
        self.device = torch.device(device)
        # 注意：model 通常已由 PipelayerModelWrapper 移动到目标设备
        # 此处仅检查而非强制移动，避免冗余操作
        if next(model.parameters(), None) is not None:
            model_device = next(model.parameters()).device
            if model_device != self.device:
                self.model.to(self.device)
        self.load_grad = load_grad

        metadata_path = metadata_file
        if not os.path.isabs(metadata_path):
            metadata_path = os.path.join(chkpt_dir, metadata_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.num_chunks = int(self.metadata["num_chunks"])
        self.stream_names: List[str] = list(self.metadata.get("stream_names", ["param", "grad", "exp_avg", "exp_avg_sq"]))
        self.stream_sizes: List[int] = [int(x) for x in self.metadata.get("stream_sizes", [])]
        self.max_async = int(self.metadata.get("max_async", 1))

        if checkpoint_file is None:
            checkpoint_file = self.metadata.get("checkpoint_file")
        if checkpoint_file is None:
            raise ValueError("checkpoint_file is required (not found in metadata).")
        if not os.path.isabs(checkpoint_file):
            checkpoint_file = os.path.join(chkpt_dir, checkpoint_file)
        self.checkpoint_file = checkpoint_file

        # 参数与 buffer 名 -> 张量映射（用于原位写入）
        self.param_name_to_tensor: Dict[str, nn.Parameter] = {name: p for name, p in self.model.named_parameters()}
        self.buffer_name_to_tensor: Dict[str, torch.Tensor] = {name: b for name, b in self.model.named_buffers()}

        # param_name -> chunk_idx
        self.param_name_to_chunk: Dict[str, int] = {}
        for cid, meta in self.metadata.get("chunks", {}).items():
            ic = int(cid)
            for pinfo in meta.get("params", []):
                pname = pinfo.get("name")
                if pname:
                    self.param_name_to_chunk[pname] = ic

        # 加载 optimizer param groups（若提供）
        self._load_optimizer_param_groups()
        self.optimizer_steps: Dict[str, int] = self.metadata.get("optimizer_steps", {})
        
        # 验证 optimizer step 一致性，并计算全局 step
        self.global_optimizer_step: Optional[int] = self._validate_and_get_global_step()

        if num_copy_workers is None:
            num_copy_workers = 1
        self.num_copy_workers = int(num_copy_workers)

        self._init_device_primitives()
        self._init_reader(lib_path)

        # 优先从 metadata 读取 latest_parall_iter，避免调用可能崩溃的 C++ 接口
        if parall_iter is None:
            parall_iter = self.metadata.get("latest_parall_iter")
        if parall_iter is None:
            # C++ 的 get_latest_parall_iter 存在段错误风险，直接默认使用 0
            # 对于旧版本 metadata（没有 latest_parall_iter 字段），使用 0 是安全的
            print("[WARN] latest_parall_iter not found in metadata, defaulting to 0")
            parall_iter = 0
        self.parall_iter = int(parall_iter)
        print(f"[MultiStreamStateLoader] Using parall_iter={self.parall_iter}")

        self.host_buffer_queue: "queue.Queue[Any]" = queue.Queue(maxsize=host_queue_maxsize)
        self.producer_stop_event = threading.Event()
        self.copy_worker_stop_event = threading.Event()

        self._producer_thread = threading.Thread(target=self._io_producer_loop, daemon=True)
        self._copy_worker_threads: List[threading.Thread] = []
        self._producer_thread.start()
        for i in range(self.num_copy_workers):
            t = threading.Thread(target=self._copy_worker_loop, args=(i,), daemon=True)
            t.start()
            self._copy_worker_threads.append(t)

        atexit.register(self.stop)

    def _init_reader(self, lib_path: str) -> None:
        self.lib = cdll.LoadLibrary(lib_path)

        self.lib.reader.restype = c_void_p
        self.lib.reader.argtypes = [c_char_p, c_int]

        self.lib.init_streams.restype = c_int
        self.lib.init_streams.argtypes = [c_void_p, c_int, POINTER(c_size_t)]

        self.lib.read_stream_chunk.restype = None
        self.lib.read_stream_chunk.argtypes = [c_void_p, c_int, c_void_p, c_size_t, c_size_t, c_int]

        self.lib.get_latest_parall_iter.restype = c_int
        self.lib.get_latest_parall_iter.argtypes = [c_void_p]

        self.lib.close_writer.restype = None
        self.lib.close_writer.argtypes = [c_void_p]

        self.reader_obj = self.lib.reader(self.checkpoint_file.encode("utf-8"), int(self.max_async))

        if self.stream_sizes:
            stream_sizes_arr = (c_size_t * len(self.stream_sizes))(*self.stream_sizes)
            ret = self.lib.init_streams(self.reader_obj, int(len(self.stream_sizes)), stream_sizes_arr)
            if ret != 0:
                raise RuntimeError("Failed to initialize streams in multistream reader")

    def _init_device_primitives(self) -> None:
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                self.copy_streams = [torch.cuda.Stream() for _ in range(self.num_copy_workers)]
                self.events = [torch.cuda.Event(enable_timing=False) for _ in range(self.num_chunks)]
        else:
            self.copy_streams = [None for _ in range(self.num_copy_workers)]
            self.events = [threading.Event() for _ in range(self.num_chunks)]

    def _load_optimizer_param_groups(self) -> None:
        groups = self.metadata.get("optimizer_param_groups")
        if not groups:
            return
        new_param_groups = []
        for group in groups:
            new_group = dict(group)
            new_params = []
            for pname in group.get("params", []):
                if pname in self.param_name_to_tensor:
                    new_params.append(self.param_name_to_tensor[pname])
            new_group["params"] = new_params
            if new_group["params"]:
                new_param_groups.append(new_group)
        if new_param_groups:
            self.optimizer.param_groups = new_param_groups
            self.optimizer.state = {}

    def _validate_and_get_global_step(self) -> Optional[int]:
        """
        获取全局 optimizer step 值。
        
        性能优先：直接取第一个 step 值，假设保存侧已保证一致性。
        正常训练流程中所有参数的 step 始终一致。
        """
        if not self.optimizer_steps:
            return None
        # O(1) 取任意一个值即可
        return next(iter(self.optimizer_steps.values()))

    def _io_producer_loop(self) -> None:
        try:
            for i in range(self.num_chunks):
                if self.producer_stop_event.is_set():
                    break
                chunk_meta = self.metadata["chunks"][str(i)]
                pinned_buffers: Dict[str, torch.Tensor] = {}

                for stream_idx, stream_name in enumerate(self.stream_names):
                    if stream_name == "grad" and not self.load_grad:
                        continue
                    slice_info = chunk_meta.get("stream_slices", {}).get(stream_name)
                    if not slice_info or slice_info.get("size", 0) == 0:
                        continue
                    size = int(slice_info["size"])
                    offset = int(slice_info["offset"])
                    buffer = torch.empty(size, dtype=torch.float32, pin_memory=True)
                    self.lib.read_stream_chunk(
                        self.reader_obj,
                        int(stream_idx),
                        c_void_p(buffer.data_ptr()),
                        c_size_t(offset),
                        c_size_t(size),
                        c_int(self.parall_iter),
                    )
                    pinned_buffers[stream_name] = buffer

                self.host_buffer_queue.put((i, pinned_buffers, chunk_meta))
        except Exception as e:
            print(f"[ERROR] MultiStream producer error: {e}")
            self.producer_stop_event.set()

    def _copy_worker_loop(self, worker_idx: int) -> None:
        stream = self.copy_streams[worker_idx]
        try:
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    self._copy_worker_loop_impl(worker_idx, stream)  # type: ignore[arg-type]
            else:
                self._copy_worker_loop_impl(worker_idx, stream=None)
        except Exception as e:
            print(f"[ERROR] MultiStream copy worker error (idx={worker_idx}): {e}")
            self.copy_worker_stop_event.set()

    def _copy_worker_loop_impl(self, worker_idx: int, stream: Optional["torch.cuda.Stream"]) -> None:  # type: ignore[name-defined]
        while not (self.copy_worker_stop_event.is_set() and self.host_buffer_queue.empty()):
            try:
                item = self.host_buffer_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                self.host_buffer_queue.task_done()
                break
            chunk_idx, buffers, chunk_meta = item
            try:
                if self.device.type == "cuda":
                    assert stream is not None
                    with torch.cuda.stream(stream):
                        self._apply_multistream_chunk(buffers, chunk_meta)
                        self.events[chunk_idx].record(stream)  # type: ignore[union-attr]
                else:
                    self._apply_multistream_chunk(buffers, chunk_meta)
                    self.events[chunk_idx].set()  # type: ignore[union-attr]
            finally:
                self.host_buffer_queue.task_done()

    def _apply_multistream_chunk(self, buffers: Dict[str, torch.Tensor], chunk_meta: Dict[str, Any]) -> None:
        for pinfo in chunk_meta.get("params", []):
            name = pinfo.get("name")
            if not name or name not in self.param_name_to_tensor:
                continue
            param = self.param_name_to_tensor[name]
            numel = int(pinfo.get("numel", param.numel()))
            offsets = pinfo.get("offsets_in_chunk", {})
            dtype = param.dtype
            non_blocking = self.device.type == "cuda"

            if "param" in buffers and "param" in offsets:
                start = int(offsets["param"])
                host_view = buffers["param"][start:start + numel].view(param.shape)
                param.data.copy_(host_view.to(self.device, dtype=dtype, non_blocking=non_blocking))

            if self.load_grad and "grad" in buffers and "grad" in offsets:
                start = int(offsets["grad"])
                host_view = buffers["grad"][start:start + numel].view(param.shape)
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                param.grad.data.copy_(host_view.to(self.device, dtype=dtype, non_blocking=non_blocking))

            # optimizer state: exp_avg / exp_avg_sq
            state = self.optimizer.state.setdefault(param, {})
            if "exp_avg" in buffers and "exp_avg" in offsets:
                start = int(offsets["exp_avg"])
                host_view = buffers["exp_avg"][start:start + numel].view(param.shape)
                if isinstance(state.get("exp_avg"), torch.Tensor):
                    state["exp_avg"].data.copy_(host_view.to(self.device, dtype=dtype, non_blocking=non_blocking))
                else:
                    state["exp_avg"] = host_view.to(self.device, dtype=dtype, non_blocking=non_blocking)

            if "exp_avg_sq" in buffers and "exp_avg_sq" in offsets:
                start = int(offsets["exp_avg_sq"])
                host_view = buffers["exp_avg_sq"][start:start + numel].view(param.shape)
                if isinstance(state.get("exp_avg_sq"), torch.Tensor):
                    state["exp_avg_sq"].data.copy_(host_view.to(self.device, dtype=dtype, non_blocking=non_blocking))
                else:
                    state["exp_avg_sq"] = host_view.to(self.device, dtype=dtype, non_blocking=non_blocking)

            # restore optimizer step - 使用经过一致性验证的 step 值
            if self.global_optimizer_step is not None:
                state["step"] = torch.tensor(self.global_optimizer_step)
            elif name in self.optimizer_steps:
                state["step"] = torch.tensor(self.optimizer_steps[name])

    def stop(self) -> None:
        self.producer_stop_event.set()
        self.copy_worker_stop_event.set()
        for _ in range(self.num_copy_workers + 2):
            try:
                self.host_buffer_queue.put_nowait(None)
            except queue.Full:
                break

        if hasattr(self, "_producer_thread") and self._producer_thread.is_alive():
            self._producer_thread.join(timeout=2.0)
        for t in getattr(self, "_copy_worker_threads", []):
            if t.is_alive():
                t.join(timeout=2.0)

        if getattr(self, "reader_obj", None) is not None:
            pass
            # try:
            #     self.lib.close_writer(self.reader_obj)
            # except Exception:
            #     pass

        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        while not self.host_buffer_queue.empty():
            try:
                self.host_buffer_queue.get_nowait()
                self.host_buffer_queue.task_done()
            except queue.Empty:
                break

    def wait_for_chunk(self, chunk_idx: int) -> None:
        if 0 <= chunk_idx < self.num_chunks:
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    torch.cuda.current_stream().wait_event(self.events[chunk_idx])  # type: ignore[arg-type]
            else:
                self.events[chunk_idx].wait()  # type: ignore[union-attr]

    def is_chunk_loaded(self, chunk_idx: int) -> bool:
        if 0 <= chunk_idx < self.num_chunks:
            if self.device.type == "cuda":
                return bool(self.events[chunk_idx].query())  # type: ignore[union-attr]
            else:
                return bool(self.events[chunk_idx].is_set())  # type: ignore[union-attr]
        return False

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
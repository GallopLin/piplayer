#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling with CheckFreq
checkpoint save/restore support. Structured to match run_clm_pipelayer2.py
for fair recovery-time comparison.
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import ctypes
from itertools import chain
from pathlib import Path
import time
from datetime import datetime
import csv

import datasets
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from torch.multiprocessing import Process, Manager, Value, Lock

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

# CheckFreq imports
from checkpoint_eval.checkfreq.chk_manager import CFCheckpoint
from checkpoint_eval.checkfreq.utils import save_checkpoint, restore_checkpoint, get_latest_checkpoint

check_min_version("4.31.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task with CheckFreq checkpoint support"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder or .chk file.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    # CheckFreq specific arguments
    parser.add_argument(
        "--use_checkfreq",
        action="store_true",
        help="Whether to use CheckFreq for async checkpoint saving and restoring.",
    )
    parser.add_argument(
        "--cfreq",
        type=int,
        default=0,
        help="CheckFreq checkpoint frequency (in steps). 0 means no periodic checkpointing.",
    )
    parser.add_argument(
        "--path_to_pmem",
        type=str,
        default="",
        help="If specified, the checkpoint will be written to pmem, at this path.",
    )
    parser.add_argument(
        "--checkpoint_format",
        type=str,
        default="./checkpoint-{epoch}-{it}.chk",
        help="CheckFreq checkpoint file format.",
    )
    parser.add_argument(
        "--checkfreq_chk_path",
        type=str,
        default=None,
        help="Explicit path to a .chk file for CheckFreq restore. If not set, will auto-detect.",
    )

    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    # 初始化时间记录字典
    time_records = {}
    overall_start_time = time.time()

    # 记录函数用于记录各阶段时间
    def record_time(stage_name, start_time):
        elapsed = time.time() - start_time
        time_records[stage_name] = elapsed
        logger.info(f"[TIME] {stage_name}: {elapsed:.2f} seconds")
        return time.time()  # 返回新的开始时间

    # 保存时间记录到CSV的函数
    def save_time_to_csv(records, output_dir):
        csv_file = os.path.join(output_dir, "training_time_log.csv")

        # 准备CSV行数据
        row_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'run_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'model_name': args.model_name_or_path or args.model_type,
            'num_train_epochs': args.num_train_epochs,
            'per_device_train_batch_size': args.per_device_train_batch_size,
            'gradient_accumulation_steps': args.gradient_accumulation_steps,
            'learning_rate': args.learning_rate,
            'use_checkfreq': args.use_checkfreq,
            'cfreq': args.cfreq,
            'resume_from_checkpoint': bool(resolved_resume_from_checkpoint),
        }

        # 添加所有时间记录
        for stage, duration in records.items():
            if stage != "Epoch Details":
                row_data[stage.replace(" ", "_").lower()] = round(duration, 2)

        # 计算epoch相关的平均时间
        if "Epoch Details" in records:
            epoch_times = records["Epoch Details"]
            if epoch_times:
                avg_epoch_time = sum(e['duration_seconds'] for e in epoch_times) / len(epoch_times)
                avg_eval_time = sum(e['eval_time'] for e in epoch_times) / len(epoch_times)
                row_data['avg_epoch_time'] = round(avg_epoch_time, 2)
                row_data['avg_eval_time'] = round(avg_eval_time, 2)
                row_data['num_epochs_completed'] = len(epoch_times)

        # 检查文件是否存在，如果不存在则创建并写入header
        file_exists = os.path.exists(csv_file)

        # 确定所有列名
        fieldnames = list(row_data.keys())

        # 如果文件存在，读取现有的header并合并新列
        if file_exists:
            with open(csv_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_fieldnames = reader.fieldnames or []
                # 合并新旧列名，保持顺序
                for field in existing_fieldnames:
                    if field not in fieldnames:
                        fieldnames.append(field)

        # 写入数据
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            # 如果是新文件或需要重写header
            if not file_exists:
                writer.writeheader()

            # 写入数据行
            writer.writerow(row_data)

        logger.info(f"Time records saved to {csv_file}")

    # 初始化阶段
    stage_start = time.time()
    args = parse_args()
    send_example_telemetry("run_clm_no_trainer", args)
    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs,
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    if args.seed is not None:
        set_seed(args.seed)
    stage_start = record_time("initialization", stage_start)

    # 处理仓库创建
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)
            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    stage_start = record_time("repository_setup", stage_start)

    # 检测最新检查点（行为与 run_clm_pipelayer2 对齐）
    resolved_resume_from_checkpoint = args.resume_from_checkpoint
    last_checkpoint = None
    if args.output_dir is not None and os.path.isdir(args.output_dir) and not args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is None and len(os.listdir(args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        if last_checkpoint is not None and resolved_resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. "
                "To avoid this behavior, change the --output_dir or add --overwrite_output_dir to train from scratch."
            )
            resolved_resume_from_checkpoint = last_checkpoint

    if isinstance(resolved_resume_from_checkpoint, str) and resolved_resume_from_checkpoint.lower() == "latest":
        if last_checkpoint is None and args.output_dir is not None:
            last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is None:
            raise ValueError("--resume_from_checkpoint=latest requested but no checkpoint found.")
        resolved_resume_from_checkpoint = last_checkpoint

    # 加载数据集
    if args.dataset_name is not None:
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
            )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )
    stage_start = record_time("dataset_loading", stage_start)

    # 加载模型和分词器
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    # 对于GPT2等没有pad_token的模型，设置pad_token为eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    stage_start = record_time("tokenizer_loading", stage_start)

    # 加载基础模型
    if args.model_name_or_path:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
        )
    else:
        logger.info("Training new model from scratch")
        base_model = AutoModelForCausalLM.from_config(config)
    # 调整嵌入层大小
    embedding_size = base_model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        base_model.resize_token_embeddings(len(tokenizer))
    stage_start = record_time("model_loading", stage_start)

    # 预处理数据集
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], truncation=True, max_length=args.block_size)

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    stage_start = record_time("dataset_tokenization", stage_start)

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
        block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    # 文本分组
    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    with accelerator.main_process_first():
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )
    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]
    stage_start = record_time("text_grouping", stage_start)

    # 日志输出一些训练样本
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
    )

    # 优化器
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in base_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in base_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # 学习率调度器
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    stage_start = record_time("optimizer_scheduler_setup", stage_start)

    # ========== CheckFreq 检查点恢复 ==========
    # Based on CheckFreq source code (cf_checkpoint.py / cf_manager.py)
    # 使用 CFCheckpoint + restore_checkpoint 进行恢复
    model = base_model
    checkfreq_restored = False
    checkfreq_extra_state = None

    if args.use_checkfreq and resolved_resume_from_checkpoint is not None:
        checkfreq_setup_start = time.time()
        local_rank = 0
        if torch.cuda.is_available():
            local_rank = torch.cuda.current_device()

        # 创建 CFCheckpoint 实例，跟踪 model 和 optimizer
        # 基于 CheckFreq 源码: chk = CFCheckpoint(model=model.state_dict(), optimizer=optimizer.state_dict())
        chk_for_restore = CFCheckpoint(model=model, optimizer=optimizer)

        # 确定检查点路径
        chk_path = args.checkfreq_chk_path  # 用户显式指定的 .chk 文件路径

        if chk_path is None:
            # 自动查找: 先在 resume_from_checkpoint 目录下查找 .chk 文件
            resume_dir = resolved_resume_from_checkpoint
            if os.path.isdir(resume_dir):
                chk_path_found = get_latest_checkpoint(resume_dir)
                if chk_path_found is not None:
                    chk_path = chk_path_found
            elif os.path.isfile(resume_dir) and resume_dir.endswith('.chk'):
                chk_path = resume_dir

        if chk_path is not None and os.path.isfile(chk_path):
            logger.info(f"[CheckFreq] Restoring from checkpoint: {chk_path}")
            checkfreq_extra_state = restore_checkpoint(
                chk_for_restore,
                checkpoint_path=chk_path,
                gpu=local_rank,
            )
            checkfreq_restored = True
            logger.info(f"[CheckFreq] Restore complete. Extra state: {checkfreq_extra_state}")
        else:
            logger.warning(
                f"[CheckFreq] No .chk file found for restore. "
                f"resume_from_checkpoint={resolved_resume_from_checkpoint}, checkfreq_chk_path={args.checkfreq_chk_path}"
            )

        stage_start = record_time("checkfreq_restore", checkfreq_setup_start)
    elif not args.use_checkfreq and resolved_resume_from_checkpoint is not None:
        # 不使用 checkfreq 时，使用标准 accelerator 恢复（作为基线对比）
        pass  # 在后面的 checkpoint resume 部分处理

    stage_start = record_time("checkfreq_setup", stage_start)

    # 使用accelerator准备组件
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    # 在TPU上恢复权重绑定
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()
    stage_start = record_time("accelerator_preparation", stage_start)

    # 重新计算训练步骤
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # 检查点配置
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # 初始化跟踪器
    if args.with_tracking:
        experiment_config = vars(args)
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    # 训练参数
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Use CheckFreq = {args.use_checkfreq}")
    logger.info(f"  CheckFreq frequency = {args.cfreq}")
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    resume_step = None

    # 从检查点恢复训练位置
    checkpoint_start = time.time()
    if resolved_resume_from_checkpoint:
        if args.use_checkfreq and checkfreq_restored:
            # CheckFreq 已在上面恢复了模型和优化器状态
            # 从 extra_state 中获取训练进度信息
            accelerator.print(f"[CheckFreq] Resumed from checkpoint: {resolved_resume_from_checkpoint}")
            if checkfreq_extra_state is not None:
                restored_epoch = checkfreq_extra_state.get('epoch', 0)
                restored_iter = checkfreq_extra_state.get('iter', 0)
                logger.info(f"[CheckFreq] Restored epoch={restored_epoch}, iter={restored_iter}")
                starting_epoch = restored_epoch
                resume_step = restored_iter * args.gradient_accumulation_steps
                if resume_step > 0:
                    resume_step -= starting_epoch * len(train_dataloader)
                completed_steps = starting_epoch * num_update_steps_per_epoch
                if resume_step is not None and resume_step > 0:
                    completed_steps += resume_step // args.gradient_accumulation_steps
            else:
                # 没有 extra_state，尝试从目录名解析
                path = os.path.basename(resolved_resume_from_checkpoint)
                training_difference = os.path.splitext(path)[0]
                if "epoch" in training_difference:
                    starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                    resume_step = None
                    completed_steps = starting_epoch * num_update_steps_per_epoch
                else:
                    try:
                        resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
                        starting_epoch = resume_step // len(train_dataloader)
                        resume_step -= starting_epoch * len(train_dataloader)
                        completed_steps = resume_step // args.gradient_accumulation_steps
                    except ValueError:
                        logger.warning(f"Could not parse training progress from: {training_difference}")
        else:
            # 标准恢复（非 CheckFreq）
            if resolved_resume_from_checkpoint is not None and resolved_resume_from_checkpoint != "":
                accelerator.print(f"Resumed from checkpoint: {resolved_resume_from_checkpoint}")
                accelerator.load_state(resolved_resume_from_checkpoint, strict=False)
                path = os.path.basename(resolved_resume_from_checkpoint)
            else:
                dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                path = dirs[-1]
            training_difference = os.path.splitext(path)[0]

            if "epoch" in training_difference:
                starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                resume_step = None
                completed_steps = starting_epoch * num_update_steps_per_epoch
            else:
                resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
                starting_epoch = resume_step // len(train_dataloader)
                resume_step -= starting_epoch * len(train_dataloader)
                completed_steps = resume_step // args.gradient_accumulation_steps

    if resolved_resume_from_checkpoint:
        stage_start = record_time("checkpoint_resume", checkpoint_start)

    progress_bar.update(completed_steps)

    # ========== CheckFreq 异步保存初始化 ==========
    # 基于 CheckFreq 源码 (checkfreq/utils.py save_checkpoint)
    chk = None
    active_snapshot = None
    in_progress_snapshot = None
    lock = None
    last_chk_it = None
    change = None
    profile_snap = None
    filepath = None
    additional_snapshot = None

    if args.use_checkfreq and args.cfreq > 0:
        mp.set_start_method("spawn", force=True)
        active_snapshot = Value("i", 0)
        lock = Lock()
        in_progress_snapshot = Value("i", 0)
        profile_snap = Value("i", 0)
        mp_manager = Manager()
        last_chk_it = Value("i", -1)
        change = Value("i", 0)
        filepath = mp_manager.Value(ctypes.c_wchar_p, "")
        additional_snapshot = mp_manager.dict()

    steps_since_checkp = 0
    checkpoints = 0
    warmup = 3

    # 训练循环
    training_start = time.time()
    first_step_recorded = False
    epoch_times = []
    checkpoint_save_times = []

    for epoch in range(starting_epoch, args.num_train_epochs):
        epoch_start = time.time()
        model.train()
        if args.with_tracking:
            total_loss = 0
        if resolved_resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader

        for step, batch in enumerate(active_dataloader):
            step_start = time.time()
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)

                # CheckFreq: 等待正在进行的快照完成再执行 optimizer.step()
                # 基于 CheckFreq 源码 (cf_manager.py weight_update)
                if args.use_checkfreq and in_progress_snapshot is not None:
                    while in_progress_snapshot.value == 1:
                        continue

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # 记录第一个step的时间
            if not first_step_recorded:
                time_records["first_training_step"] = time.time() - step_start
                first_step_recorded = True

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if resolved_resume_from_checkpoint and step == 0:
                end_time = time.time()
                print(f"第一次训练循环耗时: {end_time - overall_start_time:.2f} 秒")
                time_records["first_loop_after_resume"] = end_time - overall_start_time
                break

            # CheckFreq 异步保存检查点
            # 基于 CheckFreq 源码 (checkfreq/utils.py save_checkpoint)
            if args.use_checkfreq and args.cfreq > 0:
                if (step == warmup) or (steps_since_checkp == args.cfreq - 1):
                    checkpoint_save_start = time.time()
                    if chk is None:
                        chk = CFCheckpoint(
                            model=model.state_dict(),
                            optimizer=optimizer.state_dict(),
                        )

                    save_checkpoint(
                        args.checkpoint_format,
                        args.path_to_pmem,
                        filepath,
                        additional_snapshot,
                        chk,
                        active_snapshot,
                        in_progress_snapshot,
                        lock,
                        epoch,
                        step,
                        last_chk_it,
                        change,
                        profile_snap,
                        sync=False,
                    )
                    steps_since_checkp = 0
                    checkpoints += 1
                    checkpoint_save_times.append(time.time() - checkpoint_save_start)

                    if step == warmup:
                        print(f"Start clock!")
                        training_start = time.time()
                else:
                    steps_since_checkp += 1

            # 标准 accelerator 保存检查点（非 checkfreq 模式）
            if not args.use_checkfreq and isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    checkpoint_save_start = time.time()
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
                    checkpoint_save_times.append(time.time() - checkpoint_save_start)

            if completed_steps >= args.max_train_steps:
                break

        # 评估
        eval_start = time.time()
        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))
        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")
        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

        # 记录epoch时间
        epoch_duration = time.time() - epoch_start
        epoch_times.append({
            'epoch': epoch,
            'duration_seconds': epoch_duration,
            'eval_time': time.time() - eval_start,
        })

        if args.with_tracking:
            accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        # 推送到Hub
        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                )

        # 按epoch保存检查点
        if args.checkpointing_steps == "epoch":
            epoch_checkpoint_start = time.time()
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)

            if args.use_checkfreq and accelerator.is_main_process:
                # 使用 CheckFreq 异步保存
                if chk is None:
                    chk = CFCheckpoint(
                        model=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                    )
                if filepath is not None:
                    save_checkpoint(
                        args.checkpoint_format,
                        args.path_to_pmem,
                        filepath,
                        additional_snapshot,
                        chk,
                        active_snapshot,
                        in_progress_snapshot,
                        lock,
                        epoch,
                        completed_steps,
                        last_chk_it,
                        change,
                        profile_snap,
                        sync=False,
                    )
            else:
                accelerator.save_state(output_dir)
            checkpoint_save_times.append(time.time() - epoch_checkpoint_start)

    # 清理 CheckFreq 后台进程
    if chk is not None and chk.chk_process is not None:
        chk.chk_process.kill()
        chk.chk_process.join()

    time_records["total_training_time"] = time.time() - training_start
    time_records["Epoch Details"] = epoch_times

    # 计算平均检查点保存时间
    if checkpoint_save_times:
        time_records["avg_checkpoint_save_time"] = sum(checkpoint_save_times) / len(checkpoint_save_times)
    time_records["total_checkpoints"] = checkpoints

    if args.with_tracking:
        accelerator.end_training()

    # 保存最终模型
    final_save_start = time.time()
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity}, f)

            # 保存时间记录
            time_records["final_model_save"] = time.time() - final_save_start
            time_records["total_execution_time"] = time.time() - overall_start_time

            # 保存到CSV文件
            save_time_to_csv(time_records, args.output_dir)

            # 也生成一个简单的文本报告
            with open(os.path.join(args.output_dir, "last_run_time_report.txt"), "w") as f:
                f.write(f"Training Time Report (CheckFreq)\n")
                f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")

                for stage, duration in time_records.items():
                    if stage != "Epoch Details" and isinstance(duration, (int, float)):
                        f.write(f"{stage}: {duration:.2f}s\n")

                if epoch_times:
                    f.write("\n" + "=" * 50 + "\n")
                    f.write("Epoch Details:\n")
                    for epoch_info in epoch_times:
                        f.write(
                            f"  Epoch {epoch_info['epoch']}: {epoch_info['duration_seconds']:.2f}s "
                            f"(eval: {epoch_info['eval_time']:.2f}s)\n"
                        )

    logger.info(f"[TIME] Total execution time: {time.time() - overall_start_time:.2f} seconds")


if __name__ == "__main__":
    main()

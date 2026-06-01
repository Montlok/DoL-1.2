# -*- coding: utf-8 -*-

"""RDT supervised fine-tuning (SFT) entry point.

Reuses the pretraining loop primitives (optimizer/scheduler/train_one_step/
checkpoint) but swaps in:

- a masked chat dataset (:class:`Model.posttrain.sft_data.SFTChatDataset`) so
  loss falls only on assistant content + EOS;
- the bidirectional reverse-LM auxiliary loss **disabled**
  (``model.reverse_loss_enabled = False``) so it does not pollute the SFT
  gradient;
- an explicit SFT latent depth (``--recurrent-steps``), the depth the model is
  fine-tuned at (and should be served at).

Usage::

    python -m scripts.train_sft --config tiny --smoke
    python -m scripts.train_sft --config two_stage_tiny \\
        --tokenizer path/to/bundle --data chats/*.jsonl --output outputs/sft
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Model.config import (  # noqa: E402
    BOS_ID,
    EOS_ID,
    RDTConfig,
    TrainingConfig,
    base_config,
    pretrain_config,
    small_config,
    tiny_config,
    two_stage_pretrain_config,
    two_stage_tiny_config,
)
from Model.model import RDTForCausalLM  # noqa: E402
from Model.posttrain.sft_data import SFTChatDataset, build_sft_example  # noqa: E402
from Model.training import (  # noqa: E402
    PretrainingCollator,
    RankZeroLogger,
    TrainState,
    apply_parallelism,
    build_optimizer,
    build_scheduler,
    init_distributed,
    is_main_process,
    resume_state,
    save_checkpoint,
    throughput_str,
    train_one_step,
)

CONFIG_CHOICES = {
    "tiny": tiny_config,
    "small": small_config,
    "base": base_config,
    "pretrain": pretrain_config,
    "two_stage_tiny": two_stage_tiny_config,
    "two_stage_pretrain": two_stage_pretrain_config,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Supervised fine-tune RDT on chat data")
    p.add_argument("--config", choices=list(CONFIG_CHOICES), default="tiny")
    p.add_argument("--tokenizer", default="", help="TokenizerBundle dir (real runs)")
    p.add_argument("--data", default="", help="chat JSONL glob/path")
    p.add_argument("--output", default="outputs/sft")
    p.add_argument("--resume", default="")
    p.add_argument("--dist", choices=["single", "ddp", "fsdp"], default="single")
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument(
        "--recurrent-steps",
        type=int,
        default=None,
        help="SFT latent depth; defaults to the config's recurrent_steps",
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--dist-backend", choices=["nccl", "gloo"], default="nccl")
    p.add_argument("--smoke", action="store_true", help="run 4 in-memory steps")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def _build_model_cfg(args: argparse.Namespace) -> RDTConfig:
    cfg = CONFIG_CHOICES[args.config]()
    if args.recurrent_steps is not None:
        if args.recurrent_steps <= 0:
            raise ValueError("--recurrent-steps must be positive")
        cfg.recurrent_steps = args.recurrent_steps
    return cfg


def _build_train_cfg(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        train_data=args.data,
        seq_len=args.seq_len,
        micro_batch_size=args.micro_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        precision=args.precision,
        parallel=args.dist,
        dist_backend=args.dist_backend,
        output_dir=args.output,
        save_every=args.save_every,
        log_every=args.log_every,
        resume=args.resume,
        tensorboard=False,
    )


def _smoke_encode(text: str) -> list[int]:
    # Char-level, offset away from the special/control id band.
    return [(ord(c) % (RDTConfig().vocab_size - 1000)) + 1000 for c in text]


def _smoke_loader(train_cfg: TrainingConfig) -> DataLoader:
    chats = [
        [
            {"role": "user", "content": "сайн уу"},
            {"role": "assistant", "content": "<think>greet</think>сайн байна уу"},
        ],
        [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "<tool_call>add</tool_call>"},
            {"role": "tool", "content": "4"},
            {"role": "assistant", "content": "4"},
        ],
    ]
    rows = [
        build_sft_example(
            c, _smoke_encode, eos_id=EOS_ID, bos_id=BOS_ID, max_seq_len=train_cfg.seq_len
        )
        for c in chats
    ]
    collator = PretrainingCollator(max_seq_len=train_cfg.seq_len)
    return DataLoader(rows, batch_size=train_cfg.micro_batch_size, collate_fn=collator)


def _real_loader(
    args: argparse.Namespace,
    train_cfg: TrainingConfig,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    from Tokenizer.unified.bundle import TokenizerBundle

    bundle = TokenizerBundle.from_dir(args.tokenizer)
    dataset = SFTChatDataset(
        args.data,
        encode=lambda t: bundle.encode(t),
        eos_id=EOS_ID,
        bos_id=BOS_ID,
        max_seq_len=train_cfg.seq_len,
    )
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )
        if world_size > 1
        else None
    )
    collator = PretrainingCollator(max_seq_len=train_cfg.seq_len)
    return DataLoader(
        dataset,
        batch_size=train_cfg.micro_batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collator,
    )


def _infinite_loader(loader: DataLoader):
    epoch = 0
    sampler = getattr(loader, "sampler", None)
    while True:
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def _validate_args(args: argparse.Namespace) -> int:
    if not args.smoke:
        if not args.tokenizer:
            print("[error] --tokenizer is required (or pass --smoke)", file=sys.stderr)
            return 2
        if not args.data:
            print("[error] --data is required (or pass --smoke)", file=sys.stderr)
            return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rc = _validate_args(args)
    if rc != 0:
        return rc

    model_cfg = _build_model_cfg(args)
    train_cfg = _build_train_cfg(args)
    rank, world_size, local_rank = init_distributed(backend=train_cfg.dist_backend)
    torch.manual_seed(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main_process():
        Path(train_cfg.output_dir).mkdir(parents=True, exist_ok=True)

    model = RDTForCausalLM(model_cfg).to(device)
    # SFT: drop the pretraining bidirectional auxiliary objective.
    model.reverse_loss_enabled = False

    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)
    model = apply_parallelism(model, train_cfg, local_rank)

    state = TrainState()
    if train_cfg.resume:
        state.step = resume_state(
            train_cfg.resume, model, optimizer, scheduler, state=state
        )

    loader = (
        _smoke_loader(train_cfg)
        if args.smoke
        else _real_loader(args, train_cfg, rank=rank, world_size=world_size)
    )
    batch_iter = _infinite_loader(loader)

    logger = RankZeroLogger(train_cfg.output_dir, enable_tensorboard=False)
    t0 = time.time()
    tokens_window = 0
    try:
        while state.step < train_cfg.max_steps:
            metrics = train_one_step(
                model,
                batch_iter,
                optimizer,
                scheduler,
                train_cfg,
                state,
                device=device,
                target_recurrent_steps=model_cfg.recurrent_steps,
            )
            tokens_window += int(metrics["tokens"])
            if state.step % train_cfg.log_every == 0 or args.smoke:
                dt = max(1e-6, time.time() - t0)
                logger.log(
                    state.step,
                    {**metrics, "throughput": throughput_str(tokens_window, dt)},
                )
                t0 = time.time()
                tokens_window = 0
            if (
                train_cfg.save_every
                and state.step % train_cfg.save_every == 0
                and not args.smoke
            ):
                save_checkpoint(
                    train_cfg.output_dir,
                    state.step,
                    model,
                    optimizer,
                    scheduler,
                    metadata={"config": args.config, "phase": "sft"},
                    keep_last_n=train_cfg.keep_last_n,
                    scaler=state.extra.get("grad_scaler"),
                )
            if args.smoke and state.step >= 4:
                break
    finally:
        logger.close()
        if not args.smoke:
            save_checkpoint(
                train_cfg.output_dir,
                state.step,
                model,
                optimizer,
                scheduler,
                metadata={"config": args.config, "phase": "sft", "final": True},
                keep_last_n=train_cfg.keep_last_n,
                scaler=state.extra.get("grad_scaler"),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

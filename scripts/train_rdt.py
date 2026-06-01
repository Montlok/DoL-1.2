# -*- coding: utf-8 -*-

"""RDT text pretraining entry point.

Usage:

    python -m scripts.train_rdt --config tiny --smoke
    torchrun --nproc_per_node=2 -m scripts.train_rdt --config small \\
        --dist ddp --data path/to/shards/*.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

# Allow `torchrun scripts/train_rdt.py` (direct script invocation) to find
# the repository-root packages without requiring an editable install.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Model.config import (
    PAD_ID,
    RDTConfig,
    TrainingConfig,
    base_config,
    pretrain_config,
    small_config,
    tiny_config,
    two_stage_pretrain_config,
    two_stage_tiny_config,
)
from Model.model import RDTForCausalLM
from Model.layers.mamba3_layer import official_available  # noqa: E402
from Model.training import (
    PretrainingCollator,
    RankZeroLogger,
    TrainState,
    apply_parallelism,
    build_dataloader,
    build_image_processor,
    build_omvt_cfg,
    add_multimodal_args,
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
    p = argparse.ArgumentParser(description="Pretrain RDT on text data")
    p.add_argument("--config", choices=list(CONFIG_CHOICES), default="tiny")
    p.add_argument("--data", default="")
    p.add_argument("--eval-data", default="")
    p.add_argument("--output", default="outputs/rdt")
    p.add_argument("--resume", default="")
    p.add_argument("--dist", choices=["single", "ddp", "fsdp"], default="single")
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument(
        "--mamba",
        choices=["auto", "official", "naive"],
        default="auto",
        help=(
            "Mamba backend selection. 'auto' (default) uses the official CUDA "
            "mamba_ssm kernel when importable and otherwise falls back to the "
            "pure-PyTorch NaiveSSM with a warning; 'official' forces the CUDA "
            "kernel (errors if mamba_ssm is missing); 'naive' forces the "
            "fallback. Keeps one-click runs working on hosts without mamba_ssm."
        ),
    )
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--max-steps", type=int, default=100_000)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--bptt-window", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--dist-backend", choices=["nccl", "gloo"], default="nccl")
    p.add_argument(
        "--grad-ckpt-recurrent",
        choices=["auto", "on", "off"],
        default="auto",
        help="auto = inherit from RDTConfig; on/off override the model setting",
    )
    p.add_argument(
        "--grad-ckpt-prelude-coda",
        choices=["auto", "on", "off"],
        default="auto",
        help="auto = inherit from RDTConfig; on/off override the model setting",
    )
    p.add_argument("--smoke", action="store_true", help="run 4 in-memory steps")
    p.add_argument("--seed", type=int, default=42)
    add_multimodal_args(p)
    return p.parse_args(argv)


def _tri_to_bool(value: str) -> bool | None:
    if value == "auto":
        return None
    return value == "on"


def _resolve_mamba_backend(cfg: RDTConfig, mode: str) -> RDTConfig:
    """Apply the ``--mamba`` override so one-click runs survive hosts that lack
    the CUDA ``mamba_ssm`` kernel.

    * ``official`` forces ``use_official_mamba=True`` (model construction will
      raise a clear error if the kernel is unimportable).
    * ``naive`` forces the pure-PyTorch fallback.
    * ``auto`` keeps the config's request when the kernel is importable, and
      otherwise downgrades to the fallback with a loud warning instead of
      crashing at model construction.
    """

    if mode == "official":
        return replace(cfg, use_official_mamba=True)
    if mode == "naive":
        return replace(cfg, use_official_mamba=False)

    if cfg.use_official_mamba and not official_available():
        sys.stderr.write(
            "WARNING: --mamba=auto requested the official mamba_ssm kernel but "
            "it is not importable (no CUDA build?); falling back to NaiveSSM. "
            "This is correct but much slower; install mamba-ssm or pass "
            "--mamba=official to fail fast on a CUDA host.\n"
        )
        return replace(cfg, use_official_mamba=False)
    return cfg


def _build_model_cfg(args: argparse.Namespace) -> RDTConfig:
    cfg = CONFIG_CHOICES[args.config]()
    if args.seq_len is not None:
        cfg = replace(cfg, max_seq_len=args.seq_len)
    if args.smoke and args.seq_len is None:
        cfg = replace(cfg, max_seq_len=64)
    cfg = _resolve_mamba_backend(cfg, args.mamba)
    return cfg


def _apply_train_overrides(model_cfg: RDTConfig, train_cfg: TrainingConfig) -> RDTConfig:
    """Project ``TrainingConfig`` knobs that live on ``RDTConfig`` into the
    model config so they take effect at construction time."""

    overrides: dict = {}
    if train_cfg.grad_ckpt_recurrent is not None:
        overrides["grad_ckpt_recurrent"] = bool(train_cfg.grad_ckpt_recurrent)
    if train_cfg.grad_ckpt_prelude_coda is not None:
        overrides["grad_ckpt_prelude_coda"] = bool(train_cfg.grad_ckpt_prelude_coda)
    if overrides:
        model_cfg = replace(model_cfg, **overrides)
    return model_cfg


def _build_train_cfg(args: argparse.Namespace, model_cfg: RDTConfig) -> TrainingConfig:
    seq_len = args.seq_len if args.seq_len is not None else model_cfg.max_seq_len
    return TrainingConfig(
        train_data=args.data,
        eval_data=args.eval_data,
        seq_len=seq_len,
        micro_batch_size=args.micro_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        num_workers=args.num_workers if not args.smoke else 0,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=4 if args.smoke else args.max_steps,
        warmup_steps=1 if args.smoke else args.warmup_steps,
        precision=args.precision,
        grad_clip=args.grad_clip,
        parallel=args.dist,
        dist_backend=args.dist_backend,
        grad_ckpt_recurrent=_tri_to_bool(args.grad_ckpt_recurrent),
        grad_ckpt_prelude_coda=_tri_to_bool(args.grad_ckpt_prelude_coda),
        output_dir=args.output,
        save_every=args.save_every,
        log_every=args.log_every,
        bptt_window=args.bptt_window,
        seed=args.seed,
        resume=args.resume,
    )


def _smoke_batches(model_cfg: RDTConfig, train_cfg: TrainingConfig):
    rng = torch.Generator().manual_seed(train_cfg.seed)
    ids = torch.randint(
        300,
        320,
        (train_cfg.seq_len - 2,),
        generator=rng,
    ).tolist()
    seq = [model_cfg.bos_id] + ids + [model_cfg.eos_id]
    row = {"input_ids": seq, "attention_mask": [1] * len(seq), "labels": seq}
    collator = PretrainingCollator(max_seq_len=train_cfg.seq_len)
    while True:
        yield collator([row] * train_cfg.micro_batch_size)


def _validate_args(args: argparse.Namespace) -> int:
    """Argument-level validation.

    Runs **before** any expensive setup (distributed init, model alloc,
    output-dir creation) so misconfigured invocations fail fast with a
    clear stderr message and a non-zero exit code, instead of OOMing
    halfway through model construction or leaving stale ``outputs/`` dirs.
    Returns 0 on success, a non-zero code on failure.
    """

    if not args.smoke and not args.data:
        print(
            "scripts/train_rdt: --data is required for non-smoke runs; "
            "pass --smoke to run on synthetic tokens explicitly.",
            file=sys.stderr,
        )
        return 2
    if args.resume and not Path(args.resume).exists():
        print(
            f"scripts/train_rdt: --resume path does not exist: {args.resume}",
            file=sys.stderr,
        )
        return 2
    # Resolve the data spec **here** rather than waiting for build_dataloader
    # so an empty glob (typo'd shard pattern) fails *before* we allocate a
    # multi-billion-parameter model and initialize the process group.
    if not args.smoke and args.data:
        from Model.training.data import _resolve_shards

        shards = _resolve_shards(args.data)
        if not shards:
            print(
                f"scripts/train_rdt: --data resolved zero shards: {args.data!r}",
                file=sys.stderr,
            )
            return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rc = _validate_args(args)
    if rc != 0:
        return rc
    model_cfg_preview = _build_model_cfg(args)
    train_cfg_preview = _build_train_cfg(args, model_cfg_preview)
    omvt_cfg = build_omvt_cfg(args)
    rank, world_size, local_rank = init_distributed(backend=train_cfg_preview.dist_backend)
    if args.dist != "single" and world_size == 1 and not args.smoke:
        print(
            "[warn] --dist != single but world_size=1; running single-process",
            file=sys.stderr,
        )

    torch.manual_seed(args.seed + rank)

    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )

    model_cfg = model_cfg_preview
    train_cfg = train_cfg_preview
    model_cfg = _apply_train_overrides(model_cfg, train_cfg)

    if is_main_process():
        Path(train_cfg.output_dir).mkdir(parents=True, exist_ok=True)

    model = RDTForCausalLM(model_cfg).to(device)
    if omvt_cfg is not None:
        # Pre-install a matching-size OMVT injector so the dispatcher
        # does not lazily build a default-sized one on first forward
        # (which would mismatch the dataloader's pixel geometry).
        from Model.omvt import OMVTInjector

        model.vision._omvt_cfg = omvt_cfg
        model.vision.omvt = OMVTInjector(model_cfg, omvt_cfg).to(device)

    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)

    # NOTE: optimizer is built on the pre-wrap parameters. Under FSDP this is
    # only safe because we wrap with ``use_orig_params=True`` (see
    # ``apply_parallelism``), which keeps the original ``nn.Parameter`` objects
    # the optimizer references. If that flag ever changes, the optimizer must be
    # built *after* ``apply_parallelism`` instead. Verify on the GPU cluster.
    model = apply_parallelism(model, train_cfg, local_rank)

    state = TrainState()
    if train_cfg.resume:
        state.step = resume_state(
            train_cfg.resume, model, optimizer, scheduler, state=state
        )

    if args.smoke:
        batch_iter = _smoke_batches(model_cfg, train_cfg)
    else:
        dataloader = build_dataloader(
            train_cfg.train_data,
            train_cfg,
            world_size=world_size,
            rank=rank,
            pad_id=PAD_ID,
            image_processor=build_image_processor(args),
            omvt_cfg=omvt_cfg,
        )
        batch_iter = iter(dataloader)

    logger = RankZeroLogger(train_cfg.output_dir, enable_tensorboard=train_cfg.tensorboard)
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
                logger.log(state.step, {**metrics, "throughput": throughput_str(tokens_window, dt)})
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
                    metadata={"config": args.config},
                    keep_last_n=train_cfg.keep_last_n,
                    scaler=state.extra.get("grad_scaler"),
                )
    finally:
        logger.close()
        # NOTE: `save_checkpoint` is a collective under FSDP (the
        # `state_dict(FullStateDictConfig)` call gathers from every rank),
        # so we must enter it on every rank; the helper internally limits
        # the actual file write to rank 0. Guarding the call with
        # `is_main_process()` would deadlock rank 0 at shutdown.
        if not args.smoke:
            save_checkpoint(
                train_cfg.output_dir,
                state.step,
                model,
                optimizer,
                scheduler,
                metadata={"config": args.config, "final": True},
                keep_last_n=train_cfg.keep_last_n,
                scaler=state.extra.get("grad_scaler"),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

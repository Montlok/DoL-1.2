# -*- coding: utf-8 -*-

"""RDT Direct Preference Optimization (DPO) training entry point.

Production-usable offline alignment. Loads a policy from an SFT checkpoint,
builds a frozen reference from the same weights, and optimizes the DPO loss over
a preference JSONL (``{messages|prompt, chosen, rejected}``). Reuses the
project's optimizer/scheduler/checkpoint utilities and the SFT-consistent
preference masking.

Usage::

    python -m scripts.train_dpo --smoke
    python -m scripts.train_dpo --config two_stage_pretrain \\
        --tokenizer path/to/bundle --data prefs/*.jsonl \\
        --init-checkpoint outputs/sft/latest --output outputs/dpo \\
        --beta 0.1 --learning-rate 5e-7 --max-steps 1000
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
    PAD_ID,
    RDTConfig,
    TrainingConfig,
    base_config,
    pretrain_config,
    segmented_pretrain_config,
    segmented_tiny_config,
    small_config,
    tiny_config,
    two_stage_pretrain_config,
    two_stage_tiny_config,
)
from Model.model import RDTForCausalLM  # noqa: E402
from Model.posttrain.dpo import DPOConfig, dpo_step  # noqa: E402
from Model.posttrain.preference_data import (  # noqa: E402
    PreferenceDataset,
    build_preference_example,
    preference_collate,
)
from Model.training import (  # noqa: E402
    RankZeroLogger,
    TrainState,
    apply_parallelism,
    build_optimizer,
    build_scheduler,
    destroy_distributed,
    init_distributed,
    is_main_process,
    load_checkpoint,
    resume_state,
    save_checkpoint,
)

CONFIG_CHOICES = {
    "tiny": tiny_config,
    "small": small_config,
    "base": base_config,
    "pretrain": pretrain_config,
    "two_stage_tiny": two_stage_tiny_config,
    "two_stage_pretrain": two_stage_pretrain_config,
    "segmented_tiny": segmented_tiny_config,
    "segmented_pretrain": segmented_pretrain_config,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DPO-align RDT on preference data")
    p.add_argument("--config", choices=list(CONFIG_CHOICES), default="tiny")
    p.add_argument("--tokenizer", default="", help="TokenizerBundle dir (real runs)")
    p.add_argument("--data", default="", help="preference JSONL path")
    p.add_argument("--init-checkpoint", default="",
                   help="SFT checkpoint dir to initialize policy + reference")
    p.add_argument("--output", default="outputs/dpo")
    p.add_argument("--resume", default="")
    p.add_argument("--dist", choices=["single", "ddp", "fsdp"], default="single")
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=5e-7)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--length-normalize", action="store_true")
    p.add_argument("--recurrent-steps", type=int, default=None,
                   help="latent depth for scoring; defaults to config")
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
    return [(ord(c) % (RDTConfig().vocab_size - 1000)) + 1000 for c in text]


def _smoke_loader(train_cfg: TrainingConfig) -> DataLoader:
    pairs = [
        ([{"role": "user", "content": "2+2?"}], "<think>add</think>4", "5"),
        ([{"role": "user", "content": "сайн уу"}], "сайн байна уу", "no"),
    ]
    rows = [
        build_preference_example(
            ctx, chosen, rejected, _smoke_encode,
            eos_id=EOS_ID, bos_id=BOS_ID, max_seq_len=train_cfg.seq_len,
        )
        for ctx, chosen, rejected in pairs
    ]
    return DataLoader(
        rows,
        batch_size=train_cfg.micro_batch_size,
        collate_fn=lambda b: preference_collate(b, pad_id=PAD_ID),
    )


def _real_loader(
    args: argparse.Namespace,
    train_cfg: TrainingConfig,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    from Tokenizer.unified.bundle import TokenizerBundle

    bundle = TokenizerBundle.from_dir(args.tokenizer)
    dataset = PreferenceDataset(
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
    return DataLoader(
        dataset,
        batch_size=train_cfg.micro_batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=lambda b: preference_collate(b, pad_id=PAD_ID),
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


def _maybe_init_from_checkpoint(model: RDTForCausalLM, path: str) -> None:
    if not path:
        return
    payload = load_checkpoint(path)
    missing, unexpected = model.load_state_dict(payload.model_state, strict=False)
    if (missing or unexpected) and is_main_process():
        print(f"[init] loaded {path} (missing={len(missing)} unexpected={len(unexpected)})")


def _build_reference_model(
    policy: RDTForCausalLM,
    model_cfg: RDTConfig,
    train_cfg: TrainingConfig,
    local_rank: int,
    device: torch.device,
) -> torch.nn.Module:
    reference = RDTForCausalLM(model_cfg)
    reference.load_state_dict(policy.state_dict())
    reference.reverse_loss_enabled = False
    reference.eval()
    for param in reference.parameters():
        param.requires_grad_(False)

    if (
        train_cfg.parallel == "fsdp"
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        return apply_parallelism(reference, train_cfg, local_rank).eval()

    # DDP does not shard parameters and can reject modules with no trainable
    # parameters, so a frozen reference only benefits from FSDP wrapping.
    return reference.to(device).eval()


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

    policy = RDTForCausalLM(model_cfg)
    _maybe_init_from_checkpoint(policy, args.init_checkpoint)
    # DPO scores only the explicit objective; drop the reverse auxiliary loss.
    policy.reverse_loss_enabled = False
    reference = _build_reference_model(
        policy, model_cfg, train_cfg, local_rank, device
    )
    policy = policy.to(device)

    optimizer = build_optimizer(policy, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)
    policy = apply_parallelism(policy, train_cfg, local_rank)

    state = TrainState()
    if train_cfg.resume:
        state.step = resume_state(
            train_cfg.resume, policy, optimizer, scheduler, state=state
        )

    loader = (
        _smoke_loader(train_cfg)
        if args.smoke
        else _real_loader(args, train_cfg, rank=rank, world_size=world_size)
    )
    batch_iter = _infinite_loader(loader)
    dpo_cfg = DPOConfig(
        beta=args.beta,
        length_normalize=args.length_normalize,
        recurrent_steps=model_cfg.recurrent_steps,
    )

    logger = RankZeroLogger(train_cfg.output_dir, enable_tensorboard=False)
    t0 = time.time()
    try:
        while state.step < train_cfg.max_steps:
            optimizer.zero_grad()
            accum: dict[str, float] = {}
            for _ in range(train_cfg.grad_accum_steps):
                batch = next(batch_iter)
                loss, metrics = dpo_step(
                    policy,
                    reference,
                    batch["chosen_input_ids"].to(device),
                    batch["chosen_completion_mask"].to(device),
                    batch["rejected_input_ids"].to(device),
                    batch["rejected_completion_mask"].to(device),
                    dpo_cfg,
                    chosen_attn=batch["chosen_attention_mask"].to(device),
                    rejected_attn=batch["rejected_attention_mask"].to(device),
                )
                (loss / train_cfg.grad_accum_steps).backward()
                for k, v in metrics.items():
                    accum[k] = accum.get(k, 0.0) + v / train_cfg.grad_accum_steps
            if train_cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), train_cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            state.step += 1

            if state.step % train_cfg.log_every == 0 or args.smoke:
                dt = max(1e-6, time.time() - t0)
                logger.log(state.step, {**accum, "step_time_s": round(dt, 3)})
                t0 = time.time()
            if (
                train_cfg.save_every
                and state.step % train_cfg.save_every == 0
                and not args.smoke
            ):
                save_checkpoint(
                    train_cfg.output_dir, state.step, policy, optimizer, scheduler,
                    metadata={"config": args.config, "phase": "dpo"},
                    keep_last_n=train_cfg.keep_last_n,
                )
            if args.smoke and state.step >= 4:
                break
    finally:
        try:
            logger.close()
            if not args.smoke:
                save_checkpoint(
                    train_cfg.output_dir, state.step, policy, optimizer, scheduler,
                    metadata={"config": args.config, "phase": "dpo", "final": True},
                    keep_last_n=train_cfg.keep_last_n,
                )
        finally:
            destroy_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

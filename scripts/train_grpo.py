# -*- coding: utf-8 -*-

"""RDT GRPO (Group Relative Policy Optimization) training entry point.

Production-usable online RL with **verifiable** rewards (no reward model). Loads
a policy from an SFT/DPO checkpoint, builds a frozen reference, samples a group
of responses per prompt via the native ``generate()``, scores them with
rule-based rewards (exact/numeric match, Mongolian script purity, ``<think>``
format), normalizes advantages within each group, and takes a clipped
policy-gradient step with a KL penalty. Externally injected ``<tool_result>``
spans are excluded from the objective.

Usage::

    python -m scripts.train_grpo --smoke
    python -m scripts.train_grpo --config two_stage_pretrain \\
        --tokenizer path/to/bundle --data prompts/*.jsonl \\
        --init-checkpoint outputs/sft/latest --output outputs/grpo \\
        --group-size 8 --numeric-weight 1.0 --purity-weight 0.3 --format-weight 0.2
"""

import argparse
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import torch

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
from Model.posttrain.grpo import GRPOConfig, grpo_compute_loss  # noqa: E402
from Model.posttrain.preference_data import PromptDataset  # noqa: E402
from Model.posttrain.rewards import RewardConfig, compute_rewards  # noqa: E402
from Model.training import (  # noqa: E402
    RankZeroLogger,
    TrainState,
    apply_parallelism,
    build_optimizer,
    build_scheduler,
    clip_or_check_grad_norm,
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
    p = argparse.ArgumentParser(description="GRPO-align RDT with verifiable rewards")
    p.add_argument("--config", choices=list(CONFIG_CHOICES), default="two_stage_tiny")
    p.add_argument("--tokenizer", default="", help="TokenizerBundle dir (real runs)")
    p.add_argument("--data", default="", help="prompt JSONL path")
    p.add_argument("--init-checkpoint", default="",
                   help="checkpoint dir to initialize policy + reference")
    p.add_argument("--output", default="outputs/grpo")
    p.add_argument("--resume", default="")
    p.add_argument("--dist", choices=["single", "ddp", "fsdp"], default="single")
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--prompts-per-step", type=int, default=2)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-prompt-len", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--kl-coef", type=float, default=0.04)
    p.add_argument("--exact-weight", type=float, default=0.0)
    p.add_argument("--numeric-weight", type=float, default=0.0)
    p.add_argument("--purity-weight", type=float, default=0.0)
    p.add_argument("--purity-min-ratio", type=float, default=0.0)
    p.add_argument("--format-weight", type=float, default=0.0)
    p.add_argument("--recurrent-steps", type=int, default=None,
                   help="latent depth for sampling + scoring; defaults to config")
    p.add_argument("--dist-backend", choices=["nccl", "gloo"], default="nccl")
    p.add_argument("--no-tool-result-mask", action="store_true",
                   help="disable masking of externally-injected <tool_result> "
                        "spans from the RL objective (on by default)")
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
        micro_batch_size=args.prompts_per_step,
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


def _reward_cfg(args: argparse.Namespace) -> RewardConfig:
    return RewardConfig(
        exact_match_weight=args.exact_weight,
        numeric_match_weight=args.numeric_weight,
        purity_weight=args.purity_weight,
        purity_min_ratio=args.purity_min_ratio,
        format_weight=args.format_weight,
    )


def _smoke_encode(text: str) -> list[int]:
    return [(ord(c) % (RDTConfig().vocab_size - 1000)) + 1000 for c in text]


def _smoke_decode(ids: torch.Tensor) -> str:
    return "".join(chr((int(i) % 90) + 33) for i in ids.tolist() if int(i) != PAD_ID)


def _grpo_config(
    args: argparse.Namespace,
    model_cfg: RDTConfig,
    tool_result_open_ids: Sequence[int] | None = None,
    tool_result_close_ids: Sequence[int] | None = None,
) -> GRPOConfig:
    return GRPOConfig(
        clip_eps=args.clip_eps,
        kl_coef=args.kl_coef,
        recurrent_steps=model_cfg.recurrent_steps,
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        tool_result_open_ids=list(tool_result_open_ids or []),
        tool_result_close_ids=list(tool_result_close_ids or []),
    )


def _validate_args(args: argparse.Namespace) -> int:
    if args.group_size <= 1:
        print("[error] --group-size must be at least 2", file=sys.stderr)
        return 2
    if args.prompts_per_step <= 0:
        print("[error] --prompts-per-step must be positive", file=sys.stderr)
        return 2
    if args.max_new_tokens < 0:
        print("[error] --max-new-tokens must be non-negative", file=sys.stderr)
        return 2
    if not args.smoke:
        if not args.tokenizer:
            print("[error] --tokenizer is required (or pass --smoke)", file=sys.stderr)
            return 2
        if not args.data:
            print("[error] --data is required (or pass --smoke)", file=sys.stderr)
            return 2
        if (args.exact_weight == 0.0 and args.numeric_weight == 0.0
                and args.purity_weight == 0.0 and args.format_weight == 0.0):
            print("[error] at least one reward weight must be > 0", file=sys.stderr)
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


def _iter_prompt_batches(
    dataset,
    batch_size: int,
    *,
    rank: int = 0,
    world_size: int = 1,
):
    """Yield rank-sharded prompt rows, cycling forever.

    When the prompt set is smaller than the process count, every rank reads the
    full set to avoid distributed collectives hanging on ranks with no batches.
    """
    n = len(dataset)
    if n <= 0:
        raise ValueError("prompt dataset is empty")
    if world_size > 1 and n >= world_size:
        indices = list(range(rank, n, world_size))
    else:
        indices = list(range(n))
    if not indices:
        raise ValueError(f"rank={rank} received no prompts from dataset of size {n}")
    i = 0
    while True:
        batch = [dataset[indices[(i + j) % len(indices)]] for j in range(batch_size)]
        i = (i + batch_size) % len(indices)
        yield batch


def _smoke_dataset():
    prompts = [
        {"prompt_ids": _smoke_encode("2+2? "), "reference": "4"},
        {"prompt_ids": _smoke_encode("sain uu "), "reference": None},
    ]
    return prompts


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

    tool_open_ids: list[int] = []
    tool_close_ids: list[int] = []
    if args.smoke:
        dataset = _smoke_dataset()
        decode = _smoke_decode
    else:
        from Tokenizer.unified.bundle import TokenizerBundle

        from Model.posttrain.chat_template import (
            TOOL_RESULT_CLOSE,
            TOOL_RESULT_OPEN,
        )

        bundle = TokenizerBundle.from_dir(args.tokenizer)
        dataset = PromptDataset(
            args.data,
            encode=lambda t: bundle.encode(t),
            bos_id=BOS_ID,
            max_prompt_len=args.max_prompt_len,
        )
        decode = lambda ids: bundle.tokenizer.decode(  # noqa: E731
            [int(i) for i in ids.tolist() if int(i) != PAD_ID]
        )
        if not args.no_tool_result_mask:
            tool_open_ids = list(bundle.encode(TOOL_RESULT_OPEN, add_bos=False, add_eos=False))
            tool_close_ids = list(bundle.encode(TOOL_RESULT_CLOSE, add_bos=False, add_eos=False))

    reward_cfg = _reward_cfg(args)
    grpo_cfg = _grpo_config(args, model_cfg, tool_open_ids, tool_close_ids)
    batches = _iter_prompt_batches(
        dataset,
        args.prompts_per_step,
        rank=rank,
        world_size=world_size,
    )

    logger = RankZeroLogger(train_cfg.output_dir, enable_tensorboard=False)
    t0 = time.time()
    completed = False
    try:
        while state.step < train_cfg.max_steps:
            rows = next(batches)
            prompts = [torch.tensor(r["prompt_ids"], dtype=torch.long, device=device)
                       for r in rows]
            references = [r.get("reference") for r in rows]

            def reward_fn(responses, idx, _refs=references):
                refs = [_refs[idx]] * len(responses)
                return compute_rewards(responses, refs, reward_cfg)

            loss, metrics = grpo_compute_loss(
                policy, reference, prompts, reward_fn, decode,
                cfg=grpo_cfg, eos_id=EOS_ID, pad_id=PAD_ID,
            )
            if not bool(torch.isfinite(loss.detach())):
                raise FloatingPointError(f"non-finite GRPO loss at step {state.step}")
            optimizer.zero_grad()
            loss.backward()
            metrics["grad_norm"] = clip_or_check_grad_norm(
                policy,
                train_cfg.grad_clip,
                step=state.step,
            )
            optimizer.step()
            scheduler.step()
            state.step += 1

            if state.step % train_cfg.log_every == 0 or args.smoke:
                dt = max(1e-6, time.time() - t0)
                logger.log(state.step, {**metrics, "step_time_s": round(dt, 3)})
                t0 = time.time()
            if (
                train_cfg.save_every
                and state.step % train_cfg.save_every == 0
                and not args.smoke
            ):
                save_checkpoint(
                    train_cfg.output_dir, state.step, policy, optimizer, scheduler,
                    metadata={"config": args.config, "phase": "grpo"},
                    keep_last_n=train_cfg.keep_last_n,
                )
            if args.smoke and state.step >= 4:
                break
        completed = True
    finally:
        try:
            logger.close()
            if completed and not args.smoke:
                save_checkpoint(
                    train_cfg.output_dir, state.step, policy, optimizer, scheduler,
                    metadata={"config": args.config, "phase": "grpo", "final": True},
                    keep_last_n=train_cfg.keep_last_n,
                )
        finally:
            destroy_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

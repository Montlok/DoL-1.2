# -*- coding: utf-8 -*-

"""VLM alignment: OMVT vision tower → projector → RDT.

Runs a synthetic end-to-end forward/backward where the RDT LM consumes
``<image_patch>`` slots filled by the OMVT compressed tokens.  The LM head
is fine-tuned by default; pass ``--freeze-rdt`` to train only the
projector/tower.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

from Model.config import (
    BOS_ID,
    EOS_ID,
    IMAGE_PATCH_ID,
    OMVTConfig,
    PAD_ID,
    TrainingConfig,
)
from Model.model import RDTForCausalLM
from Model.omvt import OMVTInjector
from Model.omvt.patcher import collate_omvt_batch
from Model.training import (
    RankZeroLogger,
    TrainState,
    build_dataloader,
    build_optimizer,
    build_scheduler,
    clip_or_check_grad_norm,
    load_checkpoint,
    resume_state,
    save_checkpoint,
    train_one_step,
)
from Model.training.multimodal_cli import make_omvt_cfg
from Model.training.omvt_checkpoint import (
    load_omvt_payload,
    tower_state_from_payload,
)
from Tokenizer.multimodal import PILImageProcessor
from Tokenizer.multimodal.image_placeholders import image_patch_count
from scripts.train_rdt import CONFIG_CHOICES, _resolve_mamba_backend


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=list(CONFIG_CHOICES), default="tiny")
    p.add_argument(
        "--mamba",
        choices=["auto", "official", "naive"],
        default="auto",
        help=(
            "Mamba backend for the RDT side. auto uses official CUDA Mamba on "
            "CUDA/Linux and NaiveSSM on macOS/CPU."
        ),
    )
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--image-size", type=int, default=56)
    p.add_argument("--seq-len", type=int, default=24)
    p.add_argument("--n-image-tokens", type=int, default=None)
    p.add_argument("--freeze-rdt", action="store_true")
    p.add_argument(
        "--frozen-vision",
        action="store_true",
        help="freeze the OMVT tower as well (useful for projector-only ablations)",
    )
    p.add_argument(
        "--data",
        default="",
        help="JSONL spec for real multimodal pretraining (rows must carry an 'images' field)",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--output", default="outputs/vlm_align")
    p.add_argument("--init-rdt-checkpoint", default="")
    p.add_argument("--init-omvt-checkpoint", default="")
    p.add_argument(
        "--use-ema-tower",
        action="store_true",
        help="when --init-omvt-checkpoint carries 'tower_ema', overlay the EMA "
        "weights on the tower state before loading",
    )
    p.add_argument("--resume", default="")
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--precision",
        choices=("auto", "fp32", "bf16"),
        default="fp32",
        help="'auto' = bf16 on cuda, fp32 elsewhere; default keeps the legacy "
        "fp32 smoke behavior",
    )
    p.add_argument("--warmup-steps", type=int, default=1)
    p.add_argument(
        "--d-vision",
        type=int,
        default=64,
        help="OMVT tower width for from-scratch towers; ignored when "
        "--init-omvt-checkpoint provides its own omvt_config",
    )
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="'auto' = cuda if available else cpu (legacy behavior)",
    )
    p.add_argument(
        "--patch-preset",
        choices=("derived", "prod"),
        default="derived",
        help="for from-scratch towers: 'derived' keeps the legacy smoke "
        "geometry (patches scaled from --image-size); 'prod' uses the "
        "OMVTConfig dataclass multi-scale defaults (32x8 / 8x32 / 16x16 / "
        "56x56). Ignored when --init-omvt-checkpoint provides omvt_config.",
    )
    p.add_argument(
        "--grad-ckpt",
        action="store_true",
        help="enable RDT gradient checkpointing (grad_ckpt_recurrent + "
        "grad_ckpt_prelude_coda) to fit long sequences on small GPUs",
    )
    return p.parse_args(argv)


def _build_omvt_cfg(args) -> OMVTConfig:
    if args.init_omvt_checkpoint:
        # The checkpoint's own omvt_config is the only authoritative source of
        # tower geometry: building a CLI-derived config here and loading a
        # prod-geometry tower (e.g. d_vision=512, dataclass patch shapes) into
        # it would fail on shape mismatch.
        payload = load_omvt_payload(args.init_omvt_checkpoint, weights_only=False)
        if isinstance(payload, dict) and "omvt_config" in payload:
            cfg = OMVTConfig(**payload["omvt_config"])
            if args.image_size != cfg.image_size:
                print(
                    f"[init] --image-size {args.image_size} -> {cfg.image_size} "
                    "(from OMVT checkpoint)",
                )
            return cfg
    return make_omvt_cfg(
        args.image_size,
        args.d_vision,
        args.n_image_tokens,
        preset=getattr(args, "patch_preset", "derived"),
    )


def _make_text_batch(args, vocab_floor=300, vocab_ceil=320):
    B, L, N = args.batch_size, args.seq_len, args.n_image_tokens
    rng = torch.Generator().manual_seed(args.seed)
    # layout: [BOS] <image_patch>*N <random text...> [EOS]
    text_len = L - N - 2
    if text_len <= 0:
        raise ValueError("seq_len must be greater than 2 + n_image_tokens")
    text_ids = torch.randint(vocab_floor, vocab_ceil, (B, text_len), generator=rng)
    input_ids = torch.full((B, L), 0, dtype=torch.long)
    input_ids[:, 0] = BOS_ID
    input_ids[:, 1 : 1 + N] = IMAGE_PATCH_ID
    input_ids[:, 1 + N : 1 + N + text_len] = text_ids
    input_ids[:, -1] = EOS_ID
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return input_ids, attention_mask, labels


def _resolve_model_state(path: str):
    p = Path(path)
    if p.is_file():
        state = torch.load(p, map_location="cpu", weights_only=False)
    else:
        state = load_checkpoint(p).model_state
    if isinstance(state, dict) and "model" in state and "embed.weight" not in state:
        state = state["model"]
    return state


def _load_rdt_init(model: RDTForCausalLM, path: str) -> None:
    if not path:
        return
    missing, unexpected = model.load_state_dict(_resolve_model_state(path), strict=False)
    if missing or unexpected:
        print(
            f"[init] loaded RDT checkpoint {path} "
            f"(missing={len(missing)} unexpected={len(unexpected)})"
        )


def _resolve_omvt_state(path: str, use_ema: bool = False):
    payload = load_omvt_payload(path, weights_only=False)
    state = tower_state_from_payload(payload, use_ema=use_ema)
    if use_ema and isinstance(payload, dict) and payload.get("tower_ema"):
        print("[init] using EMA tower weights")
    return state


def _load_omvt_init(model: RDTForCausalLM, path: str, use_ema: bool = False) -> None:
    if not path:
        return
    if model.vision.omvt is None:
        raise ValueError("OMVT injector must be installed before loading tower weights")
    model.vision.omvt.tower.load_state_dict(_resolve_omvt_state(path, use_ema=use_ema))


def main(argv=None):
    args = parse_args(argv)
    # Fast-fail validation **before** any device alloc / model construction
    # (and before deriving n_image_tokens, which calls image_patch_count and
    # would otherwise raise a raw traceback on a non-positive --image-size).
    # Mirrors the train_rdt CLI pattern: misconfigured runs should not pay the
    # cost of building the model only to crash inside the first step.
    if args.image_size <= 0 or args.image_size % 4 != 0:
        print(
            "scripts/train_vlm_align: --image-size must be a positive multiple of 4",
            file=sys.stderr,
        )
        return 2
    if args.n_image_tokens is None:
        args.n_image_tokens = image_patch_count(args.image_size, args.image_size)
    if args.seq_len <= args.n_image_tokens + 2:
        print(
            "scripts/train_vlm_align: --seq-len must be > --n-image-tokens + 2 "
            f"(got seq_len={args.seq_len}, n_image_tokens={args.n_image_tokens})",
            file=sys.stderr,
        )
        return 2
    if args.resume and (args.init_rdt_checkpoint or args.init_omvt_checkpoint):
        print(
            "scripts/train_vlm_align: --resume cannot be combined with "
            "--init-rdt-checkpoint or --init-omvt-checkpoint",
            file=sys.stderr,
        )
        return 2

    torch.manual_seed(args.seed)
    if getattr(args, "device", "auto") == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    rdt_cfg = CONFIG_CHOICES[args.config]()
    # cap seq_len to the synthetic layout (tiny config is 2048 by default but
    # the smoke layout is much shorter)
    rdt_cfg = replace(rdt_cfg, max_seq_len=args.seq_len)
    if args.grad_ckpt:
        rdt_cfg = replace(
            rdt_cfg, grad_ckpt_recurrent=True, grad_ckpt_prelude_coda=True
        )
    try:
        rdt_cfg = _resolve_mamba_backend(
            rdt_cfg,
            args.mamba,
            device=device,
            context="scripts.train_vlm_align",
        )
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    omvt_cfg = _build_omvt_cfg(args)

    model = RDTForCausalLM(rdt_cfg).to(device)
    # plug in matching-size OMVT injector (otherwise dispatcher would build
    # a default-sized one on first forward and fail on tiny synthetic inputs).
    model.vision._omvt_cfg = omvt_cfg
    model.vision.omvt = OMVTInjector(rdt_cfg, omvt_cfg).to(device)
    _load_rdt_init(model, args.init_rdt_checkpoint)
    _load_omvt_init(model, args.init_omvt_checkpoint, use_ema=args.use_ema_tower)

    if args.freeze_rdt:
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.vision.omvt.parameters():
            p.requires_grad_(True)
    if args.frozen_vision:
        for p in model.vision.omvt.tower.parameters():
            p.requires_grad_(False)

    if args.precision == "auto":
        precision = "bf16" if device.type == "cuda" else "fp32"
    else:
        precision = args.precision
    train_cfg = TrainingConfig(
        train_data=args.data,
        seq_len=args.seq_len,
        micro_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.05,
        max_steps=args.steps,
        warmup_steps=max(1, args.warmup_steps),
        precision=precision,
        output_dir=args.output,
        save_every=args.save_every,
        resume=args.resume,
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("model has no trainable parameters")
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)
    state = TrainState()
    if args.resume:
        state.step = resume_state(
            args.resume,
            model,
            optimizer,
            scheduler,
            state=state,
        )

    Path(args.output).mkdir(parents=True, exist_ok=True)
    logger = RankZeroLogger(args.output, enable_tensorboard=False)

    t0 = time.time()
    completed = False
    try:
        if args.data:
            # Real-data path: pull pixel-aware batches from the streaming
            # JSONL dataloader and reuse the canonical train_one_step so
            # CLI behaviour matches train_rdt.
            dataloader = build_dataloader(
                args.data,
                train_cfg,
                world_size=1,
                rank=0,
                pad_id=PAD_ID,
                image_processor=PILImageProcessor(image_size=args.image_size),
                omvt_cfg=omvt_cfg,
            )
            batch_iter = iter(dataloader)
            while state.step < args.steps:
                metrics = train_one_step(
                    model,
                    batch_iter,
                    optimizer,
                    scheduler,
                    train_cfg,
                    state,
                    device=device,
                )
                logger.log(state.step, {"loss": metrics["loss"]})
                if args.save_every and state.step % args.save_every == 0:
                    save_checkpoint(
                        args.output,
                        state.step,
                        model,
                        optimizer,
                        scheduler,
                        metadata={"phase": "vlm_align", "config": args.config},
                    )
        else:
            while state.step < args.steps:
                step = state.step + 1
                input_ids, attention_mask, labels = _make_text_batch(args)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                images = torch.randn(
                    args.batch_size,
                    omvt_cfg.in_channels,
                    omvt_cfg.image_size,
                    omvt_cfg.image_size,
                    device=device,
                )
                batch = collate_omvt_batch(images, omvt_cfg)

                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    pixel_values=dict(batch),
                )
                loss = out["loss"]
                if not bool(torch.isfinite(loss.detach())):
                    raise FloatingPointError(
                        f"non-finite VLM align loss at step {state.step}"
                    )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                clip_or_check_grad_norm(model, 1.0, step=step)
                optimizer.step()
                scheduler.step()
                state.step = step

                logger.log(step, {"loss": float(loss.detach())})
                if args.save_every and step % args.save_every == 0:
                    save_checkpoint(
                        args.output,
                        step,
                        model,
                        optimizer,
                        scheduler,
                        metadata={"phase": "vlm_align", "config": args.config},
                    )
        completed = True
    finally:
        logger.close()
        if completed and not args.smoke:
            save_checkpoint(
                args.output,
                state.step,
                model,
                optimizer,
                scheduler,
                metadata={"phase": "vlm_align", "config": args.config, "final": True},
            )
    mode = "real-data" if args.data else "smoke"
    print(f"VLM align {mode} run OK in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

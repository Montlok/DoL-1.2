# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass


# Canonical source of special-token ids / vocab segmentation is
# ``Tokenizer/unified/vocab.py``. The literals below are a standalone fallback
# used only when the Tokenizer package cannot be imported. They MUST stay in
# sync with the canonical definitions; ``Model/tests/test_config.py`` asserts it
# whenever the Tokenizer package is importable.
_FALLBACK_SPECIAL_TOKENS = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3,
    "<img>": 4,
    "<image>": 5,
    "<image_start>": 6,
    "<image_patch>": 7,
    "<image_end>": 8,
    "<video>": 9,
    "<video_start>": 10,
    "<video_patch>": 11,
    "<video_end>": 12,
    "<bbox>": 13,
    "<ocr>": 14,
    "<ocr_start>": 15,
    "<ocr_end>": 16,
    "▁": 17,
    "◈": 18,
    "<doc>": 19,
    "<table>": 20,
    "<layout>": 21,
    "<audio>": 22,
    "<audio_start>": 23,
    "<audio_patch>": 24,
    "<audio_end>": 25,
}

_FALLBACK_SEGMENT = {
    "special": (0, 256),
    "mongolian": (256, 24576),
    "general": (24576, 65536),
}

try:
    from Tokenizer.unified.vocab import SPECIAL_TOKENS, SEGMENT
except ImportError:
    SPECIAL_TOKENS = _FALLBACK_SPECIAL_TOKENS
    SEGMENT = _FALLBACK_SEGMENT


VOCAB_SIZE = 65536
IGNORE_INDEX = -100

PAD_ID = SPECIAL_TOKENS["<pad>"]
UNK_ID = SPECIAL_TOKENS["<unk>"]
BOS_ID = SPECIAL_TOKENS["<bos>"]
EOS_ID = SPECIAL_TOKENS["<eos>"]
IMG_ID = SPECIAL_TOKENS["<img>"]

IMAGE_ID = SPECIAL_TOKENS["<image>"]
IMAGE_START_ID = SPECIAL_TOKENS["<image_start>"]
IMAGE_PATCH_ID = SPECIAL_TOKENS["<image_patch>"]
IMAGE_END_ID = SPECIAL_TOKENS["<image_end>"]

VIDEO_ID = SPECIAL_TOKENS["<video>"]
VIDEO_START_ID = SPECIAL_TOKENS["<video_start>"]
VIDEO_PATCH_ID = SPECIAL_TOKENS["<video_patch>"]
VIDEO_END_ID = SPECIAL_TOKENS["<video_end>"]

BBOX_ID = SPECIAL_TOKENS["<bbox>"]
OCR_ID = SPECIAL_TOKENS["<ocr>"]
OCR_START_ID = SPECIAL_TOKENS["<ocr_start>"]
OCR_END_ID = SPECIAL_TOKENS["<ocr_end>"]

WORD_BOUNDARY_ID = SPECIAL_TOKENS["▁"]
MORPHEME_BOUNDARY_ID = SPECIAL_TOKENS["◈"]

DOC_ID = SPECIAL_TOKENS["<doc>"]
TABLE_ID = SPECIAL_TOKENS["<table>"]
LAYOUT_ID = SPECIAL_TOKENS["<layout>"]

AUDIO_ID = SPECIAL_TOKENS["<audio>"]
AUDIO_START_ID = SPECIAL_TOKENS["<audio_start>"]
AUDIO_PATCH_ID = SPECIAL_TOKENS["<audio_patch>"]
AUDIO_END_ID = SPECIAL_TOKENS["<audio_end>"]


@dataclass
class RDTConfig:
    vocab_size: int = VOCAB_SIZE
    ignore_index: int = IGNORE_INDEX

    pad_id: int = PAD_ID
    unk_id: int = UNK_ID
    bos_id: int = BOS_ID
    eos_id: int = EOS_ID

    img_id: int = IMG_ID

    image_id: int = IMAGE_ID
    image_start_id: int = IMAGE_START_ID
    image_patch_id: int = IMAGE_PATCH_ID
    image_end_id: int = IMAGE_END_ID

    video_id: int = VIDEO_ID
    video_start_id: int = VIDEO_START_ID
    video_patch_id: int = VIDEO_PATCH_ID
    video_end_id: int = VIDEO_END_ID

    bbox_id: int = BBOX_ID
    ocr_id: int = OCR_ID
    ocr_start_id: int = OCR_START_ID
    ocr_end_id: int = OCR_END_ID

    word_boundary_id: int = WORD_BOUNDARY_ID
    morpheme_boundary_id: int = MORPHEME_BOUNDARY_ID

    doc_id: int = DOC_ID
    table_id: int = TABLE_ID
    layout_id: int = LAYOUT_ID

    audio_id: int = AUDIO_ID
    audio_start_id: int = AUDIO_START_ID
    audio_patch_id: int = AUDIO_PATCH_ID
    audio_end_id: int = AUDIO_END_ID

    d_model: int = 768
    n_heads: int = 12
    head_dim: int = 64

    kv_lora_rank: int = 256
    q_lora_rank: int = 0
    rope_head_dim: int = 32
    nope_head_dim: int = 32

    ffn_hidden: int = 2048
    ffn_multiple: int = 256

    n_prelude: int = 2
    n_coda: int = 2
    mamba_per_block: int = 5
    attn_per_block: int = 1

    recurrent_steps: int = 8
    inject_embedding: bool = True
    inject_scale: float = 1.0

    # Recurrent core selection. "interleaved" uses RecurrentCore (Mamba/attn
    # interleaved per step). "two_stage" uses TwoStageCore: a pure-Mamba
    # encoding stage followed by a pure-attention recurrent refinement stage.
    core_type: str = "interleaved"
    stage1_mamba_layers: int = 5
    stage2_attn_layers: int = 1

    # Drift control for the two-stage refinement loop:
    #   none  — plain recurrent attention refinement.
    #   norm  — RMSNorm between recurrent steps (boundary norm).
    #   decay — inject decayed Stage-1 semantics each step.
    #   both  — norm + decay.
    #   mhc   — per-layer Manifold-Constrained Hyper-Connections (MHCAttnSubLayer);
    #           the loop itself does no injection or boundary norm.
    recurrent_drift_mode: str = "none"
    recurrent_inject_decay: float = 0.5
    mhc_n_streams: int = 4
    mhc_sinkhorn_iters: int = 20

    # Order-preserving downsampling for the two-stage core. Default off: causal
    # pretraining would leak intra-word future characters through pooled
    # segments. Kept available for non-causal scenarios.
    two_stage_downsample: bool = False
    two_stage_max_segments: int = 0

    use_act: bool = False
    act_threshold: float = 0.99
    act_max_steps: int = 32
    act_ponder_cost: float = 0.01

    mamba_d_state: int = 128
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_headdim: int = 64
    mamba_dt_min: float = 0.001
    mamba_dt_max: float = 0.1
    mamba_dt_init_floor: float = 1e-4
    mamba_chunk_size: int = 64
    use_official_mamba: bool = True

    grad_ckpt_recurrent: bool = False
    grad_ckpt_blocks: bool = False
    grad_ckpt_prelude_coda: bool = False

    rope_theta: float = 10000.0
    use_morphological_rope: bool = True
    max_morph_depth: int = 8
    use_sdpa_attention: bool = True

    max_seq_len: int = 4096

    bidirectional: bool = True
    reverse_loss_weight: float = 0.5

    dropout: float = 0.0
    rmsnorm_eps: float = 1e-5
    init_std: float = 0.02
    tie_word_embeddings: bool = True
    loss_chunk_size: int | None = None

    dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        self._check_tokens()
        self._check_dims()
        self._check_depth()
        self._check_objectives()
        self._check_core()

    def _check_tokens(self) -> None:
        if self.vocab_size != VOCAB_SIZE:
            raise ValueError(f"vocab_size must be {VOCAB_SIZE}")

        lo, hi = SEGMENT["special"]
        ids = [
            self.pad_id,
            self.unk_id,
            self.bos_id,
            self.eos_id,
            self.img_id,
            self.image_id,
            self.image_start_id,
            self.image_patch_id,
            self.image_end_id,
            self.video_id,
            self.video_start_id,
            self.video_patch_id,
            self.video_end_id,
            self.bbox_id,
            self.ocr_id,
            self.ocr_start_id,
            self.ocr_end_id,
            self.word_boundary_id,
            self.morpheme_boundary_id,
            self.doc_id,
            self.table_id,
            self.layout_id,
            self.audio_id,
            self.audio_start_id,
            self.audio_patch_id,
            self.audio_end_id,
        ]

        for token_id in ids:
            if not (lo <= token_id < hi):
                raise ValueError(f"special id out of range: {token_id}")

        if len(ids) != len(set(ids)):
            raise ValueError("duplicate special token ids")

    def _check_dims(self) -> None:
        if self.d_model != self.n_heads * self.head_dim:
            raise ValueError("d_model must equal n_heads * head_dim")

        if self.rope_head_dim + self.nope_head_dim != self.head_dim:
            raise ValueError("rope_head_dim + nope_head_dim must equal head_dim")

        if self.rope_head_dim % 2 != 0:
            raise ValueError("rope_head_dim must be even")

        if self.d_model % self.mamba_headdim != 0:
            raise ValueError("d_model must be divisible by mamba_headdim")

        if self.ffn_hidden % self.ffn_multiple != 0:
            raise ValueError("ffn_hidden must be divisible by ffn_multiple")

        if self.kv_lora_rank <= 0:
            raise ValueError("kv_lora_rank must be positive")

        if self.q_lora_rank < 0:
            raise ValueError("q_lora_rank must be non-negative")

    def _check_depth(self) -> None:
        if self.n_prelude < 0 or self.n_coda < 0:
            raise ValueError("n_prelude and n_coda must be non-negative")

        if self.mamba_per_block < 0 or self.attn_per_block < 0:
            raise ValueError("block layer counts must be non-negative")

        if self.mamba_per_block + self.attn_per_block <= 0:
            raise ValueError("recurrent block cannot be empty")

        if self.recurrent_steps <= 0:
            raise ValueError("recurrent_steps must be positive")

        if self.use_act and self.act_max_steps <= 0:
            raise ValueError("act_max_steps must be positive")

        if self.inject_scale < 0:
            raise ValueError("inject_scale must be non-negative")

    def _check_objectives(self) -> None:
        if self.reverse_loss_weight < 0:
            raise ValueError("reverse_loss_weight must be non-negative")

        if self.act_ponder_cost < 0:
            raise ValueError("act_ponder_cost must be non-negative")

        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")

        if self.rmsnorm_eps <= 0:
            raise ValueError("rmsnorm_eps must be positive")

        if self.init_std <= 0:
            raise ValueError("init_std must be positive")

        if not (0.0 < self.act_threshold <= 1.0):
            raise ValueError("act_threshold must be in (0, 1]")

        if self.max_morph_depth <= 0:
            raise ValueError("max_morph_depth must be positive")

        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        if self.loss_chunk_size is not None and self.loss_chunk_size <= 0:
            raise ValueError("loss_chunk_size must be positive")

    def _check_core(self) -> None:
        if self.core_type not in {"interleaved", "two_stage"}:
            raise ValueError(
                f"core_type must be 'interleaved' or 'two_stage', got {self.core_type!r}"
            )

        if self.recurrent_drift_mode not in {"none", "norm", "decay", "both", "mhc"}:
            raise ValueError(
                "recurrent_drift_mode must be one of "
                "'none'/'norm'/'decay'/'both'/'mhc', got "
                f"{self.recurrent_drift_mode!r}"
            )

        if self.core_type == "two_stage":
            if self.use_act:
                raise ValueError(
                    "core_type='two_stage' does not support use_act=True; ACT is "
                    "only implemented for the interleaved RecurrentCore"
                )
            if self.stage1_mamba_layers <= 0:
                raise ValueError("stage1_mamba_layers must be positive")
            if self.stage2_attn_layers <= 0:
                raise ValueError("stage2_attn_layers must be positive")
            if self.mhc_n_streams <= 0:
                raise ValueError("mhc_n_streams must be positive")
            if self.mhc_sinkhorn_iters < 0:
                raise ValueError("mhc_sinkhorn_iters must be non-negative")
            if self.recurrent_inject_decay < 0:
                raise ValueError("recurrent_inject_decay must be non-negative")

            if self.recurrent_drift_mode == "mhc" and self.mhc_sinkhorn_iters < 1:
                raise ValueError(
                    "recurrent_drift_mode='mhc' requires mhc_sinkhorn_iters >= 1; "
                    "0 iterations cannot project onto the Birkhoff polytope"
                )

            # Order-preserving downsampling is not implemented for the causal
            # two-stage core: mean-pooling a word's characters into one segment
            # leaks that word's future characters into earlier positions. Reject
            # it explicitly instead of letting the knob silently no-op.
            if self.two_stage_downsample:
                raise ValueError(
                    "two_stage_downsample=True is not supported by the causal "
                    "two-stage core (it would leak intra-word future on a causal "
                    "path); keep two_stage_downsample=False"
                )

    @property
    def block_layers(self) -> int:
        return self.mamba_per_block + self.attn_per_block

    @property
    def effective_depth(self) -> int:
        steps = self.act_max_steps if self.use_act else self.recurrent_steps
        return self.n_prelude + self.block_layers * steps + self.n_coda

    @property
    def actual_layers(self) -> int:
        return self.n_prelude + self.block_layers + self.n_coda


def tiny_config() -> RDTConfig:
    return RDTConfig(
        d_model=512,
        n_heads=8,
        head_dim=64,
        kv_lora_rank=128,
        rope_head_dim=32,
        nope_head_dim=32,
        ffn_hidden=1536,
        ffn_multiple=256,
        n_prelude=2,
        n_coda=2,
        mamba_per_block=5,
        attn_per_block=1,
        recurrent_steps=4,
        max_seq_len=2048,
        use_official_mamba=False,
    )


def small_config() -> RDTConfig:
    return RDTConfig(
        d_model=1024,
        n_heads=16,
        head_dim=64,
        kv_lora_rank=256,
        ffn_hidden=3072,
        ffn_multiple=256,
        n_prelude=3,
        n_coda=3,
        mamba_per_block=5,
        attn_per_block=1,
        recurrent_steps=8,
        max_seq_len=4096,
    )


def base_config() -> RDTConfig:
    return RDTConfig(
        d_model=2048,
        n_heads=32,
        head_dim=64,
        kv_lora_rank=512,
        ffn_hidden=6144,
        ffn_multiple=256,
        n_prelude=4,
        n_coda=4,
        mamba_per_block=5,
        attn_per_block=1,
        recurrent_steps=16,
        max_seq_len=8192,
    )


def pretrain_config() -> RDTConfig:
    """~1.1B RDT for the first formal text pretraining wave.

    Defaults to grad checkpointing on; FSDP/DDP can override via TrainingConfig.
    """

    return RDTConfig(
        d_model=2048,
        n_heads=16,
        head_dim=128,
        kv_lora_rank=512,
        rope_head_dim=64,
        nope_head_dim=64,
        ffn_hidden=8192,
        ffn_multiple=256,
        n_prelude=3,
        n_coda=3,
        mamba_per_block=5,
        attn_per_block=1,
        recurrent_steps=8,
        max_seq_len=4096,
        use_official_mamba=True,
        bidirectional=False,
        grad_ckpt_recurrent=True,
        grad_ckpt_prelude_coda=True,
        loss_chunk_size=8192,
    )


def two_stage_tiny_config() -> RDTConfig:
    """Tiny two-stage core for CPU smoke / tests (NaiveSSM fallback).

    Mirrors :func:`tiny_config` shape but routes through ``TwoStageCore`` with
    per-layer mHC drift control.
    """

    return RDTConfig(
        d_model=512,
        n_heads=8,
        head_dim=64,
        kv_lora_rank=128,
        rope_head_dim=32,
        nope_head_dim=32,
        ffn_hidden=1536,
        ffn_multiple=256,
        n_prelude=2,
        n_coda=2,
        mamba_per_block=5,
        attn_per_block=1,
        recurrent_steps=4,
        max_seq_len=2048,
        use_official_mamba=False,
        core_type="two_stage",
        stage1_mamba_layers=5,
        stage2_attn_layers=1,
        recurrent_drift_mode="mhc",
        mhc_n_streams=4,
        mhc_sinkhorn_iters=20,
    )


def two_stage_pretrain_config() -> RDTConfig:
    """~1.1B two-stage RDT for formal pretraining (official Mamba on CUDA)."""

    return RDTConfig(
        d_model=2048,
        n_heads=16,
        head_dim=128,
        kv_lora_rank=512,
        rope_head_dim=64,
        nope_head_dim=64,
        ffn_hidden=8192,
        ffn_multiple=256,
        n_prelude=3,
        n_coda=3,
        mamba_per_block=5,
        attn_per_block=1,
        recurrent_steps=8,
        max_seq_len=4096,
        use_official_mamba=True,
        bidirectional=False,
        grad_ckpt_recurrent=True,
        grad_ckpt_prelude_coda=True,
        loss_chunk_size=8192,
        core_type="two_stage",
        stage1_mamba_layers=5,
        stage2_attn_layers=1,
        recurrent_drift_mode="mhc",
        mhc_n_streams=4,
        mhc_sinkhorn_iters=20,
    )


@dataclass
class TrainingConfig:
    """Top-level training-loop configuration.

    Intentionally decoupled from ``RDTConfig`` so model shape and training
    schedule can be edited independently. Field naming follows the order
    of how the loop consumes them: data → optimizer → schedule → dist →
    checkpoint → logging.
    """

    # data
    train_data: str = ""
    eval_data: str = ""
    seq_len: int = 4096
    micro_batch_size: int = 1
    grad_accum_steps: int = 1
    num_workers: int = 2
    pin_memory: bool = True
    shuffle_buffer: int = 1024

    # optimizer (AdamW, decoupled WD)
    optimizer: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip: float = 1.0

    # schedule (warmup + cosine to min_lr_ratio)
    max_steps: int = 100_000
    warmup_steps: int = 2_000
    lr_decay_steps: int | None = None  # defaults to max_steps
    min_lr_ratio: float = 0.1

    # precision / memory
    precision: str = "bf16"  # one of: fp32, bf16, fp16
    grad_ckpt_recurrent: bool | None = None  # None ⇒ inherit from RDTConfig
    grad_ckpt_prelude_coda: bool | None = None
    bptt_window: int | None = None
    use_loss_chunking: bool = True

    # recurrent step schedule (curriculum)
    recurrent_steps_start: int | None = None  # if set, ramps to cfg.recurrent_steps
    recurrent_steps_ramp: int = 0

    # distributed
    dist_backend: str = "nccl"  # nccl | gloo
    parallel: str = "single"  # single | ddp | fsdp
    fsdp_mixed_precision: str = "bf16"
    fsdp_cpu_offload: bool = False

    # checkpoint
    output_dir: str = "outputs/run"
    save_every: int = 1000
    keep_last_n: int = 3
    resume: str = ""  # path to a checkpoint dir or ""

    # logging
    log_every: int = 10
    eval_every: int = 1000
    eval_max_batches: int = 32
    tensorboard: bool = True
    wandb_project: str = ""

    # reproducibility
    seed: int = 42

    def __post_init__(self) -> None:
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive")
        if self.grad_accum_steps <= 0:
            raise ValueError("grad_accum_steps must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not (0.0 <= self.min_lr_ratio <= 1.0):
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError(f"unknown precision: {self.precision}")
        if self.parallel not in {"single", "ddp", "fsdp"}:
            raise ValueError(f"unknown parallel mode: {self.parallel}")
        if self.lr_decay_steps is None:
            self.lr_decay_steps = self.max_steps
        if self.recurrent_steps_ramp < 0:
            raise ValueError("recurrent_steps_ramp must be non-negative")


@dataclass
class OMVTConfig:
    """Orientation-aware Multiscript Vision Tower config.

    All sizes are conservative defaults suitable for smoke tests; production
    pretraining should override ``image_size``, ``patch_sizes``, ``compress_to``
    based on data inspection.
    """

    image_size: int = 224
    in_channels: int = 3

    # multi-scale patch shapes (height, width)
    vertical_patch: tuple[int, int] = (32, 8)
    horizontal_patch: tuple[int, int] = (8, 32)
    square_patch: tuple[int, int] = (16, 16)
    layout_patch: tuple[int, int] = (56, 56)

    # mixer hyperparams
    d_vision: int = 512
    n_vertical_layers: int = 2
    n_horizontal_layers: int = 2
    n_local_attn_layers: int = 2
    n_layout_layers: int = 1
    vision_n_heads: int = 8
    vision_ffn_hidden: int = 2048
    vision_dropout: float = 0.0

    # router (geometric, no LM dependency)
    router_min_route_prob: float = 0.05
    router_temperature: float = 1.0

    # Perceiver compressor.
    # ``compress_to`` is the number of visual tokens emitted per image and must
    # equal the number of ``<image_patch>`` slots the data builder writes per
    # image (the OMVT injector asserts ``count == compress_to`` at forward).
    # The training CLIs derive it from ``image_patch_count(image_size,...)`` via
    # ``build_omvt_cfg``; this dataclass default is only a standalone fallback.
    compress_to: int = 256  # output visual tokens per image
    compressor_layers: int = 2
    compressor_heads: int = 8

    # projector to RDT embedding space
    projector_hidden: int = 0  # 0 ⇒ single Linear

    # SSL loss weights (Phase 1 vision pretraining)
    w_ocr: float = 1.0
    w_masked_patch: float = 1.0
    w_orientation: float = 0.2
    w_layout_order: float = 0.2

    # fraction of square patches hidden for the masked-patch reconstruction task
    mask_ratio: float = 0.5

    # joint training loss weights (Phase 3)
    w_lm: float = 1.0
    w_patch_text_contrastive: float = 0.1
    w_layout: float = 0.1

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.d_vision <= 0:
            raise ValueError("d_vision must be positive")
        if self.compress_to <= 0:
            raise ValueError("compress_to must be positive")
        if self.in_channels not in (1, 3):
            raise ValueError("in_channels must be 1 or 3")
        if self.d_vision % self.vision_n_heads != 0:
            raise ValueError("d_vision must be divisible by vision_n_heads")
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in (0, 1)")


def main() -> None:
    for name, make_cfg in [
        ("tiny", tiny_config),
        ("small", small_config),
        ("base", base_config),
        ("pretrain", pretrain_config),
    ]:
        cfg = make_cfg()
        print(f"{name}:")
        print(f"  d_model={cfg.d_model}")
        print(f"  actual_layers={cfg.actual_layers}")
        print(f"  effective_depth={cfg.effective_depth}")
        print(f"  recurrent_steps={cfg.recurrent_steps}")
        print(f"  vocab_size={cfg.vocab_size}")
        print()


if __name__ == "__main__":
    main()

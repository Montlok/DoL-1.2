# -*- coding: utf-8 -*-

"""HuggingFace ``PretrainedConfig`` wrapper for :class:`RDTConfig`.

This adapter exposes the native RDT dataclass config through the
``transformers`` config interface so the model can be loaded by HF tooling
(``AutoConfig``, ``Trainer``, PEFT, TRL, LLaMA-Factory). The full RDT dataclass
is round-tripped verbatim under the ``rdt_config`` key; a handful of standard
HF aliases (``vocab_size``, ``hidden_size``, ...) are mirrored so generic HF
utilities that read those attributes keep working.
"""

from dataclasses import asdict, fields

from transformers import PretrainedConfig

from Model.config import RDTConfig


class RDTHFConfig(PretrainedConfig):
    """``transformers`` config carrying a full :class:`RDTConfig` payload."""

    model_type = "rdt"

    def __init__(self, rdt_config: dict | None = None, **kwargs) -> None:
        base = asdict(RDTConfig()) if rdt_config is None else dict(rdt_config)
        self.rdt_config = base

        # Mirror common HF attributes from the RDT payload so generic helpers
        # (resize_token_embeddings, generation utils, ...) read sane values.
        self.vocab_size = base["vocab_size"]
        self.hidden_size = base["d_model"]
        self.num_attention_heads = base["n_heads"]
        self.num_hidden_layers = base["n_prelude"] + base["n_coda"]
        self.max_position_embeddings = base["max_seq_len"]

        # Default token ids feed HF generation / collators.
        kwargs.setdefault("pad_token_id", base["pad_id"])
        kwargs.setdefault("bos_token_id", base["bos_id"])
        kwargs.setdefault("eos_token_id", base["eos_id"])
        kwargs.setdefault("tie_word_embeddings", base["tie_word_embeddings"])

        super().__init__(**kwargs)

    def to_rdt_config(self) -> RDTConfig:
        """Reconstruct the native dataclass, dropping unknown/extra keys."""
        valid = {f.name for f in fields(RDTConfig)}
        payload = {k: v for k, v in self.rdt_config.items() if k in valid}
        return RDTConfig(**payload)

    @classmethod
    def from_rdt_config(cls, cfg: RDTConfig, **kwargs) -> "RDTHFConfig":
        """Build an HF config from a native :class:`RDTConfig` instance."""
        return cls(rdt_config=asdict(cfg), **kwargs)

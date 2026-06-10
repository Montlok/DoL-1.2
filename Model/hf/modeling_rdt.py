# -*- coding: utf-8 -*-

"""HuggingFace ``PreTrainedModel`` wrapper around :class:`RDTForCausalLM`.

The wrapper makes the native RDT model usable by the HF training/inference
ecosystem (``Trainer``, PEFT/LoRA, TRL). It is intentionally **cache-free** on
the HF path: ``forward`` recomputes the full prefix each step. This keeps the
adapter correct and simple; the fast incremental KV/state cache lives in the
native :meth:`RDTForCausalLM.generate` and is used for production inference and
the latent-depth ("think harder") knob.

The custom ``recurrent_steps`` argument is threaded through so HF callers can
still drive the latent-depth knob, but it must stay constant within a call.
"""

import torch
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from Model.model import RDTForCausalLM

from .configuration_rdt import RDTHFConfig


class RDTForCausalLMHF(PreTrainedModel, GenerationMixin):
    """HF-compatible causal-LM wrapper for the RDT model."""

    config_class = RDTHFConfig
    base_model_prefix = "rdt"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    _no_split_modules = ["StandardBlock"]

    def __init__(self, config: RDTHFConfig) -> None:
        super().__init__(config)
        self.rdt = RDTForCausalLM(config.to_rdt_config())
        # Native module already initializes and (optionally) ties weights;
        # avoid HF re-initialization clobbering that. ``_init_weights`` is a
        # no-op so any HF-internal init pass is harmless.

        # Declare tied-weight keys so save_pretrained drops the duplicates and
        # from_pretrained re-ties them via ``_tie_weights``. transformers>=5
        # expects a mapping {tied_param_pattern: source_param}.
        keys: dict[str, str] = {}
        if self.config.tie_word_embeddings:
            keys["rdt.lm_head.weight"] = "rdt.embed.weight"
            if self.rdt.reverse_head is not None:
                keys["rdt.reverse_head.weight"] = "rdt.embed.weight"
        self._tied_weights_keys = keys
        self.post_init()

    def _init_weights(self, module: torch.nn.Module) -> None:  # noqa: D401
        # The inner RDTForCausalLM performs its own initialization.
        return None

    def _tie_weights(self) -> None:
        if self.config.tie_word_embeddings:
            self.rdt.lm_head.weight = self.rdt.embed.weight
            if self.rdt.reverse_head is not None:
                self.rdt.reverse_head.weight = self.rdt.embed.weight

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.rdt.embed

    def set_input_embeddings(self, value: torch.nn.Module) -> None:
        self.rdt.embed = value

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.rdt.lm_head

    def set_output_embeddings(self, new_embeddings: torch.nn.Module) -> None:
        self.rdt.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        recurrent_steps: int | None = None,
        past_key_values=None,
        use_cache: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        out = self.rdt(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            steps=recurrent_steps,
            return_logits=True,
        )
        return CausalLMOutputWithPast(
            loss=out["loss"],
            logits=out["logits"],
            past_key_values=None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        # Cache-free path: always re-feed the full prefix.
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if kwargs.get("recurrent_steps") is not None:
            model_inputs["recurrent_steps"] = kwargs["recurrent_steps"]
        return model_inputs

# -*- coding: utf-8 -*-

"""HuggingFace compatibility adapter for the RDT model.

Importing this package registers the RDT architecture with the ``Auto*``
factories so ``AutoConfig.from_pretrained`` / ``AutoModelForCausalLM`` can load
RDT checkpoints (with ``trust_remote_code`` when distributed externally).
"""

from transformers import AutoConfig, AutoModelForCausalLM

from .configuration_rdt import RDTHFConfig
from .modeling_rdt import RDTForCausalLMHF

# Registration is idempotent across repeated imports within one process.
try:
    AutoConfig.register("rdt", RDTHFConfig)
    AutoModelForCausalLM.register(RDTHFConfig, RDTForCausalLMHF)
except ValueError:
    pass

__all__ = ["RDTHFConfig", "RDTForCausalLMHF"]

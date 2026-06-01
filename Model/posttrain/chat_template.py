# -*- coding: utf-8 -*-

"""Versioned plain-text chat template for SFT / alignment.

The template uses **plain-text sentinels** only, so it requires no new
vocabulary entries and does not touch already-trained tokenizer bundles
(the agreed route). Role and reasoning/tool markers are ordinary strings that
the existing tokenizer encodes as regular text.

Supervision policy (see :mod:`Model.posttrain.sft_data`):
- ``assistant`` content is supervised (the model learns to produce it),
  including any inline ``<think>...</think>`` and ``<tool_call>...</tool_call>``
  spans — those are emitted by the model.
- ``system`` / ``user`` headers+content and the ``tool`` (tool-result) message
  are masked: they are prompt / externally injected and must never enter the
  loss.
- Role headers (e.g. ``<|assistant|>\n``) are scaffolding and are masked; only
  the assistant *content* and its terminating EOS are supervised.
"""

from __future__ import annotations

from dataclasses import dataclass

SFT_TEMPLATE_VERSION = "v1"

# Plain-text sentinels (no dedicated vocab ids).
ROLE_SENTINELS = {
    "system": "<|system|>\n",
    "user": "<|user|>\n",
    "assistant": "<|assistant|>\n",
    "tool": "<|tool|>\n",
}
TURN_SUFFIX = "\n"

# Reasoning / tool markers, documented here so data tooling and the inference
# orchestration layer agree on the exact strings.
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_RESULT_OPEN = "<tool_result>"
TOOL_RESULT_CLOSE = "</tool_result>"

_SUPERVISED_ROLES = {"assistant"}


@dataclass(frozen=True)
class Span:
    """A rendered text span and whether it should be supervised (loss-bearing)."""

    text: str
    supervised: bool


def render_message(role: str, content: str) -> list[Span]:
    """Render one chat message into masked header + (maybe) supervised content.

    Args:
        role: one of ``system`` / ``user`` / ``assistant`` / ``tool``.
        content: message text. For ``tool`` it is the tool result payload.

    Returns:
        Ordered spans. The role header is always masked; assistant content is
        supervised, everything else is masked.
    """
    if role not in ROLE_SENTINELS:
        raise ValueError(f"unknown role: {role!r}")
    header = Span(ROLE_SENTINELS[role], supervised=False)
    supervised = role in _SUPERVISED_ROLES
    # Assistant turns terminate with EOS (added by the builder), so they carry
    # no trailing separator; other roles use a newline separator.
    text = content if supervised else content + TURN_SUFFIX
    body = Span(text, supervised=supervised)
    return [header, body]


def render_conversation(
    messages: list[dict[str, str]],
    add_generation_prompt: bool = False,
) -> list[Span]:
    """Render a conversation into ordered spans with supervision flags.

    Args:
        messages: list of ``{"role": ..., "content": ...}``.
        add_generation_prompt: if True, append a trailing masked
            ``<|assistant|>\n`` header to prompt for generation (inference).
    """
    spans: list[Span] = []
    for msg in messages:
        spans.extend(render_message(msg["role"], msg["content"]))
    if add_generation_prompt:
        spans.append(Span(ROLE_SENTINELS["assistant"], supervised=False))
    return spans


def render_text(
    messages: list[dict[str, str]],
    add_generation_prompt: bool = False,
) -> str:
    """Flatten a conversation to its plain-text rendering (for inspection)."""
    return "".join(
        s.text for s in render_conversation(messages, add_generation_prompt)
    )

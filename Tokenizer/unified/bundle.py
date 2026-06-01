# -*- coding: utf-8 -*-
"""Persisted tokenizer bundle for reproducible pretraining data builds."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass

from Tokenizer.generic_bpe import GeneralBPEModel
from Tokenizer.morphbpe import MorphBPETokenizer
from Tokenizer.multimodal import MultimodalProcessor
from Tokenizer.traditional_mongolian.stemmer import MongolStemmer

from .dual_tokenizer import DualTrackTokenizer
from .vocab import SPECIAL_TOKENS, build_unified_vocab


BUNDLE_VERSION = 2
CONFIG_NAME = "config.json"
MORPHBPE_NAME = "morphbpe.json"
GENERAL_NAME = "general.json"
VOCAB_NAME = "vocab.json"


@dataclass
class TokenizerBundleConfig:
    version: int
    morphbpe_file: str
    general_file: str = ""
    patch_size: int = 14
    merge_size: int = 2
    temporal_patch_size: int = 2


class TokenizerBundle:
    tokenizer: DualTrackTokenizer
    processor: MultimodalProcessor
    config: TokenizerBundleConfig

    def __init__(
        self,
        tokenizer: DualTrackTokenizer,
        processor: MultimodalProcessor,
        config: TokenizerBundleConfig,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

    @classmethod
    def from_files(
        cls,
        morphbpe_path: str,
        general_path: str | None = None,
        patch_size: int = 14,
        merge_size: int = 2,
        temporal_patch_size: int = 2,
    ) -> "TokenizerBundle":
        config = TokenizerBundleConfig(
            version=BUNDLE_VERSION,
            morphbpe_file=morphbpe_path,
            general_file=general_path or "",
            patch_size=patch_size,
            merge_size=merge_size,
            temporal_patch_size=temporal_patch_size,
        )
        return cls._build(config, morphbpe_path=morphbpe_path, vocab=None)

    @classmethod
    def from_dir(cls, path: str) -> "TokenizerBundle":
        config_path = os.path.join(path, CONFIG_NAME)
        vocab_path = os.path.join(path, VOCAB_NAME)
        with open(config_path, "r", encoding="utf-8") as f:
            config = TokenizerBundleConfig(**json.load(f))
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = {str(token): int(idx) for token, idx in json.load(f).items()}
        morphbpe_path = os.path.join(path, config.morphbpe_file)
        general_path = (
            os.path.join(path, config.general_file) if config.general_file else None
        )
        return cls._build(
            config,
            morphbpe_path=morphbpe_path,
            vocab=vocab,
            general_path=general_path,
        )

    @classmethod
    def _build(
        cls,
        config: TokenizerBundleConfig,
        morphbpe_path: str,
        vocab: dict[str, int] | None,
        general_path: str | None = None,
    ) -> "TokenizerBundle":
        stemmer = MongolStemmer()
        morphbpe = MorphBPETokenizer.from_file(morphbpe_path, stemmer)

        gen_path = general_path if general_path is not None else config.general_file
        if gen_path:
            general = GeneralBPEModel.load(gen_path)
        else:
            general = GeneralBPEModel.minimal()

        if vocab is None:
            vocab = build_unified_vocab(morphbpe.vocab, general.get_vocab())

        tokenizer = DualTrackTokenizer(vocab, morphbpe, general)
        processor = MultimodalProcessor(
            tokenizer,
            patch_size=config.patch_size,
            merge_size=config.merge_size,
            temporal_patch_size=config.temporal_patch_size,
        )
        return cls(tokenizer=tokenizer, processor=processor, config=config)

    def save_dir(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

        dest_morphbpe = os.path.join(path, MORPHBPE_NAME)
        source_morphbpe = self.config.morphbpe_file
        if os.path.abspath(source_morphbpe) != os.path.abspath(dest_morphbpe):
            if os.path.exists(source_morphbpe):
                shutil.copyfile(source_morphbpe, dest_morphbpe)
            else:
                self.tokenizer.morphbpe.save(dest_morphbpe)

        dest_general = os.path.join(path, GENERAL_NAME)
        source_general = self.config.general_file
        if (
            source_general
            and os.path.exists(source_general)
            and os.path.abspath(source_general) != os.path.abspath(dest_general)
        ):
            shutil.copyfile(source_general, dest_general)
        else:
            self.tokenizer.general.save(dest_general)

        config = TokenizerBundleConfig(
            version=self.config.version,
            morphbpe_file=MORPHBPE_NAME,
            general_file=GENERAL_NAME,
            patch_size=self.config.patch_size,
            merge_size=self.config.merge_size,
            temporal_patch_size=self.config.temporal_patch_size,
        )
        with open(os.path.join(path, CONFIG_NAME), "w", encoding="utf-8") as f:
            json.dump(asdict(config), f, ensure_ascii=False, indent=2)
        with open(os.path.join(path, VOCAB_NAME), "w", encoding="utf-8") as f:
            json.dump(self.tokenizer.vocab, f, ensure_ascii=False, indent=2)

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]:
        return self.tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)

    def encode_with_spans(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ):
        return self.tokenizer.encode_with_spans(text, add_bos=add_bos, add_eos=add_eos)

    def encode_multimodal(
        self,
        text: str,
        images=None,
        image_sizes=None,
        videos=None,
        video_sizes=None,
        add_bos: bool = False,
        add_eos: bool = False,
    ):
        return self.processor(
            text,
            images=images,
            image_sizes=image_sizes,
            videos=videos,
            video_sizes=video_sizes,
            add_bos=add_bos,
            add_eos=add_eos,
        )

    def validate(self) -> list[str]:
        issues: list[str] = []
        if self.config.version != BUNDLE_VERSION:
            issues.append(f"unsupported config version: {self.config.version}")
        if not self.config.morphbpe_file:
            issues.append("config.morphbpe_file is empty")
        values = list(self.tokenizer.vocab.values())
        if len(values) != len(set(values)):
            issues.append("vocab contains duplicate ids")
        for token, expected_id in SPECIAL_TOKENS.items():
            actual = self.tokenizer.vocab.get(token)
            if actual != expected_id:
                issues.append(
                    f"special token {token!r} has id {actual}, expected {expected_id}"
                )
        if "<unk>" not in self.tokenizer.morphbpe.vocab:
            issues.append("morphbpe vocab is missing <unk>")
        try:
            text = "\u182e\u1822\u1828\u182d\u1822\u182f 文字 test \U0001f642"
            result = self.encode_with_spans(text, add_bos=True, add_eos=True)
            if len(result.input_ids) != len(result.tokens):
                issues.append("encode_with_spans produced mismatched ids/tokens")
        except Exception as exc:  # pragma: no cover - reported as validation issue.
            issues.append(f"encode smoke failed: {exc}")
        try:
            mm = self.encode_multimodal(
                "文字 <image> test",
                images=["smoke-image"],
                image_sizes=[(14, 14)],
            )
            if len(mm.input_ids) != len(mm.attention_mask):
                issues.append("multimodal smoke produced mismatched ids/mask")
            if not mm.image_token_spans:
                issues.append("multimodal smoke produced no image span")
        except Exception as exc:  # pragma: no cover - reported as validation issue.
            issues.append(f"multimodal smoke failed: {exc}")
        return issues

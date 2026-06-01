# Tokenizer Architecture

## Current Status

The tokenizer is organized as a routed two-track design. It keeps one global
token id space, segments input by route, and records token-level offsets for
downstream alignment. The two content tracks are a morphology-aware MorphBPE
track for traditional Mongolian and a general multilingual byte-level BPE track
for everything else.

## Module Responsibilities

- `Encoding Mapping/` is the Rust normalization layer before tokenizer ingest.
  Its `normalize_to_nominal_unicode` path folds Menksoft PUA, MW encodings, and
  Unicode presentation variants into nominal Unicode while preserving MVS suffix
  separators. A future PyO3 bridge can call it before Python tokenization.
- `Tokenizer/traditional_mongolian/` owns Unicode control stripping, suffix
  inventories, reverse stemming, and morpheme boundaries used by MorphBPE.
- `Tokenizer/morphbpe/` implements a boundary-constrained BPE track for
  traditional Mongolian. It uses `MongolStemmer.analyze()` boundaries to prevent
  merges across root/suffix boundaries, and seeds the full Mongolian letter
  alphabet so valid unseen letters still encode at character level instead of
  falling to `<unk>`.
- `Tokenizer/generic_bpe/` wraps the general byte-level BPE (`GeneralBPEModel` /
  `GeneralBPETrainer`, backed by HuggingFace `tokenizers`) plus byte-fallback
  helpers. Byte-level coverage means no `<unk>` for any non-Mongolian script.
- `Tokenizer/unified/` is the routed tokenizer: Mongolian block -> MorphBPE,
  everything else (Chinese, English, Japanese, Cyrillic, digits, punctuation,
  symbols, `\n`/`\t`) -> general byte-level BPE, specials -> fixed ids, plain
  spaces -> the `▁` space token.
- `Tokenizer/multimodal/` expands image/video-style placeholders and records
  multimodal token spans. The tokenizer does not process vision features.

## Token ID Layout

```python
SEGMENT = {
    "special": (0, 256),
    "mongolian": (256, 24576),
    "general": (24576, 65536),
}
```

Core special ids: `<pad>` 0, `<unk>` 1, `<bos>` 2, `<eos>` 3, `<img>` 4,
`<image>` 5, `<image_start>` 6, `<image_patch>` 7, `<image_end>` 8,
`<video>` 9, `<video_start>` 10, `<video_patch>` 11, `<video_end>` 12,
`<bbox>` 13, `<ocr>` 14, `<ocr_start>` 15, `<ocr_end>` 16, `▁` 17, `◈` 18.

## Offset Contract

`encode_with_spans()` returns `DualTrackResult`:

- `input_ids`: token ids.
- `tokens`: `EncodedToken(id, token, track, start, end)` for every emitted id.
- `spans`: language/special routing spans.
- `attention_mask`: defaults to one per input id.

Offsets are Python string offsets into the input passed to the tokenizer.
`<bos>` and `<eos>` use `-1:-1`. The general byte-level BPE track uses the
ByteLevel offset mapping, which is char-accurate; byte fallback (only reachable
when a track has no model) repeats the original character span per byte token.

## Multimodal Contract

`MultimodalProcessor` expands `<image>` into
`<image_start><image_patch>*N<image_end>`, tokenizes the expanded text, and
returns `image_token_spans` pointing into `input_ids`. The processor may call an
external `image_processor`, but the tokenizer itself only provides placeholder
ids and spans; it does not create `pixel_values` or vision features.

## Not Yet Implemented / Future Work

- PyO3 bridge from Python tokenizer ingest to the Rust `Encoding Mapping`
  normalizer.
- Production-trained MorphBPE vocab and merge artifacts.
- Full persisted unified-tokenizer load path with real HF tokenizer metadata.
- Video placeholder expansion beyond token definitions.
- Rich OCR/bbox serialization.

## How to run tests

```bash
python3 -m unittest discover Tokenizer
cd "Encoding Mapping" && cargo test && cargo fmt --check
```

## Related documents

- [offset_contract.md](offset_contract.md)
- [multimodal_contract.md](multimodal_contract.md)
- [training_pipeline.md](training_pipeline.md)
- [external_implementation_notes.md](external_implementation_notes.md)

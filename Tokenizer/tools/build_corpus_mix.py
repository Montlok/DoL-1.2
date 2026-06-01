# -*- coding: utf-8 -*-

"""Token-weighted corpus mixer for Daffodils pretraining.

The pretraining pipeline historically just concatenated every corpus and packed
it, so language/domain proportions were whatever the raw byte counts happened to
be. This tool instead hits *target* token proportions:

  1. Read a manifest of sources, each tagged with ``lang``, ``domain`` and a
     relative ``weight`` (see ``--manifest`` schema below).
  2. Measure each source's available tokens (sampled encode -> tokens/char,
     extrapolated by the full character count, so tens of GB stay tractable).
  3. Solve for a token budget per source from the weights, then up-sample
     (repeat, capped at ``max_epochs`` to avoid overfitting low-resource data)
     or sub-sample each source to meet its target.
  4. Stream a shuffled, text-only ``{"text": ...}`` JSONL ready for
     ``build_pretraining_data``, and write a JSON token-accounting report that
     compares the realized mixture against the target.

Because targets are recomputed from live measurements every run, dropping new
Mongolian corpora into a source directory automatically rebalances the mix.

Manifest schema (JSON)::

    {
      "total_tokens": 2000000000,        # optional overall budget
      "max_epochs": 4,                   # optional upsampling cap (default 4)
      "seed": 0,                         # optional shuffle seed
      "sources": [
        {"path": "corpus/mn.jsonl", "lang": "mn", "domain": "literature",
         "weight": 0.52, "text_column": "text"},
        {"path": "corpus/finemath/*.parquet", "lang": "en", "domain": "math",
         "weight": 0.12, "format": "parquet"}
      ]
    }

``weight`` values are normalized, so they need not sum to 1. ``path`` may be a
``.jsonl``/``.txt`` file, a ``.parquet`` file, a directory of parquet shards, or
a glob. ``format`` is inferred from the extension when omitted.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Iterator, Optional

_DEFAULT_MAX_EPOCHS = 4
_DEFAULT_SAMPLE_DOCS = 2000
_MIN_CHARS = 1


def _clean(text: str) -> str:
    return (text or "").strip()


def _infer_format(path: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    low = path.lower()
    if low.endswith(".parquet"):
        return "parquet"
    if low.endswith((".jsonl", ".json")):
        return "jsonl"
    if low.endswith(".txt"):
        return "txt"
    # A directory or glob: peek for parquet shards.
    if os.path.isdir(path) and glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True):
        return "parquet"
    if glob.glob(path) and all(p.endswith(".parquet") for p in glob.glob(path)):
        return "parquet"
    return "jsonl"


def _resolve_parquet_shards(path: str) -> list[str]:
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))
    if path.endswith(".parquet") and os.path.isfile(path):
        return [path]
    return sorted(p for p in glob.glob(path) if p.endswith(".parquet"))


def _resolve_text_files(path: str) -> list[str]:
    """Resolve a jsonl/txt source to concrete files (file, dir, or glob).

    Mirrors the parquet handling so a manifest may point ``path`` at a single
    file, a directory of shards, or a glob pattern.
    """
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        return sorted(
            p
            for ext in ("*.jsonl", "*.json", "*.txt")
            for p in glob.glob(os.path.join(path, "**", ext), recursive=True)
        )
    return sorted(glob.glob(path))


def _resolve_source_files(path: str, fmt: str) -> list[str]:
    """Resolve a source to its concrete backing files (parquet or text)."""
    if fmt == "parquet":
        return _resolve_parquet_shards(path)
    return _resolve_text_files(path)


def _path_fingerprint(files: list[str]) -> list[tuple[str, int, int]]:
    """Cheap (path, size, mtime_ns) fingerprint without reading file contents."""
    fp = []
    for p in files:
        try:
            st = os.stat(p)
            fp.append((p, st.st_size, st.st_mtime_ns))
        except OSError:
            fp.append((p, -1, -1))
    return fp


def _compute_signature(
    manifest_bytes: bytes,
    cfg: "MixConfig",
    bundle_path: Optional[str],
) -> str:
    """Hash the manifest, every resolved source file's (size, mtime) and the
    tokenizer bundle so the cached mix is invalidated whenever any input that
    affects the mixture changes (manifest edits, swapped/added/removed shards,
    or a retrained tokenizer).
    """
    h = hashlib.sha256()
    h.update(manifest_bytes)
    for spec in cfg.sources:
        files = _resolve_source_files(spec.path, spec.fmt)
        h.update(repr((spec.lang, spec.domain, spec.weight, spec.text_column)).encode())
        h.update(repr(_path_fingerprint(files)).encode())
    h.update(repr((cfg.total_tokens, cfg.max_epochs, cfg.seed)).encode())
    if bundle_path:
        cfg_json = os.path.join(bundle_path, "config.json")
        h.update(repr(_path_fingerprint([cfg_json])).encode())
    return h.hexdigest()


def _iter_source_texts(path: str, fmt: str, text_column: str) -> Iterator[str]:
    """Yield cleaned, non-empty texts from a source in any supported format."""
    if fmt == "parquet":
        import pyarrow.parquet as pq

        shards = _resolve_parquet_shards(path)
        if not shards:
            raise SystemExit(f"No parquet shards found under {path!r}")
        for shard in shards:
            pf = pq.ParquetFile(shard)
            if text_column not in pf.schema_arrow.names:
                raise SystemExit(
                    f"Column {text_column!r} not in {shard!r}; "
                    f"available: {pf.schema_arrow.names}"
                )
            for batch in pf.iter_batches(batch_size=1000, columns=[text_column]):
                for value in batch.column(text_column):
                    text = _clean(str(value.as_py() or ""))
                    if len(text) >= _MIN_CHARS:
                        yield text
    elif fmt == "txt":
        files = _resolve_text_files(path)
        if not files:
            raise SystemExit(f"No text files found under {path!r}")
        for fp in files:
            with open(fp, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    text = _clean(line)
                    if len(text) >= _MIN_CHARS:
                        yield text
    else:  # jsonl
        files = _resolve_text_files(path)
        if not files:
            raise SystemExit(f"No jsonl files found under {path!r}")
        for fp in files:
            with open(fp, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = _clean(str(obj.get(text_column, "") or ""))
                    if len(text) >= _MIN_CHARS:
                        yield text


@dataclass
class SourceSpec:
    path: str
    lang: str = "unknown"
    domain: str = "general"
    weight: float = 1.0
    text_column: str = "text"
    fmt: str = ""

    @classmethod
    def from_raw(cls, raw: dict) -> "SourceSpec":
        if "path" not in raw:
            raise ValueError(f"source entry missing 'path': {raw!r}")
        return cls(
            path=raw["path"],
            lang=raw.get("lang", "unknown"),
            domain=raw.get("domain", "general"),
            weight=float(raw.get("weight", 1.0)),
            text_column=raw.get("text_column", "text"),
            fmt=_infer_format(raw["path"], raw.get("format")),
        )


@dataclass
class MixConfig:
    sources: list[SourceSpec]
    total_tokens: Optional[int] = None
    max_epochs: float = _DEFAULT_MAX_EPOCHS
    seed: int = 0

    @classmethod
    def from_raw(cls, raw: dict) -> "MixConfig":
        sources = [SourceSpec.from_raw(s) for s in raw.get("sources", [])]
        if not sources:
            raise ValueError("manifest has no sources")
        return cls(
            sources=sources,
            total_tokens=raw.get("total_tokens"),
            max_epochs=float(raw.get("max_epochs", _DEFAULT_MAX_EPOCHS)),
            seed=int(raw.get("seed", 0)),
        )


@dataclass
class SourceStats:
    spec: SourceSpec
    docs: int = 0
    chars: int = 0
    sampled_docs: int = 0
    sampled_tokens: int = 0
    est_tokens: int = 0
    # Filled in by the planner:
    target_tokens: float = 0.0
    repeat_factor: float = 0.0
    emit_prob: float = 1.0


def measure_source(
    spec: SourceSpec, encode, sample_docs: int = _DEFAULT_SAMPLE_DOCS
) -> SourceStats:
    """Count docs/chars and estimate total tokens via a sampled encode.

    ``encode`` maps text -> token ids. To keep huge sources tractable we encode
    only the first ``sample_docs`` documents and scale tokens by the ratio of
    total characters to sampled characters.
    """
    stats = SourceStats(spec=spec)
    sampled_chars = 0
    for text in _iter_source_texts(spec.path, spec.fmt, spec.text_column):
        stats.docs += 1
        clen = len(text)
        stats.chars += clen
        if stats.sampled_docs < sample_docs:
            stats.sampled_docs += 1
            sampled_chars += clen
            stats.sampled_tokens += len(encode(text))

    if stats.docs == 0:
        stats.est_tokens = 0
    elif sampled_chars > 0:
        tokens_per_char = stats.sampled_tokens / sampled_chars
        stats.est_tokens = int(round(stats.chars * tokens_per_char))
    else:
        stats.est_tokens = stats.sampled_tokens
    return stats


def compute_plan(
    stats: list[SourceStats],
    total_tokens: Optional[int],
    max_epochs: float,
) -> list[SourceStats]:
    """Assign each source a token budget from its weight and solve repeat/subsample.

    The realized budget honors the ``max_epochs`` upsampling cap: a source can
    contribute at most ``max_epochs * est_tokens``. When a target exceeds that
    cap the deficit is reported (so callers can see under-served weights) but the
    source is never repeated beyond the cap.
    """
    total_weight = sum(s.spec.weight for s in stats) or 1.0
    available = sum(s.est_tokens for s in stats)
    budget = total_tokens if total_tokens else available

    for s in stats:
        share = s.spec.weight / total_weight
        s.target_tokens = share * budget
        if s.est_tokens <= 0:
            s.repeat_factor = 0.0
            s.emit_prob = 0.0
            continue
        desired = s.target_tokens / s.est_tokens
        if desired <= 1.0:
            # Sub-sample: emit each doc with probability ``desired``.
            s.repeat_factor = desired
            s.emit_prob = desired
        else:
            # Up-sample with a hard epoch cap.
            s.repeat_factor = min(desired, max_epochs)
            s.emit_prob = s.repeat_factor / math.ceil(s.repeat_factor)
    return stats


def _emit_counts(repeat_factor: float, rng: random.Random) -> int:
    """Turn a fractional repeat factor into an integer emission count.

    ``repeat_factor`` < 1 => Bernoulli keep; >= 1 => floor plus a Bernoulli for
    the fractional part, so the expected count equals ``repeat_factor``.
    """
    if repeat_factor <= 0:
        return 0
    base = int(repeat_factor)
    frac = repeat_factor - base
    count = base
    if rng.random() < frac:
        count += 1
    return count


def emit_mix(
    stats: list[SourceStats],
    out_path: str,
    seed: int = 0,
    buffer_size: int = 100000,
) -> dict:
    """Stream the weighted, shuffled mixture to ``out_path`` (text-only JSONL).

    A bounded shuffle buffer keeps memory flat on huge corpora while still
    interleaving sources. Returns a token-accounting report.
    """
    rng = random.Random(seed)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    realized = {s.spec.path: {"docs": 0, "tokens_est": 0.0} for s in stats}
    buffer: list[str] = []

    def flush_some(force: bool, fh) -> None:
        if force:
            rng.shuffle(buffer)
            for item in buffer:
                fh.write(item)
                fh.write("\n")
            buffer.clear()
        elif len(buffer) >= buffer_size:
            rng.shuffle(buffer)
            half = len(buffer) // 2
            for item in buffer[:half]:
                fh.write(item)
                fh.write("\n")
            del buffer[:half]

    total_emitted_docs = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        # Round-robin across sources so the bounded shuffle buffer always holds a
        # mixture: pulling one doc per source per cycle means a mid-stream flush
        # still interleaves every active source, instead of dumping a solid run
        # of whichever source happened to fill the buffer first.
        active = []
        for s in stats:
            if s.repeat_factor <= 0:
                continue
            tokens_per_doc = s.est_tokens / s.docs if s.docs else 0.0
            it = _iter_source_texts(s.spec.path, s.spec.fmt, s.spec.text_column)
            active.append((s, tokens_per_doc, it))

        while active:
            still_active = []
            for s, tokens_per_doc, it in active:
                try:
                    text = next(it)
                except StopIteration:
                    continue
                still_active.append((s, tokens_per_doc, it))
                n = _emit_counts(s.repeat_factor, rng)
                for _ in range(n):
                    buffer.append(json.dumps({"text": text}, ensure_ascii=False))
                    realized[s.spec.path]["docs"] += 1
                    realized[s.spec.path]["tokens_est"] += tokens_per_doc
                    total_emitted_docs += 1
                flush_some(False, fh)
            active = still_active
        flush_some(True, fh)

    report = {
        "output": out_path,
        "total_emitted_docs": total_emitted_docs,
        "max_epochs": None,
        "sources": [],
    }
    grand_tokens = sum(r["tokens_est"] for r in realized.values()) or 1.0
    for s in stats:
        r = realized[s.spec.path]
        report["sources"].append(
            {
                "path": s.spec.path,
                "lang": s.spec.lang,
                "domain": s.spec.domain,
                "weight": s.spec.weight,
                "available_docs": s.docs,
                "available_tokens_est": s.est_tokens,
                "target_tokens": round(s.target_tokens),
                "repeat_factor": round(s.repeat_factor, 4),
                "emitted_docs": r["docs"],
                "emitted_tokens_est": round(r["tokens_est"]),
                "realized_share": round(r["tokens_est"] / grand_tokens, 4),
            }
        )
    # Aggregate realized shares by language and domain.
    by_lang: dict[str, float] = {}
    by_domain: dict[str, float] = {}
    for s, src in zip(stats, report["sources"]):
        by_lang[s.spec.lang] = by_lang.get(s.spec.lang, 0.0) + src["realized_share"]
        by_domain[s.spec.domain] = (
            by_domain.get(s.spec.domain, 0.0) + src["realized_share"]
        )
    report["by_lang"] = {k: round(v, 4) for k, v in by_lang.items()}
    report["by_domain"] = {k: round(v, 4) for k, v in by_domain.items()}
    return report


def _build_encoder(bundle_path: Optional[str]):
    """Return a text->ids encoder. Falls back to a whitespace/char proxy."""
    if bundle_path:
        from Tokenizer.unified.bundle import TokenizerBundle

        bundle = TokenizerBundle.from_dir(bundle_path)
        return lambda text: bundle.encode(text)
    # Proxy: approximate tokens as max(words, chars/4) so weighting still works
    # without a trained tokenizer (useful for smoke tests).
    return lambda text: [0] * max(len(text.split()), len(text) // 4, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="mix manifest JSON")
    parser.add_argument("--output", required=True, help="combined corpus JSONL")
    parser.add_argument(
        "--tokenizer-bundle",
        default=None,
        help="bundle dir for exact token measurement (else char proxy)",
    )
    parser.add_argument(
        "--report", default=None, help="write JSON token-accounting report here"
    )
    parser.add_argument(
        "--sample-docs",
        type=int,
        default=_DEFAULT_SAMPLE_DOCS,
        help="docs per source to encode for the tokens/char estimate",
    )
    parser.add_argument(
        "--measure-only",
        action="store_true",
        help="print per-source measurements and the plan, do not emit",
    )
    parser.add_argument(
        "--skip-if-fresh",
        action="store_true",
        help=(
            "skip rebuilding when --output exists and --report carries a "
            "signature matching the current manifest/sources/tokenizer state"
        ),
    )
    args = parser.parse_args()

    with open(args.manifest, "rb") as f:
        manifest_bytes = f.read()
    cfg = MixConfig.from_raw(json.loads(manifest_bytes))

    # Cache check: bail out before the expensive measure/emit when nothing that
    # affects the mixture has changed since the report was last written.
    signature = _compute_signature(manifest_bytes, cfg, args.tokenizer_bundle)
    if args.skip_if_fresh and args.report and os.path.exists(args.output):
        try:
            with open(args.report, "r", encoding="utf-8") as f:
                prev = json.load(f)
        except (OSError, json.JSONDecodeError):
            prev = {}
        if prev.get("signature") == signature:
            print(json.dumps({"skipped": True, "reason": "fresh"}))
            return

    encode = _build_encoder(args.tokenizer_bundle)
    stats = [measure_source(s, encode, args.sample_docs) for s in cfg.sources]
    stats = compute_plan(stats, cfg.total_tokens, cfg.max_epochs)

    if args.measure_only:
        for s in stats:
            print(
                f"{s.spec.lang:>8} {s.spec.domain:>12} "
                f"docs={s.docs} est_tokens={s.est_tokens} "
                f"target={round(s.target_tokens)} repeat={round(s.repeat_factor, 3)} "
                f"<- {s.spec.path}"
            )
        return

    report = emit_mix(stats, args.output, seed=cfg.seed)
    report["max_epochs"] = cfg.max_epochs
    report["signature"] = signature
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: report[k] for k in ("output", "total_emitted_docs", "by_lang", "by_domain")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-

"""OCR accuracy metrics for traditional Mongolian.

The core measurement problem for Mongolian OCR is that the same *visual* text
can map to several different code-point sequences: free variation selectors
(FVS, U+180B-180D), the Mongolian vowel separator (MVS, U+180E), narrow no-break
space (NNBSP, U+202F) and positional/presentation variants. Comparing raw
code points therefore conflates genuine recognition errors with harmless
encoding/rendering differences.

We report two character error rates:

- **normalized CER** (primary): both prediction and reference are first folded to
  nominal Mongolian Unicode, then compared. Folding prefers the repository's Rust
  normalizer (``normalize_to_nominal_unicode`` via
  :mod:`Tokenizer.tools.normalize_mongolian`); when ``cargo`` is unavailable it
  falls back to a light Python folding that strips FVS/MVS/joiners. This keeps the
  metric meaningful (no rendering-noise penalty) without a hard Rust dependency.
- **raw CER** (secondary): compares the unmodified code points, exposing the true
  encoding gap.

Plus word accuracy (WER over whitespace tokens) and an exact line-match rate.

This module is intentionally torch-free so it can be unit-tested without loading
a model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# Code points that are pure encoding/rendering variation and must not count as
# recognition errors in the Python fallback folding.
_FVS = {0x180B, 0x180C, 0x180D, 0x180F}  # free variation selectors (incl. FVS4)
_MVS = {0x180E}  # Mongolian vowel separator
_JOINERS = {0x200C, 0x200D}  # ZWNJ / ZWJ
_BOM = {0xFEFF, 0xFFFE}
_NNBSP = 0x202F  # narrow no-break space -> regular space

_FALLBACK_DELETE = _FVS | _MVS | _JOINERS | _BOM


def _python_fold(text: str) -> str:
    """Light nominal folding used when the Rust normalizer is unavailable.

    Strips FVS/MVS/joiners/BOM and maps NNBSP to a regular space. This is a
    deliberately conservative subset of the Rust normalizer; it is *not* a full
    Menksoft/presentation-form normalizer.
    """
    out = []
    for ch in text:
        cp = ord(ch)
        if cp in _FALLBACK_DELETE:
            continue
        if cp == _NNBSP:
            out.append(" ")
            continue
        out.append(ch)
    return "".join(out)


def _rust_fold_batch(texts: Sequence[str]) -> list[str] | None:
    """Fold a batch through the Rust ``normalize_to_nominal_unicode``.

    Returns ``None`` (caller falls back) if cargo/the crate is unavailable or the
    batched round-trip does not preserve line count.
    """
    try:
        from Tokenizer.tools.normalize_mongolian import normalize
    except Exception:
        return None
    # Normalizing line-by-line would spawn one cargo process per sample; instead
    # join with newlines (which the normalizer passes through untouched) and
    # split back. Guard against any line-count drift.
    if any("\n" in t for t in texts):
        return None
    try:
        folded = normalize("\n".join(texts), nominal=True)
    except Exception:
        return None
    parts = folded.split("\n")
    if len(parts) != len(texts):
        return None
    return parts


def nominal_normalize(
    texts: Sequence[str], *, backend: str = "auto"
) -> list[str]:
    """Fold ``texts`` to nominal Mongolian Unicode for normalized CER.

    ``backend``: ``"auto"`` (Rust if available else Python fallback),
    ``"rust"`` (Rust only; raises if unavailable), or ``"python"`` (fallback only).
    """
    folded, _ = _fold_with_backend(texts, backend=backend)
    return folded


def _fold_with_backend(
    texts: Sequence[str], *, backend: str = "auto"
) -> tuple[list[str], str]:
    """Like :func:`nominal_normalize` but also returns the backend actually used."""
    texts = list(texts)
    if backend == "python":
        return [_python_fold(t) for t in texts], "python"
    if backend == "rust":
        folded = _rust_fold_batch(texts)
        if folded is None:
            raise RuntimeError(
                "rust normalizer unavailable (need cargo + 'Encoding Mapping' crate)"
            )
        return folded, "rust"
    if backend != "auto":
        raise ValueError(f"unknown backend {backend!r}")
    folded = _rust_fold_batch(texts)
    if folded is None:
        return [_python_fold(t) for t in texts], "python"
    return folded, "rust"


def _fold_pair(
    preds: Sequence[str], refs: Sequence[str], *, backend: str = "auto"
) -> tuple[list[str], list[str], str]:
    """Fold ``preds`` and ``refs`` through a *single* backend decision.

    Folding the two sides in separate calls could, under ``backend="auto"``,
    pick different backends (e.g. if only one side contains a newline that makes
    the batched Rust round-trip bail to the Python fallback), making the two
    sides normalized by different rules. Concatenating and folding once
    guarantees both sides share the same backend and folding.
    """
    preds = list(preds)
    refs = list(refs)
    combined, used = _fold_with_backend(preds + refs, backend=backend)
    n = len(preds)
    return combined[:n], combined[n:], used


def edit_distance(a: Sequence, b: Sequence) -> int:
    """Levenshtein edit distance between two sequences (O(len(a)*len(b)) time,
    O(min) space)."""
    if a is b or a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def _corpus_rate(
    preds: Sequence[Sequence], refs: Sequence[Sequence]
) -> tuple[float, int, int]:
    """Micro-averaged error rate: sum(edit distance) / sum(max(len(ref), 1)).

    Using ``max(len(ref), 1)`` per sample ensures pure-insertion errors against
    an empty reference are still reflected (a plain ``sum(len(ref))`` denominator
    would add the insertions to the numerator but nothing to the denominator,
    silently under-penalizing them — and would be 0/0 if every ref were empty).
    """
    total_dist = 0
    total_len = 0
    for p, r in zip(preds, refs):
        total_dist += edit_distance(p, r)
        total_len += max(len(r), 1)
    rate = total_dist / total_len if total_len else 0.0
    return rate, total_dist, total_len


def cer(
    preds: Sequence[str],
    refs: Sequence[str],
    *,
    normalize: bool = True,
    backend: str = "auto",
) -> float:
    """Corpus character error rate. When ``normalize`` is true, both sides are
    folded to nominal Unicode first (the primary, render-robust metric)."""
    if normalize:
        preds, refs, _ = _fold_pair(preds, refs, backend=backend)
    rate, _, _ = _corpus_rate(preds, refs)
    return rate


def wer(
    preds: Sequence[str],
    refs: Sequence[str],
    *,
    normalize: bool = True,
    backend: str = "auto",
) -> float:
    """Corpus word error rate over whitespace-split tokens."""
    if normalize:
        preds, refs, _ = _fold_pair(preds, refs, backend=backend)
    rate, _, _ = _corpus_rate([p.split() for p in preds], [r.split() for r in refs])
    return rate


@dataclass
class OCRReport:
    n: int
    norm_cer: float
    raw_cer: float
    wer: float
    line_exact: float
    rejection_rate: float
    backend: str


def ocr_report(
    preds: Sequence[str],
    refs: Sequence[str],
    *,
    backend: str = "auto",
    rejected: Sequence[bool] | None = None,
) -> OCRReport:
    """Full OCR quality report over aligned ``preds``/``refs``.

    ``rejected`` (optional): per-sample mask of predictions withheld by a
    confidence gate; only the *kept* samples are scored, and the rejection rate
    is reported separately (high-precision corpus ingestion).
    """
    preds = list(preds)
    refs = list(refs)
    if len(preds) != len(refs):
        raise ValueError(
            f"preds/refs length mismatch: {len(preds)} != {len(refs)}"
        )
    total = len(preds)
    if rejected is not None:
        rejected = list(rejected)
        if len(rejected) != total:
            raise ValueError("rejected mask length must match preds")
        keep = [i for i in range(total) if not rejected[i]]
    else:
        keep = list(range(total))
    rejection_rate = (total - len(keep)) / total if total else 0.0

    kp = [preds[i] for i in keep]
    kr = [refs[i] for i in keep]

    norm_p, norm_r, used_backend = _fold_pair(kp, kr, backend=backend)

    norm_cer, _, _ = _corpus_rate(norm_p, norm_r)
    raw_cer, _, _ = _corpus_rate(kp, kr)
    wer_rate, _, _ = _corpus_rate(
        [p.split() for p in norm_p], [r.split() for r in norm_r]
    )
    exact = sum(1 for p, r in zip(norm_p, norm_r) if p == r)
    line_exact = exact / len(keep) if keep else 0.0

    return OCRReport(
        n=len(keep),
        norm_cer=norm_cer,
        raw_cer=raw_cer,
        wer=wer_rate,
        line_exact=line_exact,
        rejection_rate=rejection_rate,
        backend=used_backend,
    )


__all__ = [
    "OCRReport",
    "cer",
    "edit_distance",
    "nominal_normalize",
    "ocr_report",
    "wer",
]

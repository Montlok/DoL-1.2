# -*- coding: utf-8 -*-

"""Extract a domain-filtered Chinese science/reasoning seed for MT synthesis.

The ``JournalArticle2013_2023`` bundle is millions of academic abstracts across
every field. For native-Mongolian STEM/reasoning synthesis we only want the
philosophy / math / physics / chemistry / general-science slice, deduplicated
and length-filtered, as clean translation units.

This tool scans the JSON files (each an array of records with ``media_c``,
``title_c``, ``keyword_c``, ``remark_c``), classifies each record into a domain
by keyword matching, and emits:

  * a ``.jsonl`` of ``{"text", "domain", "title"}`` (for the data pipeline), and
  * optionally a plain ``.txt`` (one abstract per line) for the MT translator.

Usage::

    python -m Tokenizer.tools.extract_science_seed \\
        --input "CHINESE(DO NOT GIT IT)/JournalArticle2013_2023" \\
        --output seed.jsonl --txt seed.txt \\
        --per-domain 2000 --min-chars 80
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Iterator, Optional

# Domain keyword sets. A record is tagged with the first domain whose keywords
# appear in its media/title/keyword fields. Order matters: more specific
# (philosophy/math/physics/chemistry) before the broad "science" catch-all.
DOMAIN_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("philosophy", ("哲学", "逻辑", "伦理", "形而上", "认识论", "辩证", "本体论")),
    ("math", ("数学", "代数", "几何", "微积分", "概率", "拓扑", "方程", "函数", "定理")),
    ("physics", ("物理", "力学", "量子", "热力学", "电磁", "相对论", "粒子", "光学")),
    ("chemistry", ("化学", "分子", "化合物", "反应", "催化", "有机", "无机", "电化学")),
    ("science", ("科学", "工程", "算法", "实验", "理论", "模型", "推理", "证明")),
]

_WS = re.compile(r"\s+")


def classify(text: str) -> Optional[str]:
    for domain, keywords in DOMAIN_KEYWORDS:
        if any(kw in text for kw in keywords):
            return domain
    return None


def _clean(text: str) -> str:
    return _WS.sub(" ", (text or "").replace("\n", " ")).strip()


def iter_records(input_dir: str) -> Iterator[dict]:
    files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    if not files:
        raise SystemExit(f"No JSON files under {input_dir!r}")
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8", errors="replace") as f:
                arr = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(arr, list):
            yield from (r for r in arr if isinstance(r, dict))


def extract(
    input_dir: str,
    per_domain: int,
    min_chars: int,
    max_chars: int,
) -> Iterator[dict]:
    """Yield ``{"text","domain","title"}`` seed rows, deduped, capped per domain."""
    counts: dict[str, int] = {}
    seen: set[str] = set()
    targets = {d for d, _ in DOMAIN_KEYWORDS}
    for rec in iter_records(input_dir):
        abstract = _clean(str(rec.get("remark_c", "")))
        if len(abstract) < min_chars or len(abstract) > max_chars:
            continue
        # Classify on the metadata first (precise), then fall back to abstract.
        meta = " ".join(
            str(rec.get(k, "")) for k in ("media_c", "keyword_c", "title_c")
        )
        domain = classify(meta) or classify(abstract)
        if domain is None:
            continue
        if counts.get(domain, 0) >= per_domain:
            if all(counts.get(d, 0) >= per_domain for d in targets):
                break
            continue
        key = abstract[:120]
        if key in seen:
            continue
        seen.add(key)
        counts[domain] = counts.get(domain, 0) + 1
        yield {
            "text": abstract,
            "domain": domain,
            "title": _clean(str(rec.get("title_c", ""))),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JournalArticle JSON dir")
    parser.add_argument("--output", required=True, help="output seed JSONL")
    parser.add_argument(
        "--txt", default=None, help="also write plain one-abstract-per-line txt"
    )
    parser.add_argument("--per-domain", type=int, default=2000)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--max-chars", type=int, default=2000)
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    by_domain: dict[str, int] = {}
    rows = 0
    txt_fh = open(args.txt, "w", encoding="utf-8") if args.txt else None
    try:
        with open(args.output, "w", encoding="utf-8") as out:
            for row in extract(
                args.input, args.per_domain, args.min_chars, args.max_chars
            ):
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                if txt_fh:
                    txt_fh.write(row["text"] + "\n")
                by_domain[row["domain"]] = by_domain.get(row["domain"], 0) + 1
                rows += 1
    finally:
        if txt_fh:
            txt_fh.close()

    print(json.dumps({"rows": rows, "by_domain": by_domain}, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean local gitignored corpus drops into training-ready JSONL buckets.

This is an orchestration layer for the existing corpus tools. It keeps raw
downloads untouched, extracts every available source into UTF-8
``{"text": ...}`` JSONL, then runs source-bucket quality filtering and global
deduplication.

The script is intentionally resume-friendly:
  * missing sources are skipped;
  * incomplete/corrupt parquet shards are skipped and reported;
  * existing non-empty outputs are reused unless ``--force`` is passed.

Example:
  python3 scripts/clean_do_not_git_corpora.py
  python3 scripts/clean_do_not_git_corpora.py --dry-run
  python3 scripts/clean_do_not_git_corpora.py --force --skip-chinese-web
"""

from __future__ import annotations

import argparse
import codecs
import glob
import json
import os
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "DO NOT GIT IT"
DEFAULT_OUT = DEFAULT_ROOT / "clean_stage"


@dataclass(frozen=True)
class SourceSpec:
    name: str
    group: str
    relpath: str
    kind: str
    fields: tuple[str, ...] = ("text",)
    min_chars: int = 1
    required: bool = False


@dataclass(frozen=True)
class CleanSpec:
    group: str
    script: str = ""
    min_script_ratio: float = 0.5
    min_chars: int = 40
    thresh: int = 3


SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        "mn_traditional_1000",
        "mn_traditional",
        "1000 traditional_mongolian_corpus\uff08DO NOT GIT IT\uff09",
        "prepare_mongolian",
        min_chars=20,
    ),
    SourceSpec(
        "mn_trad_cmli",
        "mn_traditional",
        "MN_TRAD_CMLI(DO NOT GIT IT)/Mongolian-pretrain-dataset.txt",
        "txt",
        min_chars=20,
    ),
    SourceSpec(
        "mn_gov",
        "mn_traditional",
        "gov_crawl/mgl_gov.jsonl",
        "jsonl",
        fields=("text",),
        min_chars=40,
    ),
    SourceSpec(
        "mn_holvoo",
        "mn_traditional",
        "holvoo_crawl/*.jsonl",
        "jsonl",
        fields=("text", "content", "body"),
        min_chars=40,
    ),
    SourceSpec(
        "mn_synth_traditional",
        "mn_traditional",
        "MONGOLIAN_SYNTH(DO NOT GIT IT)/seed_mn.txt",
        "txt",
        min_chars=20,
    ),
    SourceSpec(
        "mn_mc2",
        "mn_traditional",
        "MC2_MONGOLIAN(DO NOT GIT IT)/*.jsonl",
        "jsonl",
        fields=("text", "content"),
        min_chars=80,
    ),
    SourceSpec(
        "mn_cyr_alpaca",
        "mn_cyrillic",
        "MN_CYR_ALPACA(DO NOT GIT IT)/*.parquet",
        "parquet",
        fields=("instruction", "input", "output"),
        min_chars=40,
    ),
    SourceSpec(
        "mn_cyr_bayartsogt",
        "mn_cyrillic",
        "MN_CYR_BAYARTSOGT(DO NOT GIT IT)/*.parquet",
        "parquet",
        fields=("text", "content", "instruction", "input", "output"),
        min_chars=80,
    ),
    SourceSpec(
        "mn_cyr_fineweb2",
        "mn_cyrillic",
        "MN_CYR_FINEWEB2(DO NOT GIT IT)/*.parquet",
        "parquet",
        fields=("text",),
        min_chars=80,
    ),
    SourceSpec(
        "mn_cyr_billyyy",
        "mn_cyrillic",
        "MN_CYR_BILLYYY(DO NOT GIT IT)/*.parquet",
        "parquet",
        fields=("text",),
        min_chars=80,
    ),
    SourceSpec(
        "zh_bundle",
        "zh_general",
        "CHINESE(DO NOT GIT IT)",
        "prepare_chinese",
        min_chars=80,
    ),
    SourceSpec(
        "zh_synth_seed",
        "zh_general",
        "MONGOLIAN_SYNTH(DO NOT GIT IT)/seed_zh.jsonl",
        "jsonl",
        fields=("text",),
        min_chars=80,
    ),
    SourceSpec(
        "zh_math_gsm8k",
        "zh_math",
        "CHINESE_MATH(DO NOT GIT IT)/gsm8k_zh/*.parquet",
        "parquet",
        fields=("question_zh-cn", "answer"),
        min_chars=40,
    ),
    SourceSpec(
        "wiki_zh",
        "wiki_zh",
        "WIKIPEDIA(DO NOT GIT IT)/20231101.zh/*.parquet",
        "parquet",
        fields=("title", "text"),
        min_chars=120,
    ),
    SourceSpec(
        "wiki_en",
        "wiki_en",
        "WIKIPEDIA(DO NOT GIT IT)/20231101.en/*.parquet",
        "parquet",
        fields=("title", "text"),
        min_chars=120,
    ),
    SourceSpec(
        "wiki_ja",
        "wiki_ja",
        "WIKIPEDIA(DO NOT GIT IT)/20231101.ja/*.parquet",
        "parquet",
        fields=("title", "text"),
        min_chars=120,
    ),
    SourceSpec(
        "wiki_mn",
        "mn_cyrillic",
        "WIKIPEDIA(DO NOT GIT IT)/20231101.mn/*.parquet",
        "parquet",
        fields=("title", "text"),
        min_chars=80,
    ),
    SourceSpec(
        "finemath",
        "math_en",
        "FINEMATH(DO NOT GIT IT)/finemath-4plus/*.parquet",
        "parquet",
        fields=("text",),
        min_chars=160,
    ),
    SourceSpec(
        "openwebmath",
        "math_en",
        "OPENWEBMATH(DO NOT GIT IT)/data/*.parquet",
        "parquet",
        fields=("text",),
        min_chars=160,
    ),
    SourceSpec(
        "metamath",
        "math_en",
        "MATH_METAMATH(DO NOT GIT IT)/MetaMathQA-395K.json",
        "json_array",
        fields=("query", "response"),
        min_chars=80,
    ),
    SourceSpec(
        "gsm8k_en",
        "math_en",
        "OPENAI_GSM8K(DO NOT GIT IT)/*.parquet",
        "parquet",
        fields=("question", "answer"),
        min_chars=40,
    ),
    SourceSpec(
        "cosmopedia",
        "cosmopedia",
        "COSMOPEDIA(DO NOT GIT IT)/*.parquet",
        "parquet",
        fields=("text",),
        min_chars=120,
    ),
    SourceSpec(
        "fineweb_edu",
        "english_edu_web",
        "FINEWEB_EDU(DO NOT GIT IT)/*.parquet",
        "parquet",
        fields=("text",),
        min_chars=120,
    ),
    SourceSpec(
        "open_thoughts",
        "reasoning",
        "REASON_OPENTHOUGHTS(DO NOT GIT IT)/*.parquet",
        "parquet",
        fields=("text", "conversations", "messages", "problem", "solution"),
        min_chars=120,
    ),
    SourceSpec(
        "openr1_math",
        "reasoning",
        "REASON_OPENR1_MATH(DO NOT GIT IT)/*.parquet",
        "parquet",
        fields=("problem", "solution", "answer", "generations", "messages"),
        min_chars=120,
    ),
    SourceSpec(
        "zh_r1_reasoning",
        "reasoning",
        "REASON_ZH_R1(DO NOT GIT IT)/distill_r1_110k.jsonl",
        "jsonl",
        fields=("input", "reasoning_content", "content"),
        min_chars=120,
    ),
    SourceSpec(
        "python_edu",
        "code_python",
        "CODE_PYTHON_EDU(DO NOT GIT IT)/*.parquet",
        "parquet",
        fields=("text", "content", "code"),
        min_chars=80,
    ),
    SourceSpec(
        "flores_plus",
        "translation_seed",
        "FLORES_PLUS(DO NOT GIT IT)/*.jsonl",
        "jsonl",
        fields=("text",),
        min_chars=20,
    ),
)


CLEAN_SPECS: dict[str, CleanSpec] = {
    "mn_traditional": CleanSpec(
        "mn_traditional", script="mongolian", min_script_ratio=0.45, min_chars=30
    ),
    "mn_cyrillic": CleanSpec(
        "mn_cyrillic", script="cyrillic", min_script_ratio=0.45, min_chars=60
    ),
    "zh_general": CleanSpec("zh_general", script="zh", min_script_ratio=0.35, min_chars=80),
    "zh_math": CleanSpec("zh_math", script="zh", min_script_ratio=0.25, min_chars=40),
    "wiki_zh": CleanSpec("wiki_zh", script="zh", min_script_ratio=0.35, min_chars=120),
    "wiki_en": CleanSpec("wiki_en", min_chars=120),
    "wiki_ja": CleanSpec("wiki_ja", min_chars=80),
    "math_en": CleanSpec("math_en", min_chars=160),
    "cosmopedia": CleanSpec("cosmopedia", min_chars=120),
    "english_edu_web": CleanSpec("english_edu_web", min_chars=120),
    "reasoning": CleanSpec("reasoning", min_chars=120),
    "code_python": CleanSpec("code_python", min_chars=80),
    "translation_seed": CleanSpec("translation_seed", min_chars=20),
}


CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
INLINE_WS_RE = re.compile(r"[^\S\n]+")
MULTI_NL_RE = re.compile(r"\n{3,}")
NULLISH = {"", "none", "null", "nan", "n/a"}


def clean_text(text: object) -> str:
    if text is None:
        return ""
    if isinstance(text, (list, tuple)):
        text = "\n".join(clean_text(x) for x in text)
    elif isinstance(text, dict):
        preferred = []
        for key in ("role", "content", "from", "value", "text"):
            if key in text:
                preferred.append(str(text[key]))
        text = "\n".join(preferred) if preferred else json.dumps(text, ensure_ascii=False)
    else:
        text = str(text)
    text = text.replace("\ufeff", "")
    text = unicodedata.normalize("NFC", text)
    text = CONTROL_RE.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [INLINE_WS_RE.sub(" ", line).strip() for line in text.split("\n")]
    text = MULTI_NL_RE.sub("\n\n", "\n".join(line for line in lines if line))
    text = text.strip()
    if text.lower() in NULLISH:
        return ""
    return text


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_done(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def run_cmd(cmd: list[str], *, dry_run: bool, log_path: Path | None = None) -> int:
    print("+ " + " ".join(cmd))
    if dry_run:
        return 0
    if log_path is None:
        return subprocess.run(cmd, cwd=REPO_ROOT).returncode
    ensure_dir(log_path.parent)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        log.write(proc.stdout)
        if proc.stdout:
            print(proc.stdout.rstrip())
        return proc.returncode


def resolve_paths(root: Path, relpath: str) -> list[Path]:
    pattern = root / relpath
    if any(ch in str(pattern) for ch in "*?["):
        return sorted(Path(p) for p in glob.glob(str(pattern)))
    if pattern.is_dir():
        return [pattern]
    if pattern.exists():
        return [pattern]
    return []


def files_for_input(path: Path, suffixes: tuple[str, ...]) -> list[Path]:
    if path.is_file():
        return [path] if path.name.lower().endswith(suffixes) else []
    out: list[Path] = []
    for suffix in suffixes:
        out.extend(path.rglob(f"*{suffix}"))
    return sorted(out)


def write_jsonl(path: Path, rows: Iterable[str], *, min_chars: int, limit: int) -> int:
    ensure_dir(path.parent)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for text in rows:
            text = clean_text(text)
            if len(text) < min_chars:
                continue
            fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            count += 1
            if limit and count >= limit:
                break
    return count


def iter_jsonl(paths: list[Path], fields: tuple[str, ...]) -> Iterable[str]:
    def emit(obj: object) -> Iterable[str]:
        if isinstance(obj, str):
            yield obj
            return
        if not isinstance(obj, dict):
            return
        parts = [clean_text(obj.get(field)) for field in fields if obj.get(field)]
        if parts:
            yield "\n".join(parts)

    for path in paths:
        for file in files_for_input(path, (".jsonl", ".json", ".ndjson")):
            with file.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, list):
                        for rec in obj:
                            yield from emit(rec)
                    else:
                        yield from emit(obj)


def iter_json_array(paths: list[Path], fields: tuple[str, ...]) -> Iterable[str]:
    for path in paths:
        for file in files_for_input(path, (".json",)):
            with file.open("r", encoding="utf-8", errors="replace") as fh:
                try:
                    obj = json.load(fh)
                except json.JSONDecodeError:
                    continue
            records = obj if isinstance(obj, list) else [obj]
            for rec in records:
                if not isinstance(rec, dict):
                    continue
                parts = [clean_text(rec.get(field)) for field in fields if rec.get(field)]
                if parts:
                    yield "\n".join(parts)


def detect_text_encoding(file: Path) -> str:
    with file.open("rb") as fh:
        sample = fh.read(8192)
    if sample.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    if sample.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        codecs.getincrementaldecoder("utf-8")().decode(sample, final=False)
        return "utf-8"
    except UnicodeDecodeError:
        pass
    if sample:
        even_nulls = sample[0::2].count(0)
        odd_nulls = sample[1::2].count(0)
        if odd_nulls > len(sample) // 4 and odd_nulls > even_nulls * 2:
            return "utf-16-le"
        if even_nulls > len(sample) // 4 and even_nulls > odd_nulls * 2:
            return "utf-16-be"
    return "gb18030"


def iter_txt(paths: list[Path]) -> Iterable[str]:
    for path in paths:
        for file in files_for_input(path, (".txt",)):
            enc = detect_text_encoding(file)
            with file.open("r", encoding=enc, errors="replace") as fh:
                for line in fh:
                    chunk = line.strip()
                    if chunk:
                        yield chunk


def iter_parquet(
    paths: list[Path],
    fields: tuple[str, ...],
    *,
    report: dict[str, object],
) -> Iterable[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        report.setdefault("errors", []).append(f"pyarrow missing: {exc}")
        return

    parquet_files: list[Path] = []
    for path in paths:
        parquet_files.extend(files_for_input(path, (".parquet",)))
    for file in sorted(set(parquet_files)):
        try:
            pf = pq.ParquetFile(str(file))
        except Exception as exc:  # incomplete downloads land here
            report.setdefault("bad_parquet", []).append({"path": str(file), "error": str(exc)})
            continue
        available = set(pf.schema_arrow.names)
        use_fields = tuple(field for field in fields if field in available)
        if not use_fields:
            report.setdefault("missing_columns", []).append(
                {"path": str(file), "available": sorted(available), "wanted": list(fields)}
            )
            continue
        try:
            for batch in pf.iter_batches(batch_size=1000, columns=list(use_fields)):
                for row in batch.to_pylist():
                    parts = [clean_text(row.get(field)) for field in use_fields if row.get(field)]
                    if parts:
                        yield "\n".join(parts)
        except Exception as exc:
            report.setdefault("bad_parquet", []).append({"path": str(file), "error": str(exc)})


def extract_source(
    source: SourceSpec,
    *,
    root: Path,
    raw_dir: Path,
    reports: dict[str, object],
    force: bool,
    dry_run: bool,
    limit: int,
) -> Path | None:
    paths = resolve_paths(root, source.relpath)
    if not paths:
        reports["skipped"].append({"source": source.name, "reason": "missing"})
        return None
    out = raw_dir / f"{source.name}.raw.jsonl"
    if is_done(out) and not force:
        reports["reused"].append(str(out))
        return out
    if dry_run:
        print(f"[dry-run] extract {source.name} -> {out}")
        return out

    if source.kind in {"prepare_mongolian", "prepare_chinese"}:
        module_source = "mongolian" if source.kind == "prepare_mongolian" else "chinese"
        cmd = [
            sys.executable,
            "-m",
            "Tokenizer.tools.prepare_corpus",
            "--source",
            module_source,
            "--input",
            str(paths[0]),
            "--output",
            str(out),
        ]
        rc = run_cmd(cmd, dry_run=False)
        if rc != 0:
            reports["failed"].append({"source": source.name, "rc": rc})
            return None
        return out if is_done(out) else None

    source_report: dict[str, object] = {}
    if source.kind == "jsonl":
        rows = iter_jsonl(paths, source.fields)
    elif source.kind == "json_array":
        rows = iter_json_array(paths, source.fields)
    elif source.kind == "txt":
        rows = iter_txt(paths)
    elif source.kind == "parquet":
        rows = iter_parquet(paths, source.fields, report=source_report)
    else:
        reports["failed"].append({"source": source.name, "reason": f"unknown kind {source.kind}"})
        return None
    count = write_jsonl(out, rows, min_chars=source.min_chars, limit=limit)
    source_report["written"] = count
    reports["sources"][source.name] = source_report
    if count == 0:
        reports["skipped"].append({"source": source.name, "reason": "zero extracted"})
        return None
    return out


def clean_group(
    spec: CleanSpec,
    raw_files: list[Path],
    *,
    clean_dir: Path,
    report_dir: Path,
    force: bool,
    dry_run: bool,
) -> Path | None:
    if not raw_files:
        return None
    out = clean_dir / f"{spec.group}.clean.jsonl"
    if is_done(out) and not force:
        print(f"[skip] {out} exists")
        return out
    cmd = [
        sys.executable,
        "Tokenizer/tools/corpus_clean.py",
        "--in",
        *(str(p) for p in raw_files),
        "--out",
        str(out),
        "--min-chars",
        str(spec.min_chars),
        "--thresh",
        str(spec.thresh),
    ]
    if spec.script:
        cmd.extend(["--script", spec.script, "--min-script-ratio", str(spec.min_script_ratio)])
    rc = run_cmd(cmd, dry_run=dry_run, log_path=report_dir / f"{spec.group}.report.txt")
    return out if rc == 0 and (dry_run or is_done(out)) else None


def clean_chinese_web(
    *,
    root: Path,
    out_dir: Path,
    workers: int,
    force: bool,
    dry_run: bool,
    limit_units: int,
) -> Path | None:
    src = root / "CHINESE_WEB(DO NOT GIT IT)"
    if not src.exists():
        return None
    dst = out_dir / "zh_web_shards"
    state = dst / "_state.json"
    if state.exists() and not force:
        print(f"[skip] {dst} exists")
        return dst
    cmd = [
        sys.executable,
        "-m",
        "Tokenizer.tools.clean_chinese_web",
        "--in",
        str(src),
        "--out-dir",
        str(dst),
        "--workers",
        str(workers),
        "--batch-size",
        "2000",
        "--min-chars",
        "200",
        "--min-chinese-ratio",
        "0.55",
    ]
    if limit_units:
        cmd.extend(["--limit-units", str(limit_units)])
    rc = run_cmd(cmd, dry_run=dry_run)
    return dst if rc == 0 else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0, help="max docs per extracted source")
    ap.add_argument("--limit-chinese-web-units", type=int, default=0)
    ap.add_argument("--skip-chinese-web", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    raw_dir = args.out_dir / "raw"
    clean_dir = args.out_dir / "clean"
    report_dir = args.out_dir / "reports"
    for path in (raw_dir, clean_dir, report_dir):
        ensure_dir(path)

    reports: dict[str, object] = {
        "root": str(args.root),
        "out_dir": str(args.out_dir),
        "sources": {},
        "skipped": [],
        "failed": [],
        "reused": [],
        "clean": {},
    }

    group_inputs: dict[str, list[Path]] = {}
    for source in SOURCES:
        out = extract_source(
            source,
            root=args.root,
            raw_dir=raw_dir,
            reports=reports,
            force=args.force,
            dry_run=args.dry_run,
            limit=args.limit,
        )
        if out is not None:
            group_inputs.setdefault(source.group, []).append(out)

    clean_outputs: dict[str, str] = {}
    for group, raw_files in sorted(group_inputs.items()):
        spec = CLEAN_SPECS.get(group, CleanSpec(group))
        out = clean_group(
            spec,
            raw_files,
            clean_dir=clean_dir,
            report_dir=report_dir,
            force=args.force,
            dry_run=args.dry_run,
        )
        if out is not None:
            clean_outputs[group] = str(out)

    if not args.skip_chinese_web:
        web = clean_chinese_web(
            root=args.root,
            out_dir=args.out_dir,
            workers=args.workers,
            force=args.force,
            dry_run=args.dry_run,
            limit_units=args.limit_chinese_web_units,
        )
        if web is not None:
            clean_outputs["zh_web_shards"] = str(web)

    reports["clean"] = clean_outputs
    manifest = args.out_dir / "clean_manifest.json"
    if not args.dry_run:
        manifest.write_text(
            json.dumps(reports, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"clean outputs: {len(clean_outputs)}")
    for group, path in sorted(clean_outputs.items()):
        print(f"  {group}: {path}")
    print(f"manifest: {manifest}")
    return 0 if not reports["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

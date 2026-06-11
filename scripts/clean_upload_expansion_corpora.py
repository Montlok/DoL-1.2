#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean downloaded expansion corpora and upload sharded JSONL to Hugging Face.

The script is intentionally local-data oriented: it reads corpora under
``DO NOT GIT IT/corpus_expansion`` and writes all derived outputs under
``DO NOT GIT IT/corpus_expansion_clean``.  It keeps upstream provenance in every
record, shards data into small gzip files, and uploads to language/domain repos.
"""

from __future__ import annotations

import argparse
import codecs
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import pyarrow.parquet as pq
from lxml import etree, html


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "DO NOT GIT IT" / "corpus_expansion"
DEFAULT_OUT_ROOT = REPO_ROOT / "DO NOT GIT IT" / "corpus_expansion_clean"

RUN_ID = "expansion_2026_06_04"

SCRIPT_RANGES = {
    "zh": ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)),
    "ja": (
        (0x3040, 0x309F),  # hiragana
        (0x30A0, 0x30FF),  # katakana
        (0x31F0, 0x31FF),  # katakana phonetic extensions
        (0x3400, 0x4DBF),
        (0x4E00, 0x9FFF),
        (0xF900, 0xFAFF),
    ),
    "cyrillic": (
        (0x0400, 0x04FF),
        (0x0500, 0x052F),
        (0x2DE0, 0x2DFF),
        (0xA640, 0xA69F),
    ),
}

REPO_IDS = {
    "zh": "Montlok/chinese_corpus",
    "ja": "Montlok/japanese_corpus",
    "mn": "Montlok/mongolian_culture_corpus",
    "code": "Montlok/code_corpus",
}


@dataclass(frozen=True)
class BucketConfig:
    repo_key: str
    language: str
    domain: str
    script: str = ""
    min_chars: int = 80
    min_script_ratio: float = 0.35


BUCKETS: dict[str, BucketConfig] = {
    "zh_classical": BucketConfig(
        "zh", "zh", "classical_literature", "zh", min_chars=20, min_script_ratio=0.30
    ),
    "zh_legal": BucketConfig(
        "zh", "zh", "law_policy", "zh", min_chars=80, min_script_ratio=0.35
    ),
    "zh_philosophy_dialectics": BucketConfig(
        "zh", "zh", "philosophy_dialectics", "zh", min_chars=80, min_script_ratio=0.35
    ),
    "ja_literature": BucketConfig(
        "ja", "ja", "literature", "ja", min_chars=80, min_script_ratio=0.25
    ),
    "ja_legal": BucketConfig(
        "ja", "ja", "law_policy", "ja", min_chars=120, min_script_ratio=0.25
    ),
    "mn_cyrillic_culture": BucketConfig(
        "mn", "mn-Cyrl", "culture_history", "cyrillic", min_chars=40, min_script_ratio=0.35
    ),
    "mn_cyrillic_legal": BucketConfig(
        "mn", "mn-Cyrl", "law_policy", "cyrillic", min_chars=120, min_script_ratio=0.35
    ),
    "code": BucketConfig("code", "code", "programming", "", min_chars=80),
}


@dataclass(frozen=True)
class Record:
    text: str
    source: str
    source_path: str
    title: str = ""
    url: str = ""
    metadata: dict[str, Any] | None = None


_WS_RE = re.compile(r"[ \t\r\f\v]+")
_MANY_NEWLINES_RE = re.compile(r"\n{3,}")
_XML_DECL_RE = re.compile(r"^\s*<\?xml[^>]*\?>", flags=re.I)
_HTML_META_CHARSET_RE = re.compile(rb"charset=[\"']?([A-Za-z0-9_\-]+)", flags=re.I)
_AUDIOVISUAL_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".svg",
    ".css",
    ".js",
    ".ico",
    ".pdf",
    ".epub",
}

DROP_SUBSTRINGS = (
    "Object not found!",
    "The requested URL was not found on this server",
    "Just a moment...",
    "Access denied",
    "Enable JavaScript and cookies",
)

LEGALINFO_UI_LINES = {
    "Бүртгүүлэх",
    "Нэвтрэх",
    "Нүүр",
    "Шүүлтүүр",
    "Бүгд",
    "Бөөнөөр нь хаах",
    "Сонсох",
    "Pdf",
    "PDF",
    "Word",
    "Хэвлэх",
    "Хуваалцах",
    "Тусламж",
}


def log(msg: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def decode_bytes(data: bytes, fallback: str = "utf-8") -> str:
    if data.startswith(codecs.BOM_UTF8):
        return data.decode("utf-8-sig", errors="replace")
    m = _HTML_META_CHARSET_RE.search(data[:4096])
    candidates: list[str] = []
    if m:
        candidates.append(m.group(1).decode("ascii", errors="ignore"))
    candidates.extend([fallback, "utf-8", "gb18030", "gbk", "cp932", "shift_jis"])
    seen: set[str] = set()
    for enc in candidates:
        if not enc:
            continue
        key = enc.lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
        except LookupError:
            continue
    return data.decode(fallback, errors="replace")


def read_text(path: Path, fallback: str = "utf-8") -> str:
    return decode_bytes(path.read_bytes(), fallback=fallback)


def normalize_text(text: str, *, keep_newlines: bool = True) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u3000", " ").replace("\u00a0", " ")
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or not unicodedata.category(ch).startswith("C"))
    lines = []
    for line in text.splitlines():
        line = _WS_RE.sub(" ", line).strip()
        if not line:
            lines.append("")
            continue
        lines.append(line)
    text = "\n".join(lines) if keep_newlines else " ".join(line for line in lines if line)
    text = _MANY_NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def markdown_to_text(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"^\s{0,3}[-*+]\s+", "", text, flags=re.M)
    text = re.sub(r"^\s{0,3}>\s?", "", text, flags=re.M)
    return normalize_text(text)


def script_ratio(text: str, script: str) -> float:
    ranges = SCRIPT_RANGES[script]
    chars = [ch for ch in text if not ch.isspace() and not ch.isdigit()]
    if not chars:
        return 0.0
    hits = sum(any(lo <= ord(ch) <= hi for lo, hi in ranges) for ch in chars)
    return hits / len(chars)


def repeated_line_fraction(text: str) -> float:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 4:
        return 0.0
    seen: set[str] = set()
    dup = 0
    for line in lines:
        if line in seen:
            dup += 1
        seen.add(line)
    return dup / len(lines)


def quality_reason(text: str, cfg: BucketConfig) -> str:
    if len(text) < cfg.min_chars:
        return "too_short"
    if "\ufffd" in text and text.count("\ufffd") / max(len(text), 1) > 0.01:
        return "decode_replacement"
    if any(s in text for s in DROP_SUBSTRINGS):
        return "error_page"
    if cfg.script and script_ratio(text, cfg.script) < cfg.min_script_ratio:
        return "script_impurity"
    if repeated_line_fraction(text) > 0.55:
        return "repeated_lines"
    if cfg.repo_key != "code":
        symbols = sum(text.count(c) for c in "#<>|{}[]=_~^$@")
        if symbols / max(len(text), 1) > 0.12:
            return "symbol_spam"
    return ""


def stable_id(source: str, source_path: str, text: str) -> str:
    h = hashlib.sha1()
    h.update(source.encode("utf-8"))
    h.update(b"\0")
    h.update(source_path.encode("utf-8"))
    h.update(b"\0")
    h.update(normalize_text(text, keep_newlines=False).encode("utf-8"))
    return h.hexdigest()


def exact_text_key(text: str) -> str:
    return hashlib.sha1(normalize_text(text, keep_newlines=False).encode("utf-8")).hexdigest()


def clean_html_text(text: str) -> html.HtmlElement:
    text = _XML_DECL_RE.sub("", text)
    return html.fromstring(text)


def drop_html_noise(doc: html.HtmlElement) -> None:
    for node in doc.xpath("//script|//style|//noscript|//header|//footer|//nav"):
        parent = node.getparent()
        if parent is not None:
            parent.remove(node)


def element_text(el: html.HtmlElement) -> str:
    return normalize_text("\n".join(part.strip() for part in el.itertext() if part.strip()))


def html_file_text(path: Path, *, fallback: str = "utf-8", selectors: tuple[str, ...] = ()) -> tuple[str, str]:
    raw = read_text(path, fallback=fallback)
    doc = clean_html_text(raw)
    drop_html_noise(doc)
    title = normalize_text(doc.xpath("string(//title)"), keep_newlines=False)
    for selector in selectors:
        nodes = doc.xpath(selector)
        if nodes:
            texts = [element_text(node) for node in nodes if element_text(node)]
            if texts:
                return max(texts, key=len), title
    return element_text(doc), title


def strip_aozora_plain(text: str) -> str:
    text = text.replace("\r\n", "\n")
    parts = text.split("-------------------------------------------------------")
    if len(parts) >= 3:
        text = "-------------------------------------------------------".join(parts[2:])
    if "底本:" in text:
        text = text.split("底本:", 1)[0]
    text = re.sub(r"《[^》]{1,60}》", "", text)
    text = text.replace("｜", "")
    text = re.sub(r"［＃[^］]+］", "", text)
    return normalize_text(text)


def iter_json_documents(obj: Any, path: Path, source: str, title_hint: str = "") -> Iterator[Record]:
    if isinstance(obj, list):
        for item in obj:
            yield from iter_json_documents(item, path, source, title_hint=title_hint)
        return
    if not isinstance(obj, dict):
        return

    title = str(
        obj.get("title")
        or obj.get("chapter")
        or obj.get("section")
        or obj.get("rhythmic")
        or title_hint
        or path.stem
    )
    author = str(obj.get("author") or obj.get("poet") or "")

    paragraphs = obj.get("paragraphs") or obj.get("paragraph") or obj.get("content") or obj.get("text")
    if isinstance(paragraphs, list):
        body = "\n".join(str(x) for x in paragraphs if isinstance(x, (str, int, float)))
    elif isinstance(paragraphs, str):
        body = paragraphs
    else:
        body = ""

    if body:
        head = "\n".join(x for x in (title, author) if x)
        text = f"{head}\n{body}" if head else body
        yield Record(text=text, source=source, source_path=rel(path), title=title, metadata={"author": author})
        return

    if {"instruction", "input", "output"} & set(obj):
        chunks = [str(obj.get("instruction") or "")]
        if obj.get("input"):
            chunks.append(str(obj.get("input")))
        if obj.get("output"):
            chunks.append(str(obj.get("output")))
        yield Record(text="\n".join(chunks), source=source, source_path=rel(path), title=title)
        return

    for value in obj.values():
        if isinstance(value, (dict, list)):
            yield from iter_json_documents(value, path, source, title_hint=title)


def load_json(path: Path) -> Any:
    return json.loads(read_text(path))


def iter_chinese_poetry(root: Path) -> Iterator[Record]:
    base = root / "zh_classical" / "chinese-poetry"
    for path in sorted(base.rglob("*.json")):
        if any(part.startswith(".") for part in path.parts):
            continue
        try:
            yield from iter_json_documents(load_json(path), path, "chinese-poetry")
        except Exception as exc:
            log(f"WARN chinese-poetry parse failed {rel(path)}: {exc}")


def iter_classical_modern(root: Path) -> Iterator[Record]:
    base = root / "zh_classical" / "Classical-Modern"
    for path in sorted((base / "古文原文").rglob("text.txt")):
        text = read_text(path)
        title = " / ".join(path.relative_to(base / "古文原文").parts[:-1])
        yield Record(text=f"{title}\n{text}", source="Classical-Modern/original", source_path=rel(path), title=title)

    parallel_root = base / "双语数据"
    for path in sorted(parallel_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() in _AUDIOVISUAL_EXTS:
            continue
        if path.suffix.lower() == ".json":
            try:
                yield from iter_json_documents(load_json(path), path, "Classical-Modern/parallel")
            except Exception as exc:
                log(f"WARN Classical-Modern JSON parse failed {rel(path)}: {exc}")
            continue
        if path.suffix.lower() in {".txt", ".tsv", ".csv"}:
            text = read_text(path)
            title = " / ".join(path.relative_to(parallel_root).parts[:-1]) or path.stem
            yield Record(text=f"{title}\n{text}", source="Classical-Modern/parallel", source_path=rel(path), title=title)


def iter_lawrefbook(root: Path) -> Iterator[Record]:
    base = root / "zh_legal" / "LawRefBook-Laws"
    for path in sorted(base.rglob("*.md")):
        if path.name.startswith("_"):
            continue
        text = markdown_to_text(read_text(path))
        category = path.parent.name
        title = path.stem
        yield Record(
            text=f"{title}\n{category}\n{text}",
            source="LawRefBook-Laws",
            source_path=rel(path),
            title=title,
            metadata={"category": category},
        )


def iter_kuugo_law(root: Path) -> Iterator[Record]:
    base = root / "zh_legal" / "Kuugo_Chinese_Law"
    for path in sorted(base.rglob("*.txt")):
        text = read_text(path)
        yield Record(text=f"{path.stem}\n{text}", source="Kuugo/Chinese-Law", source_path=rel(path), title=path.stem)


def iter_dusker_law(root: Path) -> Iterator[Record]:
    base = root / "zh_legal" / "Dusker_chinese-laws-pretrain"
    for path in sorted(base.rglob("*.json")):
        try:
            yield from iter_json_documents(load_json(path), path, "Dusker/chinese-laws-pretrain", title_hint=path.stem)
        except Exception as exc:
            log(f"WARN Dusker law parse failed {rel(path)}: {exc}")


def parquet_rows(path: Path, columns: list[str] | None = None) -> Iterator[dict[str, Any]]:
    pf = pq.ParquetFile(path)
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=columns)
        for rec in table.to_pylist():
            yield rec


def iter_twang_law(root: Path) -> Iterator[Record]:
    base = root / "zh_legal" / "twang2218_chinese-law-and-regulations"
    for path in sorted((base / "data").glob("*.parquet")):
        for rec in parquet_rows(path):
            title = str(rec.get("title") or "")
            content = str(rec.get("content") or "")
            office = str(rec.get("office") or "")
            law_type = str(rec.get("type") or "")
            text = "\n".join(x for x in (title, office, law_type, content) if x)
            meta = {k: str(v) for k, v in rec.items() if k != "content" and v is not None}
            yield Record(text=text, source="twang2218/chinese-law-and-regulations", source_path=rel(path), title=title, metadata=meta)


def iter_zh_legal(root: Path) -> Iterator[Record]:
    yield from iter_lawrefbook(root)
    yield from iter_kuugo_law(root)
    yield from iter_dusker_law(root)
    yield from iter_twang_law(root)


def iter_marxists_selected(root: Path) -> Iterator[Record]:
    base = root / "philosophy" / "marxists_selected"
    for path in sorted(base.glob("*.htm*")):
        try:
            text, title = html_file_text(path, fallback="gb18030")
        except Exception as exc:
            log(f"WARN marxists parse failed {rel(path)}: {exc}")
            continue
        yield Record(text=f"{title}\n{text}", source="marxists_selected", source_path=rel(path), title=title)


def iter_aozora(root: Path) -> Iterator[Record]:
    base = root / "ja" / "aozorabunko" / "cards"
    for path in sorted(base.glob("*/files/*")):
        yield from aozora_file_records(path)


def aozora_file_records(path: Path) -> Iterator[Record]:
    suffix = path.suffix.lower()
    if suffix in _AUDIOVISUAL_EXTS:
        return
    if suffix in {".html", ".htm"}:
        try:
            text, title = html_file_text(path, fallback="cp932", selectors=("//div[contains(@class, 'main_text')]",))
        except Exception as exc:
            log(f"WARN aozora HTML parse failed {rel(path)}: {exc}")
            return
        yield Record(text=f"{title}\n{text}", source="aozorabunko/html", source_path=rel(path), title=title)
        return
    if suffix == ".zip":
        try:
            with zipfile.ZipFile(path) as zf:
                for name in zf.namelist():
                    if not name.lower().endswith(".txt"):
                        continue
                    text = strip_aozora_plain(decode_bytes(zf.read(name), fallback="cp932"))
                    title = Path(name).stem
                    yield Record(
                        text=f"{title}\n{text}",
                        source="aozorabunko/ruby_zip",
                        source_path=f"{rel(path)}::{name}",
                        title=title,
                    )
        except Exception as exc:
            log(f"WARN aozora ZIP parse failed {rel(path)}: {exc}")
        return
    if suffix == ".txt":
        text = strip_aozora_plain(read_text(path, fallback="cp932"))
        yield Record(text=f"{path.stem}\n{text}", source="aozorabunko/txt", source_path=rel(path), title=path.stem)


def iter_egov_law(root: Path) -> Iterator[Record]:
    path = root / "ja" / "e_gov_law" / "all_laws_xml_bulkdownload.bin"
    if not path.exists():
        return
    parser = etree.XMLParser(recover=True, huge_tree=True)
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".xml"):
                continue
            try:
                doc = etree.fromstring(zf.read(name), parser=parser)
            except Exception as exc:
                log(f"WARN e-Gov XML parse failed {name}: {exc}")
                continue
            title = ""
            found = doc.xpath("//*[local-name()='LawTitle']/text()")
            if found:
                title = normalize_text(str(found[0]), keep_newlines=False)
            text = normalize_text("\n".join(t.strip() for t in doc.itertext() if t.strip()))
            yield Record(text=f"{title}\n{text}" if title else text, source="e-gov-law-xml", source_path=f"{rel(path)}::{name}", title=title)


def legalinfo_text(path: Path) -> tuple[str, str]:
    raw = read_text(path, fallback="utf-8")
    doc = clean_html_text(raw)
    drop_html_noise(doc)
    title_nodes = doc.xpath("//span[contains(@class, 'breadcrumb-space')]/text()")
    title = normalize_text(title_nodes[-1] if title_nodes else doc.xpath("string(//title)"), keep_newlines=False)

    labels = []
    for node in doc.xpath("//label[contains(@class, 'line-clamp-1')]"):
        text = normalize_text(" ".join(node.itertext()), keep_newlines=False)
        if text and text not in LEGALINFO_UI_LINES:
            labels.append(text)
    if labels:
        return "\n".join([title, *labels]), title

    forms = doc.xpath("//form[contains(@class, 'sanal-form')]")
    if forms:
        texts = [element_text(form) for form in forms]
        return max(texts, key=len), title
    return element_text(doc), title


def iter_legalinfo_mn(root: Path) -> Iterator[Record]:
    base = root / "mn_culture" / "legalinfo_mn"
    for path in sorted(base.glob("law_*.html")):
        try:
            text, title = legalinfo_text(path)
        except Exception as exc:
            log(f"WARN legalinfo parse failed {rel(path)}: {exc}")
            continue
        law_id = path.stem.removeprefix("law_").lstrip("0") or "0"
        url = f"https://legalinfo.mn/mn/detail?lawId={law_id}"
        yield Record(text=text, source="legalinfo.mn", source_path=rel(path), title=title, url=url, metadata={"law_id": law_id})


def iter_namka_history(root: Path) -> Iterator[Record]:
    base = root / "mn_culture" / "Namka_mongolian_history" / "final"
    for path in sorted(base.glob("*.json")):
        if path.name == "README_JSON.md":
            continue
        try:
            yield from iter_json_documents(load_json(path), path, "Namka/mongolian_history", title_hint=path.stem)
        except Exception as exc:
            log(f"WARN Namka history parse failed {rel(path)}: {exc}")


def iter_code_search_net(root: Path) -> Iterator[Record]:
    base = root / "code" / "code_search_net"
    columns = [
        "repository_name",
        "func_path_in_repository",
        "func_name",
        "whole_func_string",
        "language",
        "func_documentation_string",
        "split_name",
        "func_code_url",
    ]
    for path in sorted(base.glob("*/*.parquet")):
        for rec in parquet_rows(path, columns=columns):
            code = rec.get("whole_func_string") or ""
            doc = rec.get("func_documentation_string") or ""
            language = rec.get("language") or path.parent.name
            title = f"{language}:{rec.get('func_name') or ''}"
            text_parts = [
                f"Language: {language}",
                f"Repository: {rec.get('repository_name') or ''}",
                f"Path: {rec.get('func_path_in_repository') or ''}",
            ]
            if doc:
                text_parts.extend(["Documentation:", str(doc)])
            text_parts.extend(["Code:", str(code)])
            meta = {k: v for k, v in rec.items() if k not in {"whole_func_string", "func_documentation_string"}}
            yield Record(
                text="\n".join(text_parts),
                source="code_search_net",
                source_path=rel(path),
                title=title,
                url=str(rec.get("func_code_url") or ""),
                metadata=meta,
            )


SourceFn = Callable[[Path], Iterator[Record]]

SOURCES: tuple[tuple[str, str, SourceFn], ...] = (
    ("chinese-poetry", "zh_classical", iter_chinese_poetry),
    ("Classical-Modern", "zh_classical", iter_classical_modern),
    ("zh-legal", "zh_legal", iter_zh_legal),
    ("marxists_selected", "zh_philosophy_dialectics", iter_marxists_selected),
    ("aozorabunko", "ja_literature", iter_aozora),
    ("e-gov-law", "ja_legal", iter_egov_law),
    ("legalinfo.mn", "mn_cyrillic_legal", iter_legalinfo_mn),
    ("Namka/mongolian_history", "mn_cyrillic_culture", iter_namka_history),
    ("code_search_net", "code", iter_code_search_net),
)


class Cleaner:
    def __init__(self, out_root: Path, limit_per_source: int = 0):
        self.out_root = out_root
        self.clean_dir = out_root / "clean"
        self.report_dir = out_root / "reports"
        self.clean_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.limit_per_source = limit_per_source
        self.handles: dict[str, Any] = {}
        self.seen: dict[str, set[str]] = defaultdict(set)
        self.stats: dict[str, Counter] = defaultdict(Counter)
        self.source_stats: dict[str, Counter] = defaultdict(Counter)

    def close(self) -> None:
        for fh in self.handles.values():
            fh.close()

    def handle(self, bucket: str):
        if bucket not in self.handles:
            self.handles[bucket] = (self.clean_dir / f"{bucket}.clean.jsonl").open("w", encoding="utf-8")
        return self.handles[bucket]

    def add(self, bucket: str, rec: Record, source_name: str) -> None:
        cfg = BUCKETS[bucket]
        self.stats[bucket]["total"] += 1
        self.source_stats[source_name]["total"] += 1

        text = normalize_text(rec.text)
        reason = quality_reason(text, cfg)
        if reason:
            self.stats[bucket][f"drop_{reason}"] += 1
            self.source_stats[source_name][f"drop_{reason}"] += 1
            return

        key = exact_text_key(text)
        if key in self.seen[bucket]:
            self.stats[bucket]["drop_duplicate"] += 1
            self.source_stats[source_name]["drop_duplicate"] += 1
            return
        self.seen[bucket].add(key)

        out = {
            "id": stable_id(rec.source, rec.source_path, text),
            "text": text,
            "language": cfg.language,
            "domain": cfg.domain,
            "bucket": bucket,
            "source": rec.source,
            "source_path": rec.source_path,
        }
        if rec.title:
            out["title"] = rec.title
        if rec.url:
            out["url"] = rec.url
        if rec.metadata:
            out["metadata"] = rec.metadata
        self.handle(bucket).write(json.dumps(out, ensure_ascii=False) + "\n")
        self.stats[bucket]["kept"] += 1
        self.source_stats[source_name]["kept"] += 1

    def run_source(self, source_name: str, bucket: str, fn: SourceFn, source_root: Path) -> None:
        log(f"extract source={source_name} bucket={bucket}")
        count = 0
        for rec in fn(source_root):
            self.add(bucket, rec, source_name)
            count += 1
            if self.limit_per_source and count >= self.limit_per_source:
                break
            if count % 100000 == 0:
                kept = self.source_stats[source_name]["kept"]
                log(f"progress source={source_name} seen={count} kept={kept}")
        log(f"done source={source_name} seen={count} kept={self.source_stats[source_name]['kept']}")

    def write_reports(self) -> None:
        self.close()
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "buckets": {k: dict(v) for k, v in sorted(self.stats.items())},
            "sources": {k: dict(v) for k, v in sorted(self.source_stats.items())},
        }
        (self.report_dir / "clean_stats.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for bucket, stats in sorted(self.stats.items()):
            total = stats["total"] or 1
            lines = [f"bucket={bucket} total={stats['total']} kept={stats['kept']} kept_pct={stats['kept'] / total * 100:.2f}"]
            for k in sorted(stats):
                if k.startswith("drop_"):
                    lines.append(f"{k}={stats[k]} pct={stats[k] / total * 100:.2f}")
            (self.report_dir / f"{bucket}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def clean_all(source_root: Path, out_root: Path, limit_per_source: int = 0) -> None:
    cleaner = Cleaner(out_root, limit_per_source=limit_per_source)
    try:
        for source_name, bucket, fn in SOURCES:
            cleaner.run_source(source_name, bucket, fn, source_root)
    finally:
        cleaner.write_reports()


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "part"


def chunked(items: list[Any], size: int) -> Iterator[list[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def clean_part_add(
    bucket: str,
    rec: Record,
    source_name: str,
    fh: Any,
    seen: set[str],
    stats: Counter,
) -> None:
    cfg = BUCKETS[bucket]
    stats["total"] += 1

    text = normalize_text(rec.text)
    reason = quality_reason(text, cfg)
    if reason:
        stats[f"drop_{reason}"] += 1
        return

    key = exact_text_key(text)
    if key in seen:
        stats["drop_duplicate_part"] += 1
        return
    seen.add(key)

    out = {
        "id": stable_id(rec.source, rec.source_path, text),
        "text": text,
        "language": cfg.language,
        "domain": cfg.domain,
        "bucket": bucket,
        "source": rec.source,
        "source_path": rec.source_path,
    }
    if rec.title:
        out["title"] = rec.title
    if rec.url:
        out["url"] = rec.url
    if rec.metadata:
        out["metadata"] = rec.metadata
    fh.write(json.dumps(out, ensure_ascii=False) + "\n")
    stats["kept_premerge"] += 1


def iter_task_records(task: dict[str, Any], source_root: Path) -> Iterator[Record]:
    kind = task["kind"]
    paths = [Path(p) for p in task.get("paths", [])]

    if kind == "chinese_poetry_files":
        for path in paths:
            try:
                yield from iter_json_documents(load_json(path), path, "chinese-poetry")
            except Exception as exc:
                log(f"WARN chinese-poetry parse failed {rel(path)}: {exc}")
        return

    if kind == "classical_original_files":
        base = source_root / "zh_classical" / "Classical-Modern" / "古文原文"
        for path in paths:
            text = read_text(path)
            title = " / ".join(path.relative_to(base).parts[:-1])
            yield Record(text=f"{title}\n{text}", source="Classical-Modern/original", source_path=rel(path), title=title)
        return

    if kind == "classical_parallel_files":
        base = source_root / "zh_classical" / "Classical-Modern" / "双语数据"
        for path in paths:
            suffix = path.suffix.lower()
            if suffix == ".json":
                try:
                    yield from iter_json_documents(load_json(path), path, "Classical-Modern/parallel")
                except Exception as exc:
                    log(f"WARN Classical-Modern JSON parse failed {rel(path)}: {exc}")
            elif suffix in {".txt", ".tsv", ".csv"}:
                text = read_text(path)
                title = " / ".join(path.relative_to(base).parts[:-1]) or path.stem
                yield Record(text=f"{title}\n{text}", source="Classical-Modern/parallel", source_path=rel(path), title=title)
        return

    if kind == "lawref_md_files":
        for path in paths:
            text = markdown_to_text(read_text(path))
            category = path.parent.name
            title = path.stem
            yield Record(text=f"{title}\n{category}\n{text}", source="LawRefBook-Laws", source_path=rel(path), title=title, metadata={"category": category})
        return

    if kind == "kuugo_txt_files":
        for path in paths:
            text = read_text(path)
            yield Record(text=f"{path.stem}\n{text}", source="Kuugo/Chinese-Law", source_path=rel(path), title=path.stem)
        return

    if kind == "dusker_json_files":
        for path in paths:
            try:
                yield from iter_json_documents(load_json(path), path, "Dusker/chinese-laws-pretrain", title_hint=path.stem)
            except Exception as exc:
                log(f"WARN Dusker law parse failed {rel(path)}: {exc}")
        return

    if kind == "twang_parquet_files":
        for path in paths:
            for rec in parquet_rows(path):
                title = str(rec.get("title") or "")
                content = str(rec.get("content") or "")
                office = str(rec.get("office") or "")
                law_type = str(rec.get("type") or "")
                text = "\n".join(x for x in (title, office, law_type, content) if x)
                meta = {k: str(v) for k, v in rec.items() if k != "content" and v is not None}
                yield Record(text=text, source="twang2218/chinese-law-and-regulations", source_path=rel(path), title=title, metadata=meta)
        return

    if kind == "marxists_files":
        for path in paths:
            try:
                text, title = html_file_text(path, fallback="gb18030")
            except Exception as exc:
                log(f"WARN marxists parse failed {rel(path)}: {exc}")
                continue
            yield Record(text=f"{title}\n{text}", source="marxists_selected", source_path=rel(path), title=title)
        return

    if kind == "aozora_files":
        for path in paths:
            yield from aozora_file_records(path)
        return

    if kind == "egov_xml_names":
        zip_path = Path(task["zip_path"])
        parser = etree.XMLParser(recover=True, huge_tree=True)
        with zipfile.ZipFile(zip_path) as zf:
            for name in task["names"]:
                try:
                    doc = etree.fromstring(zf.read(name), parser=parser)
                except Exception as exc:
                    log(f"WARN e-Gov XML parse failed {name}: {exc}")
                    continue
                title = ""
                found = doc.xpath("//*[local-name()='LawTitle']/text()")
                if found:
                    title = normalize_text(str(found[0]), keep_newlines=False)
                text = normalize_text("\n".join(t.strip() for t in doc.itertext() if t.strip()))
                yield Record(text=f"{title}\n{text}" if title else text, source="e-gov-law-xml", source_path=f"{rel(zip_path)}::{name}", title=title)
        return

    if kind == "legalinfo_files":
        for path in paths:
            try:
                text, title = legalinfo_text(path)
            except Exception as exc:
                log(f"WARN legalinfo parse failed {rel(path)}: {exc}")
                continue
            law_id = path.stem.removeprefix("law_").lstrip("0") or "0"
            url = f"https://legalinfo.mn/mn/detail?lawId={law_id}"
            yield Record(text=text, source="legalinfo.mn", source_path=rel(path), title=title, url=url, metadata={"law_id": law_id})
        return

    if kind == "namka_json_files":
        for path in paths:
            try:
                yield from iter_json_documents(load_json(path), path, "Namka/mongolian_history", title_hint=path.stem)
            except Exception as exc:
                log(f"WARN Namka history parse failed {rel(path)}: {exc}")
        return

    if kind == "code_parquet_files":
        columns = [
            "repository_name",
            "func_path_in_repository",
            "func_name",
            "whole_func_string",
            "language",
            "func_documentation_string",
            "split_name",
            "func_code_url",
        ]
        for path in paths:
            for rec in parquet_rows(path, columns=columns):
                code = rec.get("whole_func_string") or ""
                doc = rec.get("func_documentation_string") or ""
                language = rec.get("language") or path.parent.name
                title = f"{language}:{rec.get('func_name') or ''}"
                text_parts = [
                    f"Language: {language}",
                    f"Repository: {rec.get('repository_name') or ''}",
                    f"Path: {rec.get('func_path_in_repository') or ''}",
                ]
                if doc:
                    text_parts.extend(["Documentation:", str(doc)])
                text_parts.extend(["Code:", str(code)])
                meta = {k: v for k, v in rec.items() if k not in {"whole_func_string", "func_documentation_string"}}
                yield Record(
                    text="\n".join(text_parts),
                    source="code_search_net",
                    source_path=rel(path),
                    title=title,
                    url=str(rec.get("func_code_url") or ""),
                    metadata=meta,
                )
        return

    raise ValueError(f"unknown task kind {kind!r}")


def build_parallel_tasks(source_root: Path, *, limit_per_source: int = 0) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []

    def add_file_tasks(kind: str, source: str, bucket: str, paths: list[Path], batch_size: int) -> None:
        if limit_per_source:
            paths = paths[:limit_per_source]
        for idx, batch in enumerate(chunked(paths, batch_size)):
            tasks.append(
                {
                    "id": f"{slugify(source)}_{kind}_{idx:05d}",
                    "kind": kind,
                    "source": source,
                    "bucket": bucket,
                    "paths": [str(p) for p in batch],
                }
            )

    cp = source_root / "zh_classical" / "chinese-poetry"
    add_file_tasks(
        "chinese_poetry_files",
        "chinese-poetry",
        "zh_classical",
        sorted(p for p in cp.rglob("*.json") if not any(part.startswith(".") for part in p.parts)),
        128,
    )

    cm = source_root / "zh_classical" / "Classical-Modern"
    add_file_tasks("classical_original_files", "Classical-Modern/original", "zh_classical", sorted((cm / "古文原文").rglob("text.txt")), 512)
    parallel_paths = [
        p
        for p in sorted((cm / "双语数据").rglob("*"))
        if p.is_file() and p.suffix.lower() in {".json", ".txt", ".tsv", ".csv"} and p.suffix.lower() not in _AUDIOVISUAL_EXTS
    ]
    add_file_tasks("classical_parallel_files", "Classical-Modern/parallel", "zh_classical", parallel_paths, 512)

    add_file_tasks(
        "lawref_md_files",
        "LawRefBook-Laws",
        "zh_legal",
        sorted(p for p in (source_root / "zh_legal" / "LawRefBook-Laws").rglob("*.md") if not p.name.startswith("_")),
        128,
    )
    add_file_tasks("kuugo_txt_files", "Kuugo/Chinese-Law", "zh_legal", sorted((source_root / "zh_legal" / "Kuugo_Chinese_Law").rglob("*.txt")), 128)
    add_file_tasks("dusker_json_files", "Dusker/chinese-laws-pretrain", "zh_legal", sorted((source_root / "zh_legal" / "Dusker_chinese-laws-pretrain").rglob("*.json")), 64)
    add_file_tasks(
        "twang_parquet_files",
        "twang2218/chinese-law-and-regulations",
        "zh_legal",
        sorted((source_root / "zh_legal" / "twang2218_chinese-law-and-regulations" / "data").glob("*.parquet")),
        1,
    )

    add_file_tasks("marxists_files", "marxists_selected", "zh_philosophy_dialectics", sorted((source_root / "philosophy" / "marxists_selected").glob("*.htm*")), 4)

    aozora_paths = [
        p
        for p in sorted((source_root / "ja" / "aozorabunko" / "cards").glob("*/files/*"))
        if p.is_file() and p.suffix.lower() not in _AUDIOVISUAL_EXTS
    ]
    add_file_tasks("aozora_files", "aozorabunko", "ja_literature", aozora_paths, 256)

    egov_zip = source_root / "ja" / "e_gov_law" / "all_laws_xml_bulkdownload.bin"
    if egov_zip.exists():
        with zipfile.ZipFile(egov_zip) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".xml")]
        if limit_per_source:
            names = names[:limit_per_source]
        for idx, batch in enumerate(chunked(names, 256)):
            tasks.append(
                {
                    "id": f"e-gov-law_xml_{idx:05d}",
                    "kind": "egov_xml_names",
                    "source": "e-gov-law",
                    "bucket": "ja_legal",
                    "zip_path": str(egov_zip),
                    "names": batch,
                }
            )

    add_file_tasks("legalinfo_files", "legalinfo.mn", "mn_cyrillic_legal", sorted((source_root / "mn_culture" / "legalinfo_mn").glob("law_*.html")), 40)
    add_file_tasks("namka_json_files", "Namka/mongolian_history", "mn_cyrillic_culture", sorted((source_root / "mn_culture" / "Namka_mongolian_history" / "final").glob("*.json")), 2)
    add_file_tasks("code_parquet_files", "code_search_net", "code", sorted((source_root / "code" / "code_search_net").glob("*/*.parquet")), 1)

    return tasks


def clean_task_worker(task: dict[str, Any], source_root_str: str, parts_root_str: str) -> dict[str, Any]:
    source_root = Path(source_root_str)
    parts_root = Path(parts_root_str)
    bucket = task["bucket"]
    source_name = task["source"]
    part_dir = parts_root / bucket
    part_dir.mkdir(parents=True, exist_ok=True)
    part_path = part_dir / f"{task['id']}.jsonl"
    stats: Counter = Counter()
    seen: set[str] = set()
    with part_path.open("w", encoding="utf-8") as fh:
        for rec in iter_task_records(task, source_root):
            clean_part_add(bucket, rec, source_name, fh, seen, stats)
    return {
        "id": task["id"],
        "source": source_name,
        "bucket": bucket,
        "part_path": str(part_path),
        "stats": dict(stats),
    }


def merge_clean_parts(out_root: Path, task_results: list[dict[str, Any]]) -> tuple[dict[str, Counter], dict[str, Counter]]:
    clean_dir = out_root / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    bucket_stats: dict[str, Counter] = defaultdict(Counter)
    source_stats: dict[str, Counter] = defaultdict(Counter)

    bucket_parts: dict[str, list[Path]] = defaultdict(list)
    part_sources: dict[Path, str] = {}
    for result in task_results:
        bucket = result["bucket"]
        source = result["source"]
        stats = Counter(result.get("stats", {}))
        bucket_stats[bucket].update(stats)
        source_stats[source].update(stats)
        path = Path(result["part_path"])
        if path.exists() and path.stat().st_size > 0:
            bucket_parts[bucket].append(path)
            part_sources[path] = source

    for bucket, parts in sorted(bucket_parts.items()):
        seen: set[str] = set()
        tmp = clean_dir / f"{bucket}.clean.jsonl.tmp"
        out = clean_dir / f"{bucket}.clean.jsonl"
        kept = 0
        global_dups = 0
        with tmp.open("w", encoding="utf-8") as out_fh:
            for part in sorted(parts):
                source = part_sources[part]
                with part.open("r", encoding="utf-8") as in_fh:
                    for line in in_fh:
                        rec = json.loads(line)
                        key = exact_text_key(rec.get("text", ""))
                        if key in seen:
                            global_dups += 1
                            source_stats[source]["drop_duplicate_global"] += 1
                            continue
                        seen.add(key)
                        out_fh.write(line)
                        kept += 1
        tmp.replace(out)
        bucket_stats[bucket]["drop_duplicate_global"] = global_dups
        bucket_stats[bucket]["kept"] = kept
        log(f"merged bucket={bucket} parts={len(parts)} kept={kept} global_dups={global_dups}")

    return bucket_stats, source_stats


def write_clean_reports(out_root: Path, bucket_stats: dict[str, Counter], source_stats: dict[str, Counter]) -> None:
    report_dir = out_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "buckets": {k: dict(v) for k, v in sorted(bucket_stats.items())},
        "sources": {k: dict(v) for k, v in sorted(source_stats.items())},
    }
    (report_dir / "clean_stats.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for bucket, stats in sorted(bucket_stats.items()):
        total = stats["total"] or 1
        lines = [f"bucket={bucket} total={stats['total']} kept={stats['kept']} kept_pct={stats['kept'] / total * 100:.2f}"]
        for k in sorted(stats):
            if k.startswith("drop_"):
                lines.append(f"{k}={stats[k]} pct={stats[k] / total * 100:.2f}")
        (report_dir / f"{bucket}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def clean_all_parallel(source_root: Path, out_root: Path, *, workers: int, limit_per_source: int = 0) -> None:
    clean_dir = out_root / "clean"
    parts_root = out_root / "parts"
    report_dir = out_root / "reports"
    for path in (clean_dir, parts_root, report_dir):
        if path.exists():
            shutil.rmtree(path)
    clean_dir.mkdir(parents=True, exist_ok=True)
    parts_root.mkdir(parents=True, exist_ok=True)

    tasks = build_parallel_tasks(source_root, limit_per_source=limit_per_source)
    if not tasks:
        raise RuntimeError(f"no clean tasks built from {source_root}")
    workers = max(1, min(workers, len(tasks)))
    log(f"parallel clean start tasks={len(tasks)} workers={workers}")

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        future_map = {
            ex.submit(clean_task_worker, task, str(source_root), str(parts_root)): task
            for task in tasks
        }
        done = 0
        for fut in as_completed(future_map):
            task = future_map[fut]
            result = fut.result()
            results.append(result)
            done += 1
            stats = Counter(result.get("stats", {}))
            if done % 10 == 0 or done == len(tasks):
                log(
                    "parallel clean progress "
                    f"done={done}/{len(tasks)} task={task['id']} "
                    f"source={task['source']} total={stats['total']} kept={stats['kept_premerge']}"
                )

    log("parallel clean merge start")
    bucket_stats, source_stats = merge_clean_parts(out_root, results)
    write_clean_reports(out_root, bucket_stats, source_stats)
    log("parallel clean done")


def shard_jsonl(in_path: Path, out_dir: Path, prefix: str, max_raw_bytes: int) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, Any]] = []
    idx = 0
    raw_bytes = 0
    rows = 0
    fh = None
    gz_path: Path | None = None

    def open_next():
        nonlocal idx, raw_bytes, rows, fh, gz_path
        if fh is not None:
            fh.close()
            assert gz_path is not None
            shards.append(
                {
                    "file": rel(gz_path),
                    "bytes": gz_path.stat().st_size,
                    "raw_bytes": raw_bytes,
                    "rows": rows,
                    "sha256": hashlib.sha256(gz_path.read_bytes()).hexdigest(),
                }
            )
        gz_path = out_dir / f"{prefix}-{idx:05d}.jsonl.gz"
        idx += 1
        raw_bytes = 0
        rows = 0
        fh = gzip.open(gz_path, "wt", encoding="utf-8", compresslevel=6)

    open_next()
    with in_path.open("r", encoding="utf-8") as src:
        for line in src:
            data = line.encode("utf-8")
            if raw_bytes and raw_bytes + len(data) > max_raw_bytes:
                open_next()
            assert fh is not None
            fh.write(line)
            raw_bytes += len(data)
            rows += 1
    if fh is not None:
        fh.close()
        assert gz_path is not None
        if rows:
            shards.append(
                {
                    "file": rel(gz_path),
                    "bytes": gz_path.stat().st_size,
                    "raw_bytes": raw_bytes,
                    "rows": rows,
                    "sha256": hashlib.sha256(gz_path.read_bytes()).hexdigest(),
                }
            )
        else:
            gz_path.unlink(missing_ok=True)
    return shards


def read_clean_stats(out_root: Path) -> dict[str, Any]:
    path = out_root / "reports" / "clean_stats.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def dataset_readme(repo_id: str, bucket_names: list[str], stats: dict[str, Any]) -> str:
    lines = [
        "---",
        "license: other",
        "task_categories:",
        "- text-generation",
        "language:",
    ]
    langs = sorted({BUCKETS[b].language for b in bucket_names})
    for lang in langs:
        lines.append(f"- {lang}")
    lines.extend(
        [
            "---",
            "",
            f"# {repo_id.split('/')[-1]}",
            "",
            "Private Montlok corpus expansion snapshot.",
            "",
            "## Contents",
            "",
            "| Bucket | Language | Domain | Kept rows |",
            "| --- | --- | --- | ---: |",
        ]
    )
    bucket_stats = stats.get("buckets", {})
    for bucket in bucket_names:
        cfg = BUCKETS[bucket]
        kept = bucket_stats.get(bucket, {}).get("kept", 0)
        lines.append(f"| `{bucket}` | `{cfg.language}` | `{cfg.domain}` | {kept} |")
    lines.extend(
        [
            "",
            "## Layout",
            "",
            f"- `data/{RUN_ID}/`: gzip-compressed JSONL shards.",
            f"- `metadata/{RUN_ID}/clean_stats.json`: extraction and filtering counters.",
            "",
            "Each row has `id`, `text`, `language`, `domain`, `bucket`, `source`, `source_path`, and optional metadata.",
            "",
            "## License and provenance",
            "",
            "This is a derived private research corpus. Respect upstream source licenses and terms. Montlok packaging and downstream use must follow the applicable MTLK license policy plus all upstream restrictions.",
            "",
        ]
    )
    return "\n".join(lines)


def prepare_staging(out_root: Path, run_id: str, max_raw_bytes: int) -> dict[str, Path]:
    clean_dir = out_root / "clean"
    staging_root = out_root / "hf_staging"
    if staging_root.exists():
        for old in staging_root.rglob("*"):
            if old.is_file():
                old.unlink()
        for old in sorted((p for p in staging_root.rglob("*") if p.is_dir()), reverse=True):
            old.rmdir()
    staging_root.mkdir(parents=True, exist_ok=True)

    stats = read_clean_stats(out_root)
    repo_buckets: dict[str, list[str]] = defaultdict(list)
    shard_manifest: dict[str, Any] = {}
    for bucket, cfg in BUCKETS.items():
        path = clean_dir / f"{bucket}.clean.jsonl"
        if not path.exists() or path.stat().st_size == 0:
            continue
        repo_id = REPO_IDS[cfg.repo_key]
        repo_buckets[repo_id].append(bucket)
        repo_dir = staging_root / repo_id.replace("/", "__")
        shard_dir = repo_dir / "data" / run_id
        shards = shard_jsonl(path, shard_dir, bucket, max_raw_bytes=max_raw_bytes)
        shard_manifest[bucket] = shards

    repo_dirs: dict[str, Path] = {}
    for repo_id, buckets in sorted(repo_buckets.items()):
        repo_dir = staging_root / repo_id.replace("/", "__")
        metadata_dir = repo_dir / "metadata" / run_id
        metadata_dir.mkdir(parents=True, exist_ok=True)
        (metadata_dir / "clean_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (metadata_dir / "shards.json").write_text(json.dumps(shard_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        readme = dataset_readme(repo_id, buckets, stats)
        if repo_id == REPO_IDS["zh"]:
            (metadata_dir / "README.md").write_text(readme, encoding="utf-8")
        else:
            (repo_dir / "README.md").write_text(readme, encoding="utf-8")
        repo_dirs[repo_id] = repo_dir
    return repo_dirs


def upload_staging(
    repo_dirs: dict[str, Path],
    *,
    private: bool,
    force: bool,
    workers: int = 4,
    timeout_sec: int = 300,
) -> None:
    from huggingface_hub import HfApi

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("NO_PROXY", "*")
    os.environ.setdefault("no_proxy", "*")

    api = HfApi()
    upload_jobs: list[tuple[str, Path, str]] = []
    for repo_id, repo_dir in sorted(repo_dirs.items()):
        log(f"ensure HF dataset repo {repo_id}")
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
        remote_files: set[str] = set()
        if not force:
            try:
                remote_files = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
            except Exception as exc:
                log(f"WARN list remote files failed for {repo_id}: {exc}")
        files = sorted(path for path in repo_dir.rglob("*") if path.is_file())
        for path in files:
            path_in_repo = str(path.relative_to(repo_dir))
            if path_in_repo in remote_files and not force:
                log(f"skip existing {repo_id}:{path_in_repo}")
                continue
            upload_jobs.append((repo_id, path, path_in_repo))

    def upload_one(job: tuple[str, Path, str]) -> str:
        repo_id, path, path_in_repo = job
        code = (
            "import os, sys\n"
            "from huggingface_hub import HfApi\n"
            "path, path_in_repo, repo_id, run_id = sys.argv[1:5]\n"
            "os.environ.setdefault('HF_HUB_DISABLE_XET', '1')\n"
            "os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')\n"
            "os.environ.setdefault('NO_PROXY', '*')\n"
            "os.environ.setdefault('no_proxy', '*')\n"
            "HfApi().upload_file(\n"
            "    path_or_fileobj=path,\n"
            "    path_in_repo=path_in_repo,\n"
            "    repo_id=repo_id,\n"
            "    repo_type='dataset',\n"
            "    commit_message=f'Upload {run_id} corpus shard',\n"
            ")\n"
        )
        env = os.environ.copy()
        env.update(
            {
                "HF_HUB_DISABLE_XET": "1",
                "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                "NO_PROXY": "*",
                "no_proxy": "*",
            }
        )
        for attempt in range(1, 4):
            try:
                log(f"upload {repo_id}:{path_in_repo} ({path.stat().st_size} bytes)")
                subprocess.run(
                    [sys.executable, "-c", code, str(path), path_in_repo, repo_id, RUN_ID],
                    check=True,
                    timeout=timeout_sec,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                log(f"uploaded {repo_id}:{path_in_repo}")
                return path_in_repo
            except subprocess.TimeoutExpired:
                if attempt == 3:
                    raise
                log(f"WARN upload timed out attempt={attempt} file={repo_id}:{path_in_repo}")
                time.sleep(10 * attempt)
            except subprocess.CalledProcessError as exc:
                if attempt == 3:
                    raise RuntimeError(
                        f"upload failed for {repo_id}:{path_in_repo}: {exc.stderr[-1000:]}"
                    ) from exc
                log(
                    f"WARN upload failed attempt={attempt} file={repo_id}:{path_in_repo}: "
                    f"{exc.stderr[-500:]}"
                )
                time.sleep(10 * attempt)
            except Exception as exc:
                if attempt == 3:
                    raise
                log(f"WARN upload failed attempt={attempt} file={repo_id}:{path_in_repo}: {exc}")
                time.sleep(10 * attempt)
        raise RuntimeError("unreachable upload retry state")

    if not upload_jobs:
        log("upload: no files need uploading")
        return

    workers = max(1, min(workers, len(upload_jobs)))
    log(f"parallel upload start files={len(upload_jobs)} workers={workers}")
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        future_map = {ex.submit(upload_one, job): job for job in upload_jobs}
        for fut in as_completed(future_map):
            fut.result()
            done += 1
            if done % 10 == 0 or done == len(upload_jobs):
                log(f"parallel upload progress done={done}/{len(upload_jobs)}")


def write_status(out_root: Path, status: str, detail: dict[str, Any] | None = None) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "detail": detail or {},
    }
    (out_root / "run_status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    global RUN_ID

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--run-id", default=RUN_ID)
    ap.add_argument("--limit-per-source", type=int, default=0, help="debug limit per source extractor")
    ap.add_argument("--skip-clean", action="store_true")
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--force-upload", action="store_true")
    ap.add_argument("--max-shard-raw-mib", type=int, default=32)
    ap.add_argument("--clean-workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--upload-workers", type=int, default=4)
    ap.add_argument("--upload-timeout-sec", type=int, default=300)
    args = ap.parse_args(argv)

    RUN_ID = args.run_id

    write_status(args.out_root, "running", {"stage": "start", "run_id": args.run_id})
    try:
        if not args.skip_clean:
            write_status(args.out_root, "running", {"stage": "clean", "run_id": args.run_id})
            if args.clean_workers > 1:
                clean_all_parallel(
                    args.source_root,
                    args.out_root,
                    workers=args.clean_workers,
                    limit_per_source=args.limit_per_source,
                )
            else:
                clean_all(args.source_root, args.out_root, limit_per_source=args.limit_per_source)
        write_status(args.out_root, "running", {"stage": "stage_hf", "run_id": args.run_id})
        repo_dirs = prepare_staging(
            args.out_root,
            args.run_id,
            max_raw_bytes=args.max_shard_raw_mib * 1024 * 1024,
        )
        if not args.skip_upload:
            write_status(args.out_root, "running", {"stage": "upload", "repos": sorted(repo_dirs)})
            upload_staging(
                repo_dirs,
                private=args.private,
                force=args.force_upload,
                workers=args.upload_workers,
                timeout_sec=args.upload_timeout_sec,
            )
        write_status(args.out_root, "done", {"stage": "complete", "repos": sorted(repo_dirs)})
    except Exception as exc:
        write_status(args.out_root, "failed", {"error": repr(exc)})
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())

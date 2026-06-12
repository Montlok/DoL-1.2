# -*- coding: utf-8 -*-

"""Assemble the public model-distribution tree.

The project keeps two lines:

* **internal** (this repo, private) — the full training line: code, data
  pipelines, recipes, monitors.
* **public** (``Montlok/DoL``) — a model-distribution repo in the
  Hugging-Face-hub mold: model card (maintained by hand, never generated
  here), governance docs, model ``config.json``, the tokenizer bundle,
  and — when released — weights. No source code, no training material.

This script assembles that tree deterministically:

* copies the governance docs verbatim (``CONTRIBUTING.md``,
  ``SECURITY.md``, ``LICENSE``);
* emits ``config.json`` for the pretrain architecture via the HF adapter
  (:class:`Model.hf.configuration_rdt.RDTHFConfig`);
* copies the tokenizer bundle given with ``--tokenizer-bundle``;
* optionally copies released weight files given with ``--weights``;
* verifies the result: the bundle loads, ``config.json`` round-trips with
  a vocab size matching the bundle, and **no** ``.py``/source files are
  present.

It deliberately never writes ``README.md`` — the model card is authored
by hand in the public repo and must not be overwritten by tooling.

Usage::

    PYTHONPATH=. python3 -m scripts.release_public \
        --out /tmp/DoL_release \
        --tokenizer-bundle ../corpus/outputs/tok_build_v2/tokenizer/bundle
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Governance docs published verbatim. The public repo is bilingual like the
# internal one; the Chinese text is authoritative for governance statements.
GOVERNANCE_FILES = ["CONTRIBUTING.md", "SECURITY.md", "LICENSE"]


def _emit_config_json(out: Path) -> None:
    from Model.config import pretrain_config
    from Model.hf.configuration_rdt import RDTHFConfig

    hf_cfg = RDTHFConfig.from_rdt_config(pretrain_config())
    (out / "config.json").write_text(
        hf_cfg.to_json_string(), encoding="utf-8"
    )


def assemble(
    out: Path, tokenizer_bundle: Path, weights: Path | None, force: bool
) -> None:
    if out.exists():
        if not force:
            raise SystemExit(f"{out} exists; pass --force to replace it")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for name in GOVERNANCE_FILES:
        src = REPO_ROOT / name
        if not src.exists():
            raise SystemExit(f"governance file missing in repo: {name}")
        shutil.copy2(src, out / name)

    _emit_config_json(out)

    if not tokenizer_bundle.is_dir():
        raise SystemExit(f"tokenizer bundle not found: {tokenizer_bundle}")
    shutil.copytree(
        tokenizer_bundle,
        out / "tokenizer",
        ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"),
    )

    if weights is not None:
        if not weights.exists():
            raise SystemExit(f"weights path not found: {weights}")
        dst = out / "weights"
        if weights.is_dir():
            shutil.copytree(weights, dst)
        else:
            dst.mkdir()
            shutil.copy2(weights, dst / weights.name)


def verify(out: Path) -> None:
    # 1) Distribution repo carries data + docs only - no source code.
    code = [str(p.relative_to(out)) for p in out.rglob("*.py")]
    code += [str(p.relative_to(out)) for p in out.rglob("*.rs")]
    if code:
        raise SystemExit("[verify] source files leaked:\n  " + "\n  ".join(code))

    # 2) config.json parses and matches the bundle's vocabulary size.
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    rdt = cfg.get("rdt_config") or {}
    vocab_size = rdt.get("vocab_size")
    if not vocab_size:
        raise SystemExit("[verify] config.json carries no rdt_config.vocab_size")

    from Tokenizer.unified.bundle import TokenizerBundle

    bundle = TokenizerBundle.from_dir(out / "tokenizer")
    ids = bundle.encode("ᠮᠣᠩᠭᠣᠯ abc 中文", add_bos=True)
    if not ids or max(ids) >= vocab_size:
        raise SystemExit(
            f"[verify] bundle/config mismatch: max id {max(ids)} vs "
            f"vocab_size {vocab_size}"
        )
    print(
        f"[verify] OK: config vocab_size={vocab_size}, bundle encodes "
        f"{len(ids)} ids, no source files"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out", required=True, help="target directory")
    ap.add_argument(
        "--tokenizer-bundle",
        required=True,
        help="tokenizer bundle dir to publish (e.g. tok_build_v2/.../bundle)",
    )
    ap.add_argument(
        "--weights",
        default="",
        help="optional released weight file/dir to include under weights/",
    )
    ap.add_argument("--force", action="store_true", help="replace existing --out")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out).resolve()
    assemble(
        out,
        Path(args.tokenizer_bundle).resolve(),
        Path(args.weights).resolve() if args.weights else None,
        force=args.force,
    )
    n_files = sum(1 for p in out.rglob("*") if p.is_file())
    print(f"[release] assembled {n_files} files -> {out}")
    if not args.skip_verify:
        verify(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

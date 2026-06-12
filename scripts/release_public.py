# -*- coding: utf-8 -*-

"""Assemble the public runtime tree (the "use the model" surface).

The project keeps two lines:

* **internal** (this repo, private) — the full training line: data
  pipelines, corpus mixing, training/eval scripts, monitors, recipes.
* **public** — only what someone needs to *run* the model: architecture +
  inference, tokenizer runtime, encoding normalization, a standalone
  generate CLI, usage docs.

This script builds the public tree by whitelist-copying the runtime
surface into ``--out`` and generating the public-only files (trimmed
``Model/__init__``, standalone ``generate.py``, README, pyproject). It
then verifies the result: no training/posttrain/tools code present, no
imports referencing them, the tree imports cleanly, and a tiny CPU
generate smoke passes. Whitelist over blocklist: anything new in the
internal repo stays internal until explicitly added here.

Usage::

    PYTHONPATH=. python3 -m scripts.release_public --out /tmp/DoL_public
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories copied verbatim (minus _SKIP_NAMES entries).
COPY_DIRS = [
    "Model/layers",
    "Model/omvt",
    "Model/inference",
    "Model/hf",
    "Tokenizer/morphbpe",
    "Tokenizer/generic_bpe",
    "Tokenizer/unified",
    "Tokenizer/multimodal",
    "Tokenizer/traditional_mongolian",
    "Encoding Mapping",
]

COPY_FILES = [
    "Model/config.py",
    "Model/model.py",
    "Model/recurrent.py",
    "Model/two_stage.py",
    "Model/blocks.py",
    "Model/segmented.py",
    "Model/vision.py",
    "Tokenizer/__init__.py",
    "LICENSE",
]

# Pruned everywhere inside copied dirs.
_SKIP_NAMES = {"__pycache__", "target", ".DS_Store", ".ruff_cache"}

# Internal-only module roots: the assembled tree must not contain them nor
# import from them.
_FORBIDDEN = ("Model.training", "Model.posttrain", "Model.ocr", "Tokenizer.tools")

_MODEL_INIT = '''# -*- coding: utf-8 -*-

"""Recurrent depth transformer - public runtime surface."""

from Model.config import (
    OMVTConfig,
    RDTConfig,
    base_config,
    pretrain_config,
    small_config,
    tiny_config,
)
from Model.model import RDTForCausalLM

__all__ = [
    "OMVTConfig",
    "RDTConfig",
    "RDTForCausalLM",
    "base_config",
    "pretrain_config",
    "small_config",
    "tiny_config",
]
'''

_GENERATE = '''# -*- coding: utf-8 -*-

"""Generate text from an RDT checkpoint (standalone public CLI).

Usage::

    python generate.py --config pretrain --checkpoint path/to/model.pt \\
        --tokenizer-bundle path/to/bundle --prompt "..." --max-new-tokens 128
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from Model.config import (  # noqa: E402
    base_config,
    pretrain_config,
    segmented_pretrain_config,
    segmented_tiny_config,
    small_config,
    tiny_config,
    two_stage_pretrain_config,
    two_stage_tiny_config,
)
from Model.model import RDTForCausalLM  # noqa: E402
from Tokenizer.unified.bundle import TokenizerBundle  # noqa: E402

CONFIGS = {
    "tiny": tiny_config,
    "small": small_config,
    "base": base_config,
    "pretrain": pretrain_config,
    "two_stage_tiny": two_stage_tiny_config,
    "two_stage_pretrain": two_stage_pretrain_config,
    "segmented_tiny": segmented_tiny_config,
    "segmented_pretrain": segmented_pretrain_config,
}


def _official_mamba_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import mamba_ssm  # noqa: F401
    except Exception:
        return False
    return True


def resolve_mamba(cfg, mode: str, use_cache: bool):
    """official = CUDA fused kernels; naive = pure PyTorch (CPU/macOS, and
    the only backend the incremental decode cache can step)."""
    if use_cache and mode != "naive":
        raise SystemExit("--use-cache requires --mamba naive")
    if mode == "official":
        if not _official_mamba_usable():
            raise SystemExit(
                "--mamba official needs CUDA + a matching mamba-ssm wheel; "
                "use --mamba naive on CPU/macOS"
            )
        return replace(cfg, use_official_mamba=True)
    if mode == "naive":
        return replace(cfg, use_official_mamba=False)
    return replace(cfg, use_official_mamba=_official_mamba_usable())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--config", choices=list(CONFIGS), required=True)
    p.add_argument("--checkpoint", required=True,
                   help="a model.pt file or a directory containing one")
    p.add_argument("--tokenizer-bundle", required=True)
    p.add_argument("--prompt", default="")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--min-p", type=float, default=None)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--recurrent-steps", type=int, default=None,
                   help="latent refinement depth override (think harder/faster)")
    p.add_argument("--use-cache", action="store_true",
                   help="incremental decode (two_stage/segmented cores, naive)")
    p.add_argument("--mamba", choices=["auto", "official", "naive"],
                   default="auto")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") \\
        if args.device == "auto" else args.device

    ckpt = Path(args.checkpoint)
    if ckpt.is_dir():
        ckpt = ckpt / "model.pt"
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    bundle = TokenizerBundle.from_dir(args.tokenizer_bundle)
    cfg = resolve_mamba(CONFIGS[args.config](), args.mamba, args.use_cache)
    model = RDTForCausalLM(cfg)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state and "embed.weight" not in state:
        state = state["model"]
    model.load_state_dict(state)
    model.to(device).eval()

    prompt_ids = bundle.encode(args.prompt, add_bos=True)
    out = model.generate(
        torch.tensor([prompt_ids], dtype=torch.long, device=device),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        greedy=args.greedy,
        repetition_penalty=args.repetition_penalty,
        use_cache=args.use_cache,
        recurrent_steps=args.recurrent_steps,
    )
    completion = out[0].tolist()[len(prompt_ids):]
    print(bundle.tokenizer.decode(completion))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

_README = """# DoL — traditional Mongolian LLM (runtime)

DoL is a recurrent-depth transformer (RDT) for traditional Mongolian
(Hudum bichig) plus Chinese / English / Japanese, with an optional vision
tower for reading vertically-set Mongolian pages. This repository contains
everything needed to **run** the model: architecture, inference, the
dual-track tokenizer runtime, and encoding normalization.

DoL 是面向传统蒙古文(回鹘式蒙古文)及中/英/日的循环深度
Transformer,带可选视觉塔以阅读竖排蒙文页面。本仓库只包含**运行**模型
所需的部分:模型架构、推理、双轨 tokenizer 运行时与编码归一化。

## Install / 安装

```bash
pip install -e '.[model]'        # torch + tokenizer runtime
```

## Generate / 生成

```bash
python generate.py \\
    --config pretrain \\
    --checkpoint path/to/model.pt \\
    --tokenizer-bundle path/to/tokenizer/bundle \\
    --prompt "ᠮᠣᠩᠭᠣᠯ" --max-new-tokens 128
```

Python API:

```python
import torch
from Model import RDTForCausalLM, pretrain_config
from Tokenizer.unified.bundle import TokenizerBundle

bundle = TokenizerBundle.from_dir("path/to/bundle")
model = RDTForCausalLM(pretrain_config()).eval()
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
ids = torch.tensor([bundle.encode("ᠮᠣᠩᠭᠣᠯ", add_bos=True)])
out = model.generate(ids, max_new_tokens=64, greedy=True)
print(bundle.tokenizer.decode(out[0].tolist()))
```

## Encoding normalization / 编码归一化

Traditional Mongolian text exists in several historical encodings
(Menksoft PUA, MW, GB2010). The `Encoding Mapping/` Rust crate converges
them to nominal Unicode — the tokenizer's canonical representation — via
`normalize_to_nominal_unicode`. Pure-Unicode input needs no extra step.

传统蒙古文存在多套历史编码;`Encoding Mapping/` Rust crate 将其收敛为
名义 Unicode(tokenizer 的规范表示)。纯 Unicode 输入无需此步骤。

## Notes / 说明

- `--mamba naive` runs the pure-PyTorch SSM path (CPU/macOS); `--mamba
  official` uses fused CUDA kernels (`pip install mamba-ssm`).
- `--recurrent-steps` trades test-time compute for quality (latent depth).
- Weights and tokenizer bundles are published separately as model
  releases. 权重与 tokenizer bundle 以模型发布的形式单独提供。

## License

Apache-2.0
"""

_PYPROJECT = """[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "dol-runtime"
version = "0.1.0"
description = "Runtime for DoL: a recurrent-depth transformer for traditional Mongolian + zh/en/ja"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "Apache-2.0" }
dependencies = ["tokenizers>=0.19"]

[project.optional-dependencies]
model = ["torch>=2.1"]
hf = ["transformers>=4.40"]
image = ["Pillow>=9"]
cuda = ["mamba-ssm>=2.2"]

[tool.setuptools.packages.find]
where = ["."]
include = ["Tokenizer*", "Model*"]
"""

_GITIGNORE = """__pycache__/
*.pyc
*.pt
target/
.DS_Store
"""


def _copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns(*_SKIP_NAMES),
        dirs_exist_ok=False,
    )


def assemble(out: Path, force: bool) -> None:
    if out.exists():
        if not force:
            raise SystemExit(f"{out} exists; pass --force to replace it")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for rel in COPY_DIRS:
        _copy_tree(REPO_ROOT / rel, out / rel)
    for rel in COPY_FILES:
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / rel, dst)

    (out / "Model" / "__init__.py").write_text(_MODEL_INIT, encoding="utf-8")
    (out / "generate.py").write_text(_GENERATE, encoding="utf-8")
    (out / "README.md").write_text(_README, encoding="utf-8")
    (out / "pyproject.toml").write_text(_PYPROJECT, encoding="utf-8")
    (out / ".gitignore").write_text(_GITIGNORE, encoding="utf-8")


def verify(out: Path) -> None:
    # 1) No internal-only module roots made it into the tree.
    for forbidden in ("Model/training", "Model/posttrain", "Model/ocr",
                      "Tokenizer/tools", "Tokenizer/evals", "scripts"):
        if (out / forbidden).exists():
            raise SystemExit(f"[verify] internal path leaked: {forbidden}")

    # 2) No file references an internal-only module.
    offenders = []
    for path in out.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        for mod in _FORBIDDEN:
            if mod in text:
                offenders.append(f"{path.relative_to(out)}: {mod}")
    if offenders:
        raise SystemExit("[verify] internal imports leaked:\n  " + "\n  ".join(offenders))

    # 3) The tree imports on its own and a tiny CPU generate works.
    smoke = (
        "import torch; "
        "from Model import RDTForCausalLM, tiny_config; "
        "from Tokenizer.unified.bundle import TokenizerBundle; "
        "from dataclasses import replace; "
        "cfg = replace(tiny_config(), use_official_mamba=False, max_seq_len=32); "
        "m = RDTForCausalLM(cfg).eval(); "
        "out = m.generate(torch.tensor([[1, 300, 301]]), max_new_tokens=4, greedy=True); "
        "assert out.shape == (1, 7); "
        "print('[verify] import + generate smoke OK')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", smoke],
        cwd=out,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(out)},
    )
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit("[verify] smoke failed in assembled tree")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out", required=True, help="target directory for the public tree")
    ap.add_argument("--force", action="store_true", help="replace an existing --out")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out).resolve()
    assemble(out, force=args.force)
    n_files = sum(1 for _ in out.rglob("*") if _.is_file())
    print(f"[release] assembled {n_files} files -> {out}")
    if not args.skip_verify:
        verify(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

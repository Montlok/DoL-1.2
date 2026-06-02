# -*- coding: utf-8 -*-

"""Build a generative-OCR training set for traditional Mongolian.

Renders each transcription string to a vertical-script image and emits a
pre-tokenized JSONL row (see :mod:`Model.ocr.data` for the token contract) that
``scripts/train_vlm_align.py --data ...`` consumes directly. The rendered text
*is* the label, so the ground truth is exact and free of annotation cost.

Input: a UTF-8 text file (one transcription per line) or a JSONL with a
``text`` field per line. Output: ``<out>/images/*.png`` + ``<out>/data.jsonl``.

Vertical rendering needs Pillow built with **libraqm** (for ``direction="ttb"``)
and a traditional-Mongolian font (e.g. Menksoft Qagan / Noto Sans Mongolian).
This environment may lack both; the renderer fails with an actionable message
rather than producing wrong images. The token-row contract (:mod:`Model.ocr.data`)
is independently unit-tested without rendering.

Usage::

    python -m scripts.build_ocr_data \
        --input lines.txt --out data/ocr_synth \
        --font /path/to/MongolianFont.ttf \
        --tokenizer-bundle outputs/tok_build/tokenizer/bundle \
        --image-size 224
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Model.config import (  # noqa: E402
    BOS_ID,
    EOS_ID,
    IMAGE_END_ID,
    IMAGE_PATCH_ID,
    IMAGE_START_ID,
)
from Model.ocr.data import build_ocr_row  # noqa: E402


def _check_vertical_support() -> None:
    """Raise an actionable error if Pillow cannot render vertical text."""
    try:
        from PIL import features
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(f"Pillow is required for rendering: {exc}") from exc
    if not features.check("raqm"):
        raise RuntimeError(
            "Pillow lacks libraqm, so direction='ttb' vertical rendering is "
            "unavailable. Install a Pillow build with raqm (e.g. system libraqm "
            "+ 'pip install --force-reinstall pillow') and rerun on that host."
        )


def render_vertical_line(
    text: str,
    font_path: str,
    *,
    image_size: int = 224,
    font_size: int = 28,
    padding: int = 12,
    bg: int = 255,
    fg: int = 0,
):
    """Render ``text`` as a top-to-bottom Mongolian line on a square canvas.

    Returns a grayscale ``PIL.Image`` of side ``image_size``. Requires libraqm.
    """
    _check_vertical_support()
    from PIL import Image, ImageDraw, ImageFont

    if not Path(font_path).exists():
        raise FileNotFoundError(f"font not found: {font_path}")
    font = ImageFont.truetype(font_path, font_size)

    img = Image.new("L", (image_size, image_size), bg)
    draw = ImageDraw.Draw(img)
    # direction="ttb" needs raqm; layout columns left-to-right is the script's
    # natural flow and must not be pre-rotated to horizontal.
    draw.text(
        (padding, padding),
        text,
        font=font,
        fill=fg,
        direction="ttb",
    )
    return img


def _read_lines(path: str) -> list[str]:
    raw = Path(path).read_text(encoding="utf-8").splitlines()
    lines: list[str] = []
    for line in raw:
        line = line.strip()
        if not line:
            continue
        if line[0] in "{[":
            obj = json.loads(line)
            text = obj.get("text")
            if not text:
                raise ValueError(f"JSONL row missing 'text': {line[:80]}")
            lines.append(text)
        else:
            lines.append(line)
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Mongolian generative-OCR data")
    ap.add_argument("--input", required=True, help="text (one line/sample) or JSONL")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--font", required=True, help="traditional Mongolian .ttf/.otf")
    ap.add_argument("--tokenizer-bundle", required=True)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--font-size", type=int, default=28)
    ap.add_argument(
        "--n-image-tokens",
        type=int,
        default=None,
        help="<image_patch> slots per image; must equal OMVT compress_to. "
        "Defaults to image_patch_count(image_size, image_size).",
    )
    ap.add_argument(
        "--instruction",
        default="",
        help="optional prompt text inserted before the transcription target",
    )
    args = ap.parse_args()

    from Tokenizer.multimodal.image_placeholders import image_patch_count
    from Tokenizer.unified.bundle import TokenizerBundle

    n_image_tokens = args.n_image_tokens or image_patch_count(
        args.image_size, args.image_size
    )

    bundle = TokenizerBundle.from_dir(args.tokenizer_bundle)
    instruction_ids = (
        bundle.encode(args.instruction, add_bos=False, add_eos=False)
        if args.instruction
        else []
    )

    out_dir = Path(args.out)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    lines = _read_lines(args.input)
    if not lines:
        print("[build-ocr] no input lines")
        return 1

    n_written = 0
    with (out_dir / "data.jsonl").open("w", encoding="utf-8") as fh:
        for i, text in enumerate(lines):
            img = render_vertical_line(
                text,
                args.font,
                image_size=args.image_size,
                font_size=args.font_size,
            )
            rel = f"images/{i:08d}.png"
            img.save(out_dir / rel)

            target_ids = bundle.encode(text, add_bos=False, add_eos=False)
            if not target_ids:
                continue
            row = build_ocr_row(
                target_ids,
                n_image_tokens,
                rel,
                bos_id=BOS_ID,
                image_start_id=IMAGE_START_ID,
                image_patch_id=IMAGE_PATCH_ID,
                image_end_id=IMAGE_END_ID,
                eos_id=EOS_ID,
                instruction_ids=instruction_ids,
            )
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1

    print(
        f"[build-ocr] wrote {n_written} rows -> {out_dir/'data.jsonl'} "
        f"(n_image_tokens={n_image_tokens}, image_size={args.image_size})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

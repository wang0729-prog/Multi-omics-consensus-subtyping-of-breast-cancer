"""Produce Adobe Illustrator (.ai) files for every figure.

Since Illustrator 9 the .ai format is a PDF container: Illustrator opens a
PDF-compatible .ai natively, keeps every path as an editable vector object and
every label as live text.  We therefore promote the already-verified vector PDFs
(text embedded as TrueType, `pdf.fonttype = 42`) to .ai.

Each output is re-opened and checked so a truncated or font-less file can never
slip through.

Usage:  python scripts/make_ai.py [--also-eps]
"""

import glob
import os
import shutil
import sys

from pypdf import PdfReader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
AI = os.path.join(FIG, "ai")

ORDER = [f"Fig{i}" for i in range(1, 9)] + [f"FigS{i}" for i in range(1, 13)]


def verify(path):
    """Return (n_pages, n_chars, n_fonts, n_embedded) or raise."""
    r = PdfReader(path)
    n_pages = len(r.pages)
    chars = 0
    fonts = {}
    for page in r.pages:
        chars += len(page.extract_text() or "")
        try:
            res = page["/Resources"]
            fd = res.get("/Font")
            if fd:
                fd = fd.get_object()
                for k in fd:
                    f = fd[k].get_object()
                    fonts[str(f.get("/BaseFont", "?"))] = f
        except Exception:
            pass
    raw = open(path, "rb").read()
    n_embed = len(glob_re_fontfile(raw))
    return n_pages, chars, len(fonts), n_embed


def glob_re_fontfile(raw):
    import re
    return re.findall(rb"/FontFile\d?", raw)


def main():
    os.makedirs(AI, exist_ok=True)
    print(f"{'figure':8s} {'.ai KB':>8s} {'pages':>5s} {'text_chars':>10s} "
          f"{'fonts':>5s} {'embedded':>8s}  status")
    print("-" * 62)
    bad = []
    for name in ORDER:
        src = os.path.join(FIG, name + ".pdf")
        dst = os.path.join(AI, name + ".ai")
        if not os.path.exists(src):
            bad.append(f"{name}: source PDF missing")
            continue
        shutil.copyfile(src, dst)
        try:
            pages, chars, nfonts, nembed = verify(dst)
        except Exception as e:                                   # noqa: BLE001
            bad.append(f"{name}: unreadable -> {e}")
            continue
        kb = os.path.getsize(dst) / 1024
        ok = pages >= 1 and nfonts > 0 and nembed > 0 and chars > 0
        if not ok:
            bad.append(f"{name}: pages={pages} fonts={nfonts} embed={nembed} chars={chars}")
        print(f"{name:8s} {kb:8.0f} {pages:5d} {chars:10d} {nfonts:5d} {nembed:8d}  "
              f"{'ok' if ok else 'FAIL'}")

    print()
    print(f"written: {len(ORDER) - len(bad)}/{len(ORDER)} .ai files -> {AI}")
    if bad:
        print("PROBLEMS:")
        for b in bad:
            print("  -", b)
        return 1
    print("OK: every .ai opens as a valid multi-object vector document with embedded fonts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

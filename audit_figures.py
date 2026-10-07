"""Audit every figure for real editability.

SVG must carry live <text> nodes (not paths) so labels stay editable.
PDF must embed fonts as TrueType (fonttype 42), not outlined glyphs.
"""

import glob
import os
import re
import subprocess
import sys

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

ORDER = [f"Fig{i}" for i in range(1, 9)] + [f"FigS{i}" for i in range(1, 13)]


def svg_stats(path):
    s = open(path, encoding="utf-8").read()
    texts = re.findall(r"<text[^>]*>(.*?)</text>", s, re.S)
    labels = [re.sub(r"<[^>]+>", "", t).strip() for t in texts]
    labels = [l for l in labels if l]
    paths = len(re.findall(r"<path", s))
    imgs = len(re.findall(r"<image", s))
    return len(labels), paths, imgs, len(s)


def pdf_fonts(path):
    """Return (n_fonts, n_embedded, font_names).

    matplotlib with pdf.fonttype=42 writes composite fonts as
    /Type0 -> /CIDFontType2 with a /FontFile2 (embedded TrueType) stream.
    A simple /Subtype /TrueType check misses that, so scan the raw bytes.
    """
    b = open(path, "rb").read()
    names = sorted({
        x.decode("latin-1")
        for x in re.findall(rb"/BaseFont\s*/([#\w+.,-]+)", b)
    })
    embedded = len(re.findall(rb"/FontFile\d?", b))
    return len(names), embedded, names


def outlined_glyphs(path):
    """True when text was converted to outlines (no font resources at all)."""
    n, emb, _ = pdf_fonts(path)
    return n == 0 or emb == 0


def main():
    rows = []
    bad = []
    for name in ORDER:
        svg = os.path.join(FIG, name + ".svg")
        pdf = os.path.join(FIG, name + ".pdf")
        png = os.path.join(FIG, name + ".png")
        if not (os.path.exists(svg) and os.path.exists(pdf) and os.path.exists(png)):
            bad.append(f"{name}: missing file")
            continue
        ntxt, npath, nimg, size = svg_stats(svg)
        nfont, nembed, names = pdf_fonts(pdf)
        dejavu = [x for x in names if "DejaVu" in x]
        rows.append((name, ntxt, npath, nimg, size, os.path.getsize(pdf),
                     os.path.getsize(png), nfont, nembed, dejavu))
        if ntxt == 0:
            bad.append(f"{name}: SVG has no live text (labels are paths)")
        if nembed == 0:
            bad.append(f"{name}: PDF fonts not embedded (text outlined)")
        if dejavu:
            bad.append(f"{name}: non-Arial font in PDF -> {dejavu}")

    print(f"{'figure':8s} {'svg_txt':>7s} {'vec_path':>8s} {'raster':>6s} "
          f"{'svg_kb':>7s} {'pdf_kb':>7s} {'png_kb':>7s} {'pdf_fonts':>9s} {'embed':>5s}  face")
    print("-" * 92)
    for (name, ntxt, npath, nimg, ssz, psz, qsz, nfont, nembed, dejavu) in rows:
        face = "mixed!" if dejavu else "Arial"
        print(f"{name:8s} {ntxt:7d} {npath:8d} {nimg:6d} "
              f"{ssz/1024:7.0f} {psz/1024:7.0f} {qsz/1024:7.0f} {nfont:9d} {nembed:5d}  {face}")

    print()
    tot_txt = sum(r[1] for r in rows)
    print(f"figures audited : {len(rows)}")
    print(f"live SVG texts  : {tot_txt}")
    print(f"raster <image> in SVG: {sum(r[3] for r in rows)}")
    if bad:
        print("\nPROBLEMS:")
        for b in bad:
            print("  -", b)
    else:
        print("\nOK: every figure has live SVG text and embedded PDF fonts.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

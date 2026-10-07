"""Check that every figure panel actually drawn is cited in the manuscript.

Why: figure legends and manuscript text are written independently of the
plotting code, so panels silently drift -- a legend that describes a "study
design" panel that was never drawn, or a panel such as Fig. 1f that no sentence
ever cites.  This script derives the ground truth mechanically from the
`panel_label(ax, "x")` calls in the plotting scripts and then scans the
manuscript for `Fig. Na` / `Supplementary Fig. Na` references, including
compact NC-style lists ("Fig. 4d, e") and ranges ("Fig. 7a-f").

Usage:
    python scripts/check_panel_coverage.py [manuscript.md]

Exit code is 1 when any panel is uncited or any citation points at a panel
that does not exist.
"""

import re
import sys
import pathlib
import collections

ROOT = pathlib.Path(__file__).resolve().parents[1]

# ---- ground truth: panels actually drawn, derived from the plotting scripts --
# Each entry maps (script, figure name) -> ordered set of panel letters.
SCRIPT_FIGURES = [
    ("fig1.py", "F1", "Fig1"),
    ("fig2.py", "F2", "Fig2"),
    ("fig3.py", "F3", "Fig3"),
    ("fig4_5.py", "F4", "Fig4"),
    ("fig4_5.py", "F5", "Fig5"),
    ("fig6.py", "F6", "Fig6"),
    ("20_singlecell.py", "F7", "Fig7"),
    ("20_singlecell.py", "F8", "Fig8"),
    ("figS.py", "S1", "FigS1"),
    ("figS.py", "S2", "FigS2"),
    ("figS.py", "S3", "FigS3"),
    ("figS.py", "S4", "FigS4"),
    ("figS.py", "S5", "FigS5"),
    ("figS.py", "S6", "FigS6"),
    ("figS.py", "S7", "FigS7"),
    ("figS.py", "S8", "FigS8"),
    ("figS.py", "S9", "FigS9"),
    ("figS.py", "S10", "FigS10"),
    ("figS.py", "S11", "FigS11"),
    ("figS.py", "S12", "FigS12"),
]

# letter-expanding calls such as panel_label(ax, "abcd"[k]) need the index
EXPANDERS = re.compile(r'panel_label\([^,]+,\s*["\']([a-z]{2,})["\']\s*\[')

SAVE = re.compile(r'save_fig\([^,]+,\s*\w+,\s*["\'](\w+)["\']\s*\)')


def split_by_save(src):
    """Segment a plotting script at its save_fig() calls.

    Several scripts emit more than one figure from a single main(), so slicing
    by top-level def is not enough: the panels drawn for FigN are exactly the
    panel_label() calls between the previous save_fig() and the one that writes
    FigN.
    """
    segments = {}
    prev_end = 0
    for m in SAVE.finditer(src):
        segments[m.group(1)] = src[prev_end:m.start()]
        prev_end = m.end()
    return segments


def panels_in(segment):
    out = set(re.findall(r'panel_label\([^,]+,\s*["\']([a-z])["\']', segment))
    for grp in EXPANDERS.findall(segment):
        out |= set(grp)
    return out


def build_actual():
    cache = {}
    actual = {}
    for script, key, wanted in SCRIPT_FIGURES:
        if script not in cache:
            src = (ROOT / "scripts" / script).read_text(encoding="utf-8")
            cache[script] = split_by_save(src)
        seg = cache[script].get(wanted)
        if seg is not None:
            letters = panels_in(seg)
            if letters:
                actual[key] = "".join(sorted(letters))
    return actual


# ---- cited panels ----------------------------------------------------------
# Panel letters may be lowercase (Nature-family style, "Fig. 1a") or uppercase
# (Cell style, "Figure 1A"); both are accepted and normalised to lowercase.
# The figure number is optionally repeated before later panels and inside
# ranges, which is how the two common phrasings are written:
#   "Figure 4D and 4E", "Figure 2B-2D", "Fig. 4d, e", "Fig. 7a-f".
_N = r"(?:\d+\s*)?"                    # optional repeated figure number
_ONE = r"(?:[a-hA-H](?:\s*[\u2013-]\s*%s[a-hA-H])?)" % _N
REF = re.compile(
    r"((?:Supplementary\s+)?Fig(?:ure)?s?\.?\s*)"          # label
    r"(S?\d+)"                                            # figure number (S = supplementary)
    r"((?:\s*(?:and|,|\u2013|-)\s*)?" + _ONE +         # first panel / range
    r"(?:\s*(?:and|,)\s*%s" % _N + _ONE + r")*)?"      # more panels (optional)
)


def cited_panels(text):
    """Return {figure key: set of cited panel letters}.

    A citation with no panel letter ("Supplementary Fig. 6") refers to the
    figure as a whole and is recorded under the empty-string key so that
    single-panel figures are not reported as uncited.
    """
    found = collections.defaultdict(set)
    for m in REF.finditer(text):
        num = m.group(2)
        supp = num.upper().startswith("S") or "Supplementary" in m.group(1)
        key = ("S" if supp else "F") + num.lstrip("Ss")
        tail = (m.group(3) or "").lower()
        if not tail.strip():
            found[key].add("")            # figure-level citation
            continue
        for a, b in re.findall(r"([a-h])\s*[\u2013-]\s*([a-h])", tail):
            for c in range(ord(a), ord(b) + 1):
                found[key].add(chr(c))
        for c in re.findall(r"[a-h]", tail):
            found[key].add(c)
        # letters glued straight onto the number, e.g. "Fig. 1a" / "Figure 1A"
        for c in re.findall(r"[Ss]?\d+\s*([a-hA-H])\b", m.group(0)):
            found[key].add(c.lower())
    return found


def cited_sequence(text):
    """Return (figure_order, panel_order) from a first-citation scan.

    figure_order: list of figure keys ("F1", "S4", ...) in order of first
    citation.  panel_order: {figure key: string of panel letters in order of
    first citation} (ranges expanded; figure-level citations ignored).
    NC requires both to be alphabetical.
    """
    fig_order, seen_figs = [], set()
    panel_order = {}
    for m in REF.finditer(text):
        num = m.group(2)
        supp = num.upper().startswith("S") or "Supplementary" in m.group(1)
        key = ("S" if supp else "F") + num.lstrip("Ss")
        if key not in seen_figs:
            seen_figs.add(key)
            fig_order.append(key)
        tail = (m.group(3) or "").lower()
        if not tail.strip():
            continue
        letters = []
        for a, b in re.findall(r"([a-h])\s*[\u2013-]\s*([a-h])", tail):
            letters.extend(chr(c) for c in range(ord(a), ord(b) + 1))
        letters.extend(re.findall(r"[a-h]", tail))
        seq = panel_order.setdefault(key, [])
        for c in letters:
            if c not in seq:
                seq.append(c)
    return fig_order, panel_order


def order_check(fig_order, panel_order):
    """Return a list of (figure key, detail) describing order violations."""
    problems = []
    for family in ("F", "S"):
        keys = [k for k in fig_order if k.startswith(family)]
        nums = [int(k[1:]) for k in keys]
        if nums != sorted(nums):
            bad = [k for i, k in enumerate(keys)
                   if i and int(k[1:]) < nums[i - 1]]
            problems.append((family,
                             f"figure numbering out of citation order: "
                             f"{' -> '.join(keys)} (first offender {bad[0]})"))
    for key, seq in sorted(panel_order.items()):
        if seq != sorted(seq):
            problems.append((key, f"panels first cited as '{seq}' "
                                  f"(should be '{''.join(sorted(seq))}')"))
    return problems


def main():
    target = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 \
        else ROOT / "Manuscript_NC.md"
    text = target.read_text(encoding="utf-8")
    # body only: from Introduction up to the figure-legend block.  Legends must
    # be excluded, otherwise the "Fig. N | title" headings count as figure-level
    # citations and every panel looks cited.
    start = text.find("## Introduction")
    if start == -1:
        start = 0
    stop = len(text)
    for marker in ("## Figure legends", "## Figure titles and legends",
                   "## Supplementary figure legends"):
        at = text.find(marker, start)
        if at != -1:
            stop = min(stop, at)
    body = text[start:stop]

    actual = build_actual()
    cited = cited_panels(body)

    problems = 0
    print(f"{'fig':6s} {'drawn':10s} {'cited':10s} status")
    print("-" * 62)
    for key in actual:
        drawn = set(actual[key])
        got = cited.get(key, set())
        whole = "" in got                      # cited without a panel letter
        got = {c for c in got if c}
        missing = [] if whole else sorted(drawn - got)
        ghost = sorted(got - drawn)
        notes = []
        if whole:
            notes.append("figure-level citation")
        if missing:
            notes.append("UNCITED " + "".join(missing))
            problems += 1
        if ghost:
            notes.append("GHOST " + "".join(ghost))
            problems += 1
        print(f"{key:6s} {actual[key]:10s} {''.join(sorted(got)) or '-':10s} "
              f"{'; '.join(notes) if notes else 'ok'}")
    # figures cited but not found in the scripts at all
    for key in cited:
        if key not in actual:
            print(f"{key:6s} {'-':10s} {'(figure-level)' if '' in cited[key] else ''.join(sorted(cited[key])):10s} "
                  f"NO SUCH FIGURE IN SCRIPTS")
            problems += 1

    # citation-order check (NC: panels and figures are numbered in the order
    # they are first cited)
    fig_order, panel_seq = cited_sequence(body)
    for key, detail in order_check(fig_order, panel_seq):
        print(f"{key:6s} {'':10s} {'':10s} ORDER {detail}")
        problems += 1

    print()
    print("RESULT:", "clean" if problems == 0 else f"{problems} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

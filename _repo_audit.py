# -*- coding: utf-8 -*-
"""仓库合规断言：公开副本不能带的东西。

用法： python _qa/_repo_audit.py [repo_root]
退出码 0 = 通过
"""
import os
import re
import subprocess
import sys

LOCAL_PATH = re.compile(
    r"[A-Za-z]:[\\/]Users|[A-Za-z]:[\\/]workbuddy|/Users/|/c/Users|workbuddy\d\d-\d\d-"
)
REQUIRED = ["README.md", "requirements.txt",
            "scripts/01_download.sh", "scripts/run_all.sh", "scripts/lib.py",
            "_qa/_static_check.py", "_qa/_repo_audit.py", "_qa/_negative_control.py"]
BANNED_EXT = (".rds", ".dta", ".sav", ".rdata")
# .sh 必须参与扫描：run_all.sh 这类入口最容易藏本机解释器路径，
# 不扫就等于给本机路径留了一个免检口子。
TEXT_EXT = {".py", ".R", ".sh", ".md", ".txt", ".yml", ".yaml", ".json", ".csv",
            ".cfg", ".ini"}
_ID_COLS = {"id", "sample_id", "patient_id", "barcode", "submitted_sample_id"}


def iter_files(root):
    # _qa/ 是检测工具，源码里必然含有检测用的模式字面量（C:/Users 等），
    # 不排除会自证清白地误报。分析代码与 README 才参与文本扫描。
    for dp, dn, fns in os.walk(root):
        dn[:] = [d for d in dn
                 if not d.startswith("_") and d not in {".git", ".venv"}]
        if os.path.basename(dp).startswith("_"):
            continue
        for fn in fns:
            yield os.path.join(dp, fn)


def main(argv):
    root = argv[1] if len(argv) > 1 else os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))
    probs = []

    for p in iter_files(root):
        ext = os.path.splitext(p)[1].lower()
        if ext in BANNED_EXT:
            probs.append(f"banned-data-file  {os.path.relpath(p, root)}")
            continue
        if ext not in TEXT_EXT:
            continue
        try:
            txt = open(p, encoding="utf-8").read()
        except UnicodeDecodeError:
            continue
        for m in LOCAL_PATH.finditer(txt):
            line = txt[:m.start()].count("\n") + 1
            probs.append(f"local-path  {os.path.relpath(p, root)} L{line}  {m.group()[:60]!r}")

    for rel in REQUIRED:
        if not os.path.exists(os.path.join(root, rel)):
            probs.append(f"missing-required-file  {rel}")

    import csv
    for p in iter_files(root):
        if p.lower().endswith(".csv"):
            try:
                with open(p, encoding="utf-8", errors="ignore") as fh:
                    r = csv.reader(fh)
                    head = next(r, [])
                    n = sum(1 for _ in r)
            except Exception:
                continue
            if n > 2000 and any(h.strip().lower() in _ID_COLS for h in head):
                probs.append(f"individual-level-csv  {os.path.relpath(p, root)} ({n} rows)")

    rc = subprocess.run([sys.executable, os.path.join("_qa", "_static_check.py"), root],
                        cwd=root, capture_output=True, text=True)
    if rc.stdout and rc.stdout.strip():
        print(rc.stdout.strip().splitlines()[-1])
    if rc.returncode:
        probs.append("static-check  FAILED")

    for p in probs:
        print("PROBLEM:", p)
    print(f"REPO AUDIT: {len(probs)} problem(s) in {root}")
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

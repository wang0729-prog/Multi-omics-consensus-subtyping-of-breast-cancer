# -*- coding: utf-8 -*-
"""负对照：往干净副本注入 5 类缺陷，确认检查器真会报错。

只输出 "0 problems" 的检查器等于没有检查器。
用法： python _qa/_negative_control.py
退出码 0 = 5 类全部被抓到
"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)


def run(repo, script):
    rc = subprocess.run([sys.executable, os.path.join("_qa", script), repo],
                        cwd=repo, capture_output=True, text=True)
    return rc.stdout or ""


def write(repo, rel, text):
    p = os.path.join(repo, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w", encoding="utf-8").write(text)


def defects(repo):
    d = os.path.join(repo, "tmpmod")
    os.makedirs(d, exist_ok=True)

    write(repo, os.path.join("tmpmod", "g1.py"), "x = undefined_thing_xyz + 1\n")
    yield ("undefined-global", "undefined-global", run(repo, "_static_check.py"))

    write(repo, os.path.join("tmpmod", "g2.py"),
          "import os\nD = 'D:/workbuddy/2026-09-11-08-55-33'\nOUT = os.path.join(D, 'out')\n")
    yield ("path-var-local-path", "path-var-local-path", run(repo, "_static_check.py"))

    write(repo, os.path.join("tmpmod", "g3.py"), 'f = "C:/Users/tester/data"\n')
    yield ("local-path-literal", "local-path-literal", run(repo, "_static_check.py"))

    write(repo, os.path.join("tmpmod", "g4.py"), 'send = open("session.rds", "rb")\n')
    yield ("private-data-ref", "private-data-ref", run(repo, "_static_check.py"))

    write(repo, os.path.join("tmpmod", "g5.py"),
          'note = "see \u2014 the C:/Users/tester folder"\n')
    yield ("ai-glyph + local-path-literal", "ai-glyph", run(repo, "_static_check.py"))


def main():
    tmp = tempfile.mkdtemp(prefix="negctl_")
    repo = os.path.join(tmp, "repo")
    shutil.copytree(SRC, repo, ignore=shutil.ignore_patterns("__pycache__"))
    try:
        caught = 0
        for label, kw, out in defects(repo):
            ok = kw in out
            caught += ok
            print(("CAUGHT   " if ok else "MISSED   ") + label)
            if not ok:
                print("".join(out[-600:]))
        # 注入的缺陷文件必须清掉，否则 BASELINE 会包含它们，基线永远不干净
        shutil.rmtree(os.path.join(repo, "tmpmod"), ignore_errors=True)
        base = run(repo, "_static_check.py")
        print("BASELINE:", base.strip().splitlines()[-1] if base.strip() else "")
        return 0 if caught == 5 else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""静态检查：语法、未定义全局、路径变量、私有数据引用、AI 排版痕迹。

用法： python _qa/_static_check.py [repo_root]
退出码 0 = 无 problem
"""
import ast
import builtins
import os
import sys

def _an(a):
    """ast.arg 只有 .arg（asname/name 是 ast.alias 的属性）。"""
    return getattr(a, "asname", None) or getattr(a, "arg", None) or a.name


_KNOWN = set(dir(builtins)) | {
    "np", "pd", "plt", "os", "sys", "re", "json", "math", "itertools", "warnings",
    "pickle", "collections", "Counter", "defaultdict", "Path", "List", "Dict",
    "Tuple", "Optional",
    "__file__", "__name__", "__doc__", "__package__", "__loader__", "__spec__",
}

_PRIVATE_SUFFIX = (".rds", ".dta", ".sav", ".rdata")

_PATH_VARS = {
    "dir", "D", "f", "p", "path", "proj", "PROJ", "LIB", "TMPD", "TMP",
    "OUT", "RES", "FIG", "PRC", "RAW", "ROOT", "BASE", "WD",
}

_LOCAL_PATH_HINTS = (
    "workbuddy", "researchdata", r"C:\\", "C:/", r"D:\\", "D:/", "/Users/",
    "C:\\Users", "D:\\Users",
)

_AI_GLYPHS = "\u2014\u2013\u2018\u2019\u201c\u201d"


def _bindings(tree):
    out = set()

    class V(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            out.add(node.name)
            out.update(_an(a) for a in node.args.args)
            out.update(_an(a) for a in node.args.kwonlyargs)
            if node.args.vararg:
                out.add(node.args.vararg.arg)
            if node.args.kwarg:
                out.add(node.args.kwarg.arg)
            self.generic_visit(node)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node):
            out.add(node.name)
            self.generic_visit(node)

        def visit_Name(self, node):
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                out.add(node.id)

        def visit_ExceptHandler(self, node):
            if node.name:
                out.add(node.name)
            self.generic_visit(node)

        def visit_Import(self, node):
            for a in node.names:
                out.add((a.asname or a.name).split(".")[0])

        def visit_ImportFrom(self, node):
            for a in node.names:
                out.add(a.asname or a.name)

    V().visit(tree)
    for n in ast.walk(tree):
        if isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for g in n.generators:
                for t in ast.walk(g.target):
                    if isinstance(t, ast.Name):
                        out.add(t.id)
        if isinstance(n, (ast.With, ast.AsyncWith)):
            for item in n.items:
                if item.optional_vars is not None:
                    for t in ast.walk(item.optional_vars):
                        if isinstance(t, ast.Name):
                            out.add(t.id)
        if isinstance(n, ast.NamedExpr):
            out.add(n.target.id)
        if isinstance(n, ast.Global):
            out.update(n.names)
    return out


def _loads(tree):
    return {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _str_consts(tree):
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def _is_local_path(s):
    return any(h.lower() in s.lower() for h in _LOCAL_PATH_HINTS)


def check_file(path):
    probs = []
    src = open(path, encoding="utf-8").read()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [("syntax", e.lineno or 0, f"{path}: {e.msg}")]

    for name in sorted(_loads(tree) - _bindings(tree) - _KNOWN):
        probs.append(("undefined-global", 0, name))

    for s in _str_consts(tree):
        if _is_local_path(s):
            flag = "  <-- 路径变量被写进引号" if any(v in s for v in _PATH_VARS) else ""
            probs.append(("local-path-literal", 0, f"{s[:70]}{flag}"))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if not (isinstance(tgt, ast.Name) and tgt.id in _PATH_VARS):
                continue
            v = node.value
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                if _is_local_path(v.value):
                    probs.append(("path-var-local-path", getattr(node, "lineno", 0),
                                  f"{tgt.id} = {v.value[:70]!r}"))
            elif isinstance(v, ast.Constant) and v.value is None:
                probs.append(("path-var-dangling", getattr(node, "lineno", 0),
                              f"{tgt.id} = None"))

    for s in _str_consts(tree):
        low = s.lower()
        if any(suf in low for suf in _PRIVATE_SUFFIX):
            if any(k in low for k in ("save", "write", "output", "out")):
                continue
            probs.append(("private-data-ref", 0, s[:60]))

    for i, line in enumerate(src.splitlines(), 1):
        if any(g in line for g in _AI_GLYPHS):
            probs.append(("ai-glyph", i, line.strip()[:70]))
    return probs


def iter_py(root):
    # 跳过下划线开头的目录：_qa/ 是检测工具本身（源码里就含检测用的模式字面量），
    # _negctl* 是负对照注入的缺陷目录。两者计入会让基线永远不干净。
    for dp, dn, fns in os.walk(root):
        dn[:] = [d for d in dn if not d.startswith("_") and d != ".git"]
        for fn in sorted(fns):
            if fn.endswith(".py"):
                yield os.path.join(dp, fn)


def main(argv):
    root = argv[1] if len(argv) > 1 else os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))
    total = 0
    for path in iter_py(root):
        probs = check_file(path)
        if probs:
            print(f"--- {os.path.relpath(path, root)}")
            for rule, line, text in probs:
                total += 1
                print(f"    [{rule}] L{line if line else '-'}  {text}")
    n = len(list(iter_py(root)))
    print(f"STATIC CHECK: {total} problem(s) over {n} python file(s) in {root}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

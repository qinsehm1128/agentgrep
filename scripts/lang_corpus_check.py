#!/usr/bin/env python3
"""Real-corpus check for non-Rust languages.

For each repo, pick definitions whose name is unique in the repo (declared
once, in a non-test file), query smart mode with the identifier and with its
split words, and report hit@1 (top file is the defining file).

usage: lang_corpus_check.py BIN REPO LANG [sample]
LANG: go | java
"""
import json
import random
import re
import subprocess
import sys
from pathlib import Path

PATTERNS = {
    "go": (".go", re.compile(r"^func (?:\([^)]*\)\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.M), "_test.go"),
    "java": (
        ".java",
        re.compile(r"^\s+(?:public|protected|private)[\w\s<>\[\],?]*?\s([a-z][A-Za-z0-9_]*)\s*\(", re.M),
        "/test/",
    ),
}


def split_words(name):
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name).replace("_", " ").lower().split()
    return " ".join(words)


def main():
    binary, repo, lang = sys.argv[1:4]
    sample = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    ext, pattern, test_marker = PATTERNS[lang]
    root = Path(repo)
    defs = {}
    for path in root.rglob(f"*{ext}"):
        rel = str(path.relative_to(root))
        if test_marker in rel or "/." in rel:
            continue
        for name in pattern.findall(path.read_text(errors="replace")):
            defs.setdefault(name, set()).add(rel)
    unique = sorted(
        (n, next(iter(p))) for n, p in defs.items() if len(p) == 1 and len(split_words(n).split()) >= 2
    )
    random.Random(7).shuffle(unique)
    unique = unique[:sample]
    hits = {"symbol": 0, "split": 0}
    for name, expected in unique:
        for mode, subject in (("symbol", name), ("split", split_words(name))):
            out = subprocess.run(
                [binary, "smart", f"subject:{subject}", "relation:defined", "--json"],
                cwd=root, capture_output=True, text=True,
            ).stdout
            files = json.loads(out)["files"] if out.strip() else []
            if files and files[0]["path"] == expected:
                hits[mode] += 1
    n = len(unique)
    print(f"{lang:5} n={n:3}  symbol hit@1={100 * hits['symbol'] / n:5.1f}%  split hit@1={100 * hits['split'] / n:5.1f}%")


if __name__ == "__main__":
    main()

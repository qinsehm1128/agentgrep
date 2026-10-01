#!/usr/bin/env python3
"""Compare smart-mode latency of two agentgrep binaries on one corpus.

usage: ranking_latency.py OLD_BIN NEW_BIN CORPUS [runs]
"""
import statistics
import subprocess
import sys
import time

QUERIES = [
    ("active_backend", "defined"),
    ("CognitionDb", "defined"),
    ("reconcile candidates", "defined"),
    ("memory agent", "defined"),
    ("topic shift detection", "implementation"),
    ("convert web page to markdown", "implementation"),
    ("reciprocal rank fusion of bm25 and dense", "implementation"),
]


def run(binary, corpus, subject, relation, runs):
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        subprocess.run(
            [binary, "smart", f"subject:{subject}", f"relation:{relation}", "--json"],
            cwd=corpus,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def main():
    old, new, corpus = sys.argv[1:4]
    runs = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    print(f"{'query':45} {'old ms':>8} {'new ms':>8} {'ratio':>6}")
    for subject, relation in QUERIES:
        o = run(old, corpus, subject, relation, runs)
        n = run(new, corpus, subject, relation, runs)
        print(f"{subject:45} {o:8.0f} {n:8.0f} {n / o:6.2f}")


if __name__ == "__main__":
    main()

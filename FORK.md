# Fork notes

This is a fork of [1jehuang/agentgrep](https://github.com/1jehuang/agentgrep),
maintained for use inside qin-code.

- **Base:** upstream `master` at `66b1148` (one commit after tag `v0.1.7`).
- **License:** upstream declares MIT in `Cargo.toml` and the README but ships
  no LICENSE file. `LICENSE` here is the standard MIT text with the original
  author's copyright, added to satisfy MIT's notice requirement.
- **Branches:**
  - `master`: tracks upstream, no local commits.
  - `qin/main`: our changes. Release tags use `v<upstream>-qin.<n>`, for
    example `v0.1.7-qin.1`.
- **Syncing upstream:** `git fetch upstream && git checkout master && git merge --ff-only upstream/master`,
  then rebase or merge `qin/main` onto it.

## Local changes

| Commit | Change |
|---|---|
| `style: cargo fmt` | Formatting only; upstream had unformatted code. |
| `test: fix probe compile error ... skip non-UTF-8 name tests on APFS` | `tests/nonutf8_adversarial_probe.rs` did not compile on rustc 1.95. Tests that create non-UTF-8 file names now skip on filesystems that reject them (macOS APFS). Overlaps upstream PR #6. |

## Ranking work (`v0.1.7-qin.2`)

Borrowed from [MinishLab/semble](https://github.com/MinishLab/semble).
Measured with `tests/ranking_bench.rs` (55 queries over qin-code @ 430b840,
12 held out) and `scripts/lang_corpus_check.py` (gin, gson).

| Change | Build | Effect |
|---|---|---|
| Stemmed term matching, stopwords, query shape, BM25, definition and coherence boosts, stub/mock penalty | default | hit@1 63.6% -> 87.3%, zero-result 29.1% -> 0%, natural-language hit@1 0% -> 62.5%. Latency 1.0-1.25x. |
| Model2Vec re-ranking of the top 10 (RRF, weight 0.4) for natural-language subjects | `--features semantic` + `AGENTGREP_SEMANTIC_MODEL=<dir>` | hit@1 87.3% -> 90.9%, natural 62.5% -> 75%, holdout 75% -> 83.3%. +2.5 MB binary; model is never downloaded by agentgrep. |
| tree-sitter structure for Go, Java, C, C++, C#, Ruby, PHP, Kotlin, Swift | `--features treesitter` (all) or per grammar: `ts-go`, `ts-java`, `ts-c`, `ts-cpp`, `ts-csharp`, `ts-ruby`, `ts-php`, `ts-kotlin`, `ts-swift` | Go and Java hit@1 0% -> 85-95%. Binary 3.8 MB -> 24.5 MB. |

Invariants kept: semantic is a re-ranker only (never adds or drops a file);
files passing the original strict subject gate are always scored; rg parity
36/36 (synthetic + jcode corpora; the Linux corpus was not available).

### Running the benchmarks

```bash
AGENTGREP_BENCH_ROOT=/path/to/qin-code@430b840 \
  cargo test --release --test ranking_bench -- --ignored --nocapture
python3 scripts/lang_corpus_check.py target/release/agentgrep /path/to/gin go 40
python3 scripts/ranking_latency.py OLD_BIN NEW_BIN /path/to/corpus 5
```

## Not done yet

- Holdout natural-language queries still miss when the code uses different
  words than the query ("convert web page to markdown" vs `html_to_markdown`).
  A code-trained embedding model or query expansion would be the next step.
- tree-sitter is not used for Rust/TS/JS/Python; their line-based extractors
  already score well on the bench and switching would need its own measurement.

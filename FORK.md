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

## Planned work

Ranking improvements for `smart` mode, borrowed from
[MinishLab/semble](https://github.com/MinishLab/semble), in this order:

1. Identifier stemming, plus adaptive weighting for symbol-like vs. natural-language queries.
2. File-level boost when several regions in one file match.
3. BM25 (IDF-weighted) scoring in place of raw hit counts.
4. Optional semantic channel with Model2Vec static embeddings, fused by RRF, behind a feature gate.
5. tree-sitter chunking in place of the regex structure extractor.

Every step must keep the rg parity benchmark at 54/54 and add a ranking-quality benchmark.

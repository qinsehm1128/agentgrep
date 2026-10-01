//! Ranking-quality benchmark for `smart` mode.
//!
//! Runs a fixed query set against an external corpus and reports hit@1,
//! recall@5, MRR@5 and the zero-result rate, per query class and overall.
//!
//! The corpus is the qin-code repository pinned at a known commit, so the
//! expected answers are stable. It is not vendored here; point the benchmark
//! at a checkout with:
//!
//! ```text
//! AGENTGREP_BENCH_ROOT=/path/to/cea-jcode cargo test --release \
//!     --test ranking_bench -- --ignored --nocapture
//! ```
//!
//! Set `AGENTGREP_BENCH_OUT=path.json` to also write the metrics as JSON,
//! and `AGENTGREP_BENCH_VERBOSE=1` to print the top files for every query.
//!
//! The test never fails on quality numbers (they are a measurement, not a
//! gate); it only fails if the corpus is missing an expected file, which
//! means the corpus is at the wrong commit.

use agentgrep::cli::{FullRegionMode, SmartArgs};
use agentgrep::smart_dsl::parse_smart_query;
use agentgrep::smart_engine::run_smart;
use std::path::Path;
use std::time::Instant;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    /// Exact identifier as it appears in code: `active_backend`, `CognitionDb`.
    Symbol,
    /// Identifier split into words: `embedding backend`, `tokenize slots`.
    Split,
    /// Natural-language description that shares few literal words with the code.
    Natural,
}

struct Case {
    /// Holdout cases are never looked at while tuning; they only check that
    /// gains on the dev set generalize.
    holdout: bool,
    class: Class,
    subject: &'static str,
    relation: &'static str,
    /// Any of these files counts as a correct answer.
    expected: &'static [&'static str],
}

const fn case(
    class: Class,
    subject: &'static str,
    relation: &'static str,
    expected: &'static [&'static str],
) -> Case {
    Case {
        holdout: false,
        class,
        subject,
        relation,
        expected,
    }
}

const fn holdout(
    class: Class,
    subject: &'static str,
    relation: &'static str,
    expected: &'static [&'static str],
) -> Case {
    Case {
        holdout: true,
        class,
        subject,
        relation,
        expected,
    }
}

const EMB_BACKEND: &str = "crates/jcode-base/src/embedding_backend.rs";
const EMBEDDING: &str = "crates/jcode-base/src/embedding.rs";
const EMB_CRATE: &str = "crates/jcode-embedding/src/lib.rs";
const COG_STORE: &str = "crates/qin-code-cognition/src/store.rs";
const COG_OPS: &str = "crates/jcode-app-core/src/cognition_ops.rs";
const SS_INDEX: &str = "crates/jcode-app-core/src/tool/session_search_index.rs";
const RECONCILE: &str = "crates/qin-code-knowledge/src/reconcile.rs";
const MEMORY: &str = "crates/jcode-base/src/memory.rs";
const MEMORY_AGENT: &str = "crates/jcode-base/src/memory_agent.rs";
const COMPACTION: &str = "crates/jcode-base/src/compaction.rs";
const COMPACTION_CORE: &str = "crates/jcode-compaction-core/src/lib.rs";
const LOSSLESS: &str = "crates/jcode-compaction-core/src/lossless.rs";
const PERSISTENCE: &str = "crates/jcode-base/src/session/persistence.rs";
const AG_ARGS: &str = "crates/jcode-app-core/src/tool/agentgrep/args.rs";
const CONFIG_TYPES: &str = "crates/jcode-config-types/src/lib.rs";

const WS_HTML: &str = "crates/qin-code-websearch/src/html.rs";
const KEYBIND: &str = "crates/jcode-config-types/src/keybindings.rs";
const UPDATE: &str = "crates/jcode-app-core/src/update.rs";
const SECRETS: &str = "crates/jcode-app-core/src/tool/discover_secrets.rs";
const BASH_GATE: &str = "crates/jcode-app-core/src/tool/bash_destructive_gate.rs";

use Class::{Natural, Split, Symbol};

const CASES: &[Case] = &[
    // --- Symbol: the exact identifier.
    case(Symbol, "active_backend", "defined", &[EMB_BACKEND]),
    case(Symbol, "CognitionDb", "defined", &[COG_STORE]),
    case(Symbol, "cognition_compile_in", "defined", &[COG_OPS]),
    case(Symbol, "embed_query_active", "defined", &[EMB_BACKEND]),
    case(Symbol, "tokenize_slots_parallel", "defined", &[SS_INDEX]),
    case(Symbol, "MemoryAgent", "defined", &[MEMORY_AGENT]),
    case(Symbol, "CompactionMode", "defined", &[CONFIG_TYPES]),
    case(Symbol, "reconcile_candidates", "defined", &[RECONCILE]),
    case(Symbol, "mean_embedding", "defined", &[COMPACTION_CORE]),
    case(Symbol, "OpenAiEmbeddingBackend", "defined", &[EMB_BACKEND]),
    case(Symbol, "semantic_cutoff", "defined", &[COMPACTION]),
    case(Symbol, "find_similar_scoped", "defined", &[MEMORY]),
    case(
        Symbol,
        "durable_vectors_are_provably_empty",
        "defined",
        &[PERSISTENCE],
    ),
    case(Symbol, "lossless_fold_text", "defined", &[LOSSLESS]),
    case(Symbol, "build_smart_args_and_query", "defined", &[AG_ARGS]),
    case(Symbol, "vec_search", "defined", &[COG_STORE]),
    case(
        Symbol,
        "openai_backend_from_config",
        "defined",
        &[EMB_BACKEND],
    ),
    // --- Split: identifier words separated by spaces.
    case(Split, "active backend", "defined", &[EMB_BACKEND]),
    case(Split, "cognition compile", "implementation", &[COG_OPS]),
    case(Split, "embed query active", "defined", &[EMB_BACKEND]),
    case(Split, "tokenize slots", "implementation", &[SS_INDEX]),
    case(Split, "memory agent", "defined", &[MEMORY_AGENT]),
    case(Split, "compaction mode", "defined", &[CONFIG_TYPES]),
    case(Split, "reconcile candidates", "defined", &[RECONCILE]),
    case(Split, "mean embedding", "defined", &[COMPACTION_CORE]),
    case(Split, "semantic cutoff", "implementation", &[COMPACTION]),
    case(Split, "find similar scoped", "defined", &[MEMORY]),
    case(Split, "lossless fold", "implementation", &[LOSSLESS]),
    case(Split, "unload idle", "implementation", &[EMBEDDING]),
    case(
        Split,
        "cosine similarity",
        "implementation",
        &[EMBEDDING, EMB_CRATE],
    ),
    case(Split, "lexical search", "implementation", &[COG_STORE]),
    // --- Natural: describes behaviour; few literal identifier words.
    case(
        Natural,
        "download embedding model",
        "implementation",
        &[EMB_CRATE],
    ),
    case(
        Natural,
        "topic shift detection",
        "implementation",
        &[COMPACTION, MEMORY_AGENT],
    ),
    case(
        Natural,
        "unload embedder when idle",
        "implementation",
        &[EMBEDDING],
    ),
    case(
        Natural,
        "remote embeddings api key",
        "implementation",
        &[EMB_BACKEND],
    ),
    case(
        Natural,
        "bloom filter for session search",
        "implementation",
        &[SS_INDEX],
    ),
    case(
        Natural,
        "brute force cosine over vectors",
        "implementation",
        &[COG_STORE, MEMORY],
    ),
    case(
        Natural,
        "reciprocal rank fusion of bm25 and dense",
        "implementation",
        &[MEMORY],
    ),
    case(
        Natural,
        "rebuild compiled cognition database",
        "implementation",
        &[COG_STORE, COG_OPS],
    ),
    case(
        Natural,
        "cross encoder rerank",
        "implementation",
        &[EMB_CRATE],
    ),
    case(
        Natural,
        "batch embed passages remote api",
        "implementation",
        &[EMB_BACKEND],
    ),
    case(
        Natural,
        "parse knowledge entry and score duplicate",
        "implementation",
        &[RECONCILE],
    ),
    case(
        Natural,
        "embedding cache capacity",
        "implementation",
        &[EMBEDDING],
    ),
    // --- Holdout: different areas of the codebase, not used for tuning.
    holdout(Symbol, "html_to_markdown", "defined", &[WS_HTML]),
    holdout(
        Symbol,
        "validate_keybinding_defaults",
        "defined",
        &[KEYBIND],
    ),
    holdout(
        Symbol,
        "fetch_latest_release_blocking",
        "defined",
        &[UPDATE],
    ),
    holdout(Symbol, "looks_like_jwt", "defined", &[SECRETS]),
    holdout(Split, "html to markdown", "implementation", &[WS_HTML]),
    holdout(Split, "verify asset checksum", "implementation", &[UPDATE]),
    holdout(Split, "keybinding defaults report", "defined", &[KEYBIND]),
    holdout(Split, "contains bearer token", "implementation", &[SECRETS]),
    holdout(
        Natural,
        "convert web page to markdown",
        "implementation",
        &[WS_HTML],
    ),
    holdout(
        Natural,
        "detect leaked credentials and payment cards",
        "implementation",
        &[SECRETS],
    ),
    holdout(
        Natural,
        "check github for newer release",
        "implementation",
        &[UPDATE],
    ),
    holdout(
        Natural,
        "block destructive shell commands",
        "implementation",
        &[BASH_GATE],
    ),
];

#[derive(Default, Clone, Copy)]
struct Tally {
    n: usize,
    hit1: usize,
    hit5: usize,
    rr: f64,
    zero: usize,
}

impl Tally {
    fn add(&mut self, rank: Option<usize>, zero: bool) {
        self.n += 1;
        if zero {
            self.zero += 1;
        }
        if let Some(r) = rank {
            if r == 1 {
                self.hit1 += 1;
            }
            if r <= 5 {
                self.hit5 += 1;
                self.rr += 1.0 / r as f64;
            }
        }
    }
    fn pct(x: usize, n: usize) -> f64 {
        if n == 0 {
            0.0
        } else {
            100.0 * x as f64 / n as f64
        }
    }
    fn json(&self) -> String {
        format!(
            "{{\"n\":{},\"hit_at_1\":{:.1},\"recall_at_5\":{:.1},\"mrr_at_5\":{:.3},\"zero_result\":{:.1}}}",
            self.n,
            Self::pct(self.hit1, self.n),
            Self::pct(self.hit5, self.n),
            if self.n == 0 {
                0.0
            } else {
                self.rr / self.n as f64
            },
            Self::pct(self.zero, self.n),
        )
    }
    fn line(&self, name: &str) -> String {
        format!(
            "{name:<8} n={:>2}  hit@1={:>5.1}%  recall@5={:>5.1}%  MRR@5={:.3}  zero={:>5.1}%",
            self.n,
            Self::pct(self.hit1, self.n),
            Self::pct(self.hit5, self.n),
            if self.n == 0 {
                0.0
            } else {
                self.rr / self.n as f64
            },
            Self::pct(self.zero, self.n),
        )
    }
}

fn smart_args(terms: Vec<String>) -> SmartArgs {
    SmartArgs {
        terms,
        json: true,
        max_files: 5,
        max_regions: 6,
        full_region: FullRegionMode::Auto,
        debug_plan: false,
        debug_score: false,
        paths_only: false,
        path: None,
        file_type: None,
        glob: None,
        hidden: false,
        no_ignore: false,
        context_json: None,
    }
}

#[test]
#[ignore = "needs AGENTGREP_BENCH_ROOT pointing at the pinned corpus"]
fn ranking_quality() {
    let Ok(root) = std::env::var("AGENTGREP_BENCH_ROOT") else {
        eprintln!("AGENTGREP_BENCH_ROOT not set; skipping");
        return;
    };
    let root = Path::new(&root);
    let verbose = std::env::var_os("AGENTGREP_BENCH_VERBOSE").is_some();

    for c in CASES {
        for e in c.expected {
            assert!(
                root.join(e).is_file(),
                "corpus is missing {e}; wrong commit for AGENTGREP_BENCH_ROOT?"
            );
        }
    }

    let mut by_class = [Tally::default(); 3];
    let mut all = Tally::default();
    let mut dev = Tally::default();
    let mut hold = Tally::default();
    let mut misses = Vec::new();
    let started = Instant::now();

    for c in CASES {
        let terms = vec![
            format!("subject:{}", c.subject),
            format!("relation:{}", c.relation),
        ];
        let query = parse_smart_query(&terms).expect("valid query");
        let result = run_smart(root, &query, &smart_args(terms)).expect("smart run");
        let paths: Vec<&str> = result.files.iter().map(|f| f.path.as_str()).collect();
        let rank = paths
            .iter()
            .position(|p| c.expected.contains(p))
            .map(|i| i + 1);
        let zero = paths.is_empty();
        all.add(rank, zero);
        if c.holdout {
            hold.add(rank, zero);
        } else {
            dev.add(rank, zero);
        }
        by_class[c.class as usize].add(rank, zero);
        if rank != Some(1) {
            misses.push(format!(
                "  [{:?}{}] {:<45} rank={} top={}",
                c.class,
                if c.holdout { " holdout" } else { "" },
                c.subject,
                rank.map_or("-".to_string(), |r| r.to_string()),
                paths.first().copied().unwrap_or("<none>")
            ));
        }
        if verbose {
            eprintln!("{:?} {:?} -> {:?}", c.class, c.subject, &paths);
        }
    }
    let elapsed = started.elapsed();

    eprintln!(
        "\nagentgrep smart ranking benchmark ({} queries, {:.2?})",
        CASES.len(),
        elapsed
    );
    eprintln!("{}", by_class[Symbol as usize].line("symbol"));
    eprintln!("{}", by_class[Split as usize].line("split"));
    eprintln!("{}", by_class[Natural as usize].line("natural"));
    eprintln!("{}", dev.line("dev"));
    eprintln!("{}", hold.line("holdout"));
    eprintln!("{}", all.line("overall"));
    if !misses.is_empty() {
        eprintln!("\nnot ranked first:");
        for m in &misses {
            eprintln!("{m}");
        }
    }

    if let Ok(out) = std::env::var("AGENTGREP_BENCH_OUT") {
        let json = format!(
            "{{\"queries\":{},\"elapsed_ms\":{},\"symbol\":{},\"split\":{},\"natural\":{},\"dev\":{},\"holdout\":{},\"overall\":{}}}\n",
            CASES.len(),
            elapsed.as_millis(),
            by_class[Symbol as usize].json(),
            by_class[Split as usize].json(),
            by_class[Natural as usize].json(),
            dev.json(),
            hold.json(),
            all.json(),
        );
        std::fs::write(&out, json).expect("write bench output");
    }
}

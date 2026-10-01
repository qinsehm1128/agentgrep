//! Lexical ranking primitives for `smart` mode: query term extraction,
//! light stemming, query-shape detection, and BM25.
//!
//! Matching is substring-based on lowercased text, so a stem such as
//! `embed` matches `embedder`, `embedding` and `embed_query` alike. That is
//! what makes natural-language words line up with identifiers without
//! tokenizing every file.

use crate::smart_engine::normalize_match_text;

/// One query term: the word as written and the stem used for matching.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Term {
    pub word: String,
    pub stem: String,
}

/// Shape of the subject, used to balance exact-identifier signals against
/// term-coverage signals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryShape {
    /// Looks like an identifier: `active_backend`, `CognitionDb`, `a::b`.
    Symbol,
    /// Plain words: `topic shift detection`.
    Natural,
}

const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from", "how", "in",
    "into", "is", "it", "its", "of", "on", "or", "over", "that", "the", "this", "to", "via",
    "what", "when", "where", "which", "with",
];

/// Minimum stem length. Shorter stems match too much as substrings.
const MIN_STEM: usize = 4;

pub fn query_shape(subject: &str) -> QueryShape {
    let trimmed = subject.trim();
    if trimmed.contains(char::is_whitespace) {
        return QueryShape::Natural;
    }
    let has_separator = trimmed.contains('_') || trimmed.contains("::") || trimmed.contains('.');
    let mut prev_lower = false;
    let mut camel = false;
    for ch in trimmed.chars() {
        if ch.is_ascii_uppercase() && prev_lower {
            camel = true;
        }
        prev_lower = ch.is_ascii_lowercase();
    }
    if has_separator || camel {
        QueryShape::Symbol
    } else {
        // A single plain word behaves like a natural-language query of one term.
        QueryShape::Natural
    }
}

/// Extract de-duplicated, stemmed, stopword-free terms from a subject.
pub fn query_terms(subject: &str) -> Vec<Term> {
    let mut out: Vec<Term> = Vec::new();
    for word in normalize_match_text(subject).split_whitespace() {
        if STOPWORDS.contains(&word) {
            continue;
        }
        let stem = stem(word);
        if out.iter().any(|t| t.stem == stem) {
            continue;
        }
        out.push(Term {
            word: word.to_string(),
            stem,
        });
    }
    out
}

/// A light suffix-stripping stemmer.
///
/// The stem is used as a substring needle against raw text, so it must be a
/// prefix of every inflection it stands for. Rules therefore only strip
/// letters, never add them: `policies` -> `polic` matches both `policies` and
/// `policy`, whereas `policy` would miss `policies`.
pub fn stem(word: &str) -> String {
    let w = word.to_ascii_lowercase();
    if w.len() <= MIN_STEM || !w.bytes().all(|b| b.is_ascii_alphabetic()) {
        return w;
    }
    // Longest suffix first.
    const SUFFIXES: &[&str] = &[
        "ational", "ization", "ations", "ation", "ingly", "ments", "ment", "ness", "ings", "ing",
        "ions", "ion", "ies", "ers", "er", "ed", "es", "ly", "s",
    ];
    for suffix in SUFFIXES {
        let Some(base) = w.strip_suffix(suffix) else {
            continue;
        };
        // Words ending in "ss"/"us"/"is" ("class", "status") keep their "s".
        if *suffix == "s" && (base.ends_with('s') || base.ends_with('u') || base.ends_with('i')) {
            continue;
        }
        // "ion" only after t/s: detection -> detect, compression -> compress.
        if (*suffix == "ion" || *suffix == "ions") && !(base.ends_with('t') || base.ends_with('s'))
        {
            continue;
        }
        let stemmed = match *suffix {
            // "ies" -> drop "ies" and keep the shared prefix ("polic").
            // "ational"/"ization"/"ation(s)": keep the "at"/"iz" so the stem
            // stays a prefix of the base verb ("normalization" -> "normaliz").
            "ational" | "ations" | "ation" => format!("{base}at"),
            "ization" => format!("{base}iz"),
            _ => base.to_string(),
        };
        let stemmed = collapse_double_consonant(stemmed, &w);
        if stemmed.len() >= MIN_STEM {
            return stemmed;
        }
        // Too short ("files" -> "fil" via "es"): try a shorter suffix.
    }
    w
}

/// "embedd" -> "embed", "stopp" -> "stop". Only when the shorter form is
/// still a prefix of the original word, which it always is.
fn collapse_double_consonant(mut s: String, original: &str) -> String {
    let bytes = s.as_bytes();
    let n = bytes.len();
    if n >= 2 {
        let last = bytes[n - 1];
        if last == bytes[n - 2] && !b"aeiouls".contains(&last) {
            s.pop();
        }
    }
    debug_assert!(original.starts_with(&s));
    s
}

/// Count non-overlapping occurrences of `needle` in `haystack`.
pub fn count_occurrences(haystack: &str, needle: &str) -> u32 {
    if needle.is_empty() {
        return 0;
    }
    haystack.matches(needle).count() as u32
}

/// Precompiled SIMD substring counters, one per query term.
pub struct TermCounter {
    finders: Vec<memchr::memmem::Finder<'static>>,
}

impl TermCounter {
    pub fn new(terms: &[Term]) -> Self {
        Self {
            finders: terms
                .iter()
                .map(|t| memchr::memmem::Finder::new(t.stem.as_bytes()).into_owned())
                .collect(),
        }
    }

    /// Per-term occurrence counts in `haystack`.
    pub fn counts(&self, haystack: &str) -> Vec<u32> {
        self.finders
            .iter()
            .map(|f| {
                if f.needle().is_empty() {
                    0
                } else {
                    f.find_iter(haystack.as_bytes()).count() as u32
                }
            })
            .collect()
    }
}

/// Okapi BM25 over whole files, with substring term frequencies.
pub struct Bm25 {
    idf: Vec<f64>,
    avg_len: f64,
}

const K1: f64 = 1.2;
const B: f64 = 0.75;

impl Bm25 {
    /// `doc_freq[i]` = number of files containing term `i`; `total_docs` =
    /// number of files scanned; `total_len` = sum of their lengths.
    pub fn new(doc_freq: &[u32], total_docs: usize, total_len: u64) -> Self {
        let n = total_docs.max(1) as f64;
        let idf = doc_freq
            .iter()
            .map(|&df| {
                let df = df as f64;
                (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
            })
            .collect();
        Self {
            idf,
            avg_len: (total_len as f64 / n).max(1.0),
        }
    }

    pub fn idf(&self, term: usize) -> f64 {
        self.idf[term]
    }

    pub fn total_idf(&self) -> f64 {
        self.idf.iter().sum()
    }

    pub fn score(&self, tf: &[u32], doc_len: usize) -> f64 {
        let norm = K1 * (1.0 - B + B * doc_len as f64 / self.avg_len);
        tf.iter()
            .zip(&self.idf)
            .map(|(&tf, idf)| {
                let tf = tf as f64;
                idf * tf * (K1 + 1.0) / (tf + norm)
            })
            .sum()
    }
}

/// Paths that usually hold non-canonical copies of real code.
pub fn is_noise_path(relative_lower: &str) -> bool {
    const SEGMENTS: &[&str] = &[
        "stub", "mock", "fake", "fixture", "legacy", "compat", "example", "vendor", "testdata",
    ];
    relative_lower.ends_with(".d.ts")
        || relative_lower
            .split(['/', '\\', '_', '-', '.'])
            .any(|part| {
                SEGMENTS
                    .iter()
                    .any(|s| part == *s || part == format!("{s}s"))
            })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stems_share_prefix_with_identifier_forms() {
        assert_eq!(stem("embedding"), "embed");
        assert_eq!(stem("embedder"), "embed");
        assert_eq!(stem("embeddings"), "embed");
        assert_eq!(stem("detection"), "detect");
        assert_eq!(stem("policies"), "polic");
        assert_eq!(stem("entries"), "entr");
        assert_eq!(stem("normalization"), "normaliz");
        assert_eq!(stem("compression"), "compress");
        assert_eq!(stem("vectors"), "vector");
        assert_eq!(stem("commands"), "command");
        assert_eq!(stem("passages"), "passag");
        assert_eq!(stem("leaked"), "leak");
        assert_eq!(stem("filter"), "filt");
    }

    #[test]
    fn short_plurals_reduce_to_singular_prefix() {
        for (plural, singular) in [
            ("files", "file"),
            ("names", "name"),
            ("rules", "rule"),
            ("types", "type"),
            ("pages", "page"),
            ("nodes", "node"),
            ("caches", "cache"),
            ("entries", "entry"),
            ("policies", "policy"),
            ("queries", "query"),
        ] {
            let st = stem(plural);
            assert!(
                singular.starts_with(&st),
                "{plural} -> {st} does not prefix {singular}"
            );
        }
    }

    #[test]
    fn stem_is_always_a_prefix_of_the_word() {
        for word in [
            "policies",
            "entries",
            "queries",
            "embedding",
            "embedder",
            "detection",
            "normalization",
            "configurations",
            "relational",
            "stopped",
            "running",
            "matches",
            "statuses",
            "files",
        ] {
            let st = stem(word);
            assert!(word.starts_with(&st), "{word} -> {st}");
        }
    }

    #[test]
    fn short_and_protected_words_are_kept() {
        assert_eq!(stem("api"), "api");
        assert_eq!(stem("key"), "key");
        assert_eq!(stem("idle"), "idle");
        assert_eq!(stem("class"), "class");
        assert_eq!(stem("status"), "status");
        assert_eq!(stem("fusion"), "fusion");
        assert_eq!(stem("newer"), "newer");
        assert_eq!(stem("bm25"), "bm25");
    }

    #[test]
    fn query_terms_drop_stopwords_and_duplicates() {
        let terms: Vec<_> = query_terms("unload the embedder when embedding is idle")
            .into_iter()
            .map(|t| t.stem)
            .collect();
        assert_eq!(terms, vec!["unload", "embed", "idle"]);
    }

    #[test]
    fn query_terms_split_identifiers() {
        let terms: Vec<_> = query_terms("parseConfig")
            .into_iter()
            .map(|t| t.word)
            .collect();
        assert_eq!(terms, vec!["parse", "config"]);
    }

    #[test]
    fn shape_detection() {
        assert_eq!(query_shape("active_backend"), QueryShape::Symbol);
        assert_eq!(query_shape("CognitionDb"), QueryShape::Symbol);
        assert_eq!(query_shape("a::b"), QueryShape::Symbol);
        assert_eq!(query_shape("topic shift detection"), QueryShape::Natural);
        assert_eq!(query_shape("bloom"), QueryShape::Natural);
    }

    #[test]
    fn bm25_prefers_rare_terms_and_shorter_docs() {
        let bm = Bm25::new(&[1, 100], 100, 100 * 1000);
        assert!(bm.idf(0) > bm.idf(1));
        assert!(bm.score(&[1, 0], 500) > bm.score(&[1, 0], 5000));
        assert!(bm.score(&[1, 0], 1000) > bm.score(&[0, 1], 1000));
    }

    #[test]
    fn term_counter_matches_naive_count() {
        let terms = query_terms("embedding backend");
        let counter = TermCounter::new(&terms);
        let text = "embed_query embedding backend; backends embedder";
        let naive: Vec<u32> = terms
            .iter()
            .map(|t| count_occurrences(text, &t.stem))
            .collect();
        assert_eq!(counter.counts(text), naive);
        assert_eq!(naive, vec![3, 2]);
    }

    #[test]
    fn noise_paths() {
        assert!(is_noise_path("crates/x/src/embedding_stub.rs"));
        assert!(is_noise_path("src/mocks/client.ts"));
        assert!(is_noise_path("types/index.d.ts"));
        assert!(is_noise_path("examples/demo.rs"));
        assert!(!is_noise_path("crates/x/src/embedding.rs"));
        assert!(!is_noise_path("src/stubborn.rs"));
    }
}

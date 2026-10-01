use crate::cli::{FullRegionMode, SmartArgs};
use crate::context::{Familiarity, HarnessContext};
use crate::rank::{Bm25, QueryShape, Term, TermCounter, is_noise_path, query_shape, query_terms};
use crate::smart_dsl::{Relation, SmartQuery};
use crate::structure::{StructureItem, extract_file_structure, infer_role};
use crate::workspace::{SearchScope, TextFile, collect_file_entries, read_text_file};
use serde::Serialize;
use std::path::Path;

const MAX_REGION_BODY_LINE_CHARS: usize = 240;

#[derive(Debug, Clone, Serialize)]
pub struct SmartResult {
    pub query: SmartQuery,
    pub root: String,
    pub summary: SmartSummary,
    pub files: Vec<SmartFile>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SmartSummary {
    pub total_files: usize,
    pub total_regions: usize,
    pub best_file: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SmartFile {
    pub path: String,
    /// Hex of the raw relative path bytes when the name is not valid UTF-8,
    /// so JSON consumers can address the real file. Omitted otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path_bytes: Option<String>,
    pub role: String,
    pub language: String,
    pub score: i32,
    pub why: Vec<String>,
    pub structure: SmartStructure,
    pub regions: Vec<SmartRegion>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_applied: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SmartStructure {
    pub items: Vec<StructureItem>,
    pub omitted_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct SmartRegion {
    pub kind: String,
    pub label: String,
    pub start_line: usize,
    pub end_line: usize,
    pub line_count: usize,
    pub score: i32,
    pub body: String,
    pub full_region: bool,
    pub why: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_applied: Option<String>,
}

pub fn run_smart(root: &Path, query: &SmartQuery, args: &SmartArgs) -> Result<SmartResult, String> {
    let scope = SearchScope {
        root,
        file_type: args.file_type.as_deref(),
        glob: args.glob.as_deref(),
        hidden: args.hidden,
        no_ignore: args.no_ignore,
        follow: true,
    };

    let relation_terms = relation_terms(&query.relation);
    let subject_lower = query.subject.to_ascii_lowercase();
    let subject_tokens = tokenize_subject(&query.subject);
    let support_terms = query
        .support
        .iter()
        .map(|s| s.to_ascii_lowercase())
        .collect::<Vec<_>>();
    let path_hint = query.path_hint.as_ref().map(|s| s.to_ascii_lowercase());
    let context = HarnessContext::load(args.context_json.as_deref())?;

    let shape = query_shape(&query.subject);
    let terms = query_terms(&query.subject);
    let soft_required = soft_required_terms(terms.len());
    let counter = TermCounter::new(&terms);

    // Pass 1: walk every file once, collect BM25 statistics, and keep the
    // candidates. A file is a candidate if it passes the original strict
    // subject gate, or (for multi-term subjects) contains enough of the
    // stemmed query terms.
    let mut doc_freq = vec![0u32; terms.len()];
    let mut total_docs = 0usize;
    let mut total_len = 0u64;
    let mut pending: Vec<Pending> = Vec::new();
    for entry in collect_file_entries(&scope) {
        let relative_lower = entry.relative_path.to_ascii_lowercase();
        if let Some(path_hint) = &path_hint
            && !relative_lower.contains(path_hint)
        {
            continue;
        }

        let inferred_role = infer_role(&entry.relative_path);
        if should_filter_kind(query.kind.as_deref(), &inferred_role) {
            continue;
        }

        let Some(text) = read_text_file(&entry.path) else {
            continue;
        };
        let file = TextFile {
            path: entry.path,
            relative_path: entry.relative_path,
            relative_raw: entry.relative_raw,
            text,
        };
        let text_lower = file.text.to_ascii_lowercase();
        let tf = term_frequencies(&counter, &relative_lower, &text_lower);
        total_docs += 1;
        total_len += file.text.len() as u64;
        let mut matched_terms = 0;
        for (df, &count) in doc_freq.iter_mut().zip(&tf) {
            if count > 0 {
                *df += 1;
                matched_terms += 1;
            }
        }
        let strict = file_may_contain_subject(
            &relative_lower,
            &text_lower,
            &query.subject,
            &subject_lower,
            &subject_tokens,
        );
        let lexical = terms.len() >= 2 && matched_terms >= soft_required;
        if !strict && !lexical {
            continue;
        }
        // Keep only what ranking needs; text is re-read in pass 2, so memory
        // stays at one file at a time no matter how many files match.
        let len = file.text.len();
        pending.push(Pending {
            path: file.path,
            relative_path: file.relative_path,
            relative_raw: file.relative_raw,
            strict,
            len,
            tf,
            bm25: 0.0,
        });
    }

    let bm25 = Bm25::new(&doc_freq, total_docs, total_len);
    for entry in &mut pending {
        entry.bm25 = bm25.score(&entry.tf, entry.len);
    }
    // Strict candidates are always scored, as before. Lexical-only
    // candidates are capped to the best by BM25 so broad queries stay fast.
    let (mut candidates, mut lexical_only): (Vec<Pending>, Vec<Pending>) =
        pending.into_iter().partition(|p| p.strict);
    lexical_only.sort_by(|a, b| {
        b.bm25
            .total_cmp(&a.bm25)
            .then_with(|| a.relative_path.cmp(&b.relative_path))
    });
    lexical_only.truncate(MAX_LEXICAL_CANDIDATES);
    candidates.append(&mut lexical_only);
    let max_bm25 = candidates.iter().map(|c| c.bm25).fold(0.0, f64::max);
    let region_lex = RegionLexicon {
        terms: &terms,
        idf: (0..terms.len()).map(|i| bm25.idf(i)).collect(),
        total_idf: bm25.total_idf(),
        required: soft_required,
        shape,
    };

    // Pass 2: structural scoring (unchanged) plus lexical signals.
    let mut files = Vec::new();
    for candidate in candidates {
        let file_bm25 = candidate.bm25;
        let Some(text) = read_text_file(&candidate.path) else {
            continue;
        };
        let file = TextFile {
            path: candidate.path,
            relative_path: candidate.relative_path,
            relative_raw: candidate.relative_raw,
            text,
        };
        let relative_lower = file.relative_path.to_ascii_lowercase();
        let text_lower = file.text.to_ascii_lowercase();
        let structure = extract_file_structure(&file.path, &file.relative_path, &file.text);
        let lower_lines = collect_lower_lines(&file.text);
        let subject_mentions = count_lines(&lower_lines, &subject_lower);

        let relation_hits = relation_terms
            .iter()
            .filter(|term| {
                relative_lower.contains(term.as_str())
                    || structure
                        .items
                        .iter()
                        .any(|item| item.label.to_ascii_lowercase().contains(term.as_str()))
                    || text_lower.contains(term.as_str())
            })
            .count();

        let support_hits = support_terms
            .iter()
            .filter(|term| text_lower.contains(term.as_str()))
            .count();

        let mut file_score = 0;
        let mut why = vec!["exact subject match or symbol hit".to_string()];
        file_score += (subject_mentions as i32) * 5;
        if exact_subject_path_match(&relative_lower, &subject_lower) {
            file_score += match query.relation {
                Relation::Defined | Relation::Implementation => 140,
                _ => 60,
            };
            why.push("path matches subject variant".to_string());
        }
        if relation_hits > 0 {
            file_score += (relation_hits as i32) * 20;
            why.push(format!("relation-context hits: {relation_hits}"));
        }
        if support_hits > 0 {
            file_score += (support_hits as i32) * 10;
            why.push(format!("support-term hits: {support_hits}"));
        }
        if role_aligns(&structure.role, &query.relation) {
            file_score += 20;
            why.push(format!("role aligned: {}", structure.role));
        }
        match structure.role.as_str() {
            "implementation" | "auth" | "provider" | "ui" | "handler" => {
                file_score += 25;
                why.push(format!("code role boost: {}", structure.role));
            }
            "docs" => {
                file_score -= 50;
                why.push("docs penalty".to_string());
            }
            "test" => {
                file_score -= 20;
                why.push("test penalty".to_string());
            }
            _ => {}
        }
        if let Some(path_hint) = &path_hint {
            file_score += 30;
            why.push(format!("path hint matched: {path_hint}"));
        }

        let noise_path = is_noise_path(&relative_lower);
        if max_bm25 > 0.0 {
            let weight = match shape {
                QueryShape::Symbol => LEXICAL_FILE_WEIGHT_SYMBOL,
                QueryShape::Natural => LEXICAL_FILE_WEIGHT_NATURAL,
            };
            let mut lexical = file_bm25 / max_bm25 * weight;
            if noise_path {
                // Stubs and mocks repeat the real names; do not let that
                // repetition buy them relevance.
                lexical *= 0.5;
            }
            let lexical = lexical.round() as i32;
            if lexical > 0 {
                file_score += lexical;
                why.push(format!("bm25 term relevance: +{lexical}"));
            }
        }
        if noise_path {
            file_score -= NOISE_PATH_PENALTY;
            why.push("stub/mock/legacy path penalty".to_string());
        }

        let mut regions = build_regions(
            &file,
            &structure.items,
            &lower_lines,
            &subject_lower,
            &subject_tokens,
            &query.relation,
            args,
            context.as_ref(),
            &region_lex,
        );
        if regions.is_empty() {
            continue;
        }
        if matches!(query.relation, Relation::Defined | Relation::Implementation)
            && regions
                .iter()
                .any(|r| r.why.iter().any(|w| w == "exact subject label match"))
        {
            file_score += DEFINITION_FILE_BONUS;
            why.push("defines the subject".to_string());
        }
        let extra_regions = regions.len().saturating_sub(1).min(COHERENCE_MAX_REGIONS) as i32;
        if extra_regions > 0 {
            file_score += extra_regions * COHERENCE_PER_REGION;
            why.push(format!("{} matching regions in file", regions.len()));
        }
        regions.sort_by(|a, b| {
            b.score
                .cmp(&a.score)
                .then_with(|| a.start_line.cmp(&b.start_line))
        });
        let best_region_score = regions.first().map(|r| r.score).unwrap_or(0);
        if let Some(best_region) = regions.first() {
            if best_region
                .why
                .iter()
                .any(|reason| reason == "test/example penalty")
            {
                file_score -= 40;
                why.push("best region is test/example-like".to_string());
            }
            if best_region
                .why
                .iter()
                .any(|reason| reason == "cli/example penalty")
            {
                file_score -= 50;
                why.push("best region is cli/example-like".to_string());
            }
        }
        file_score += best_region_score / 2;
        why.push(format!("best region score: {best_region_score}"));
        regions.truncate(args.max_regions);

        let file_familiarity = context
            .as_ref()
            .map(|ctx| ctx.file_familiarity(&file.display_path()))
            .unwrap_or_default();
        let structure_budget = structure_budget_for_file(file_familiarity);
        let shown_items = select_structure_items(&structure.items, &regions, structure_budget);
        let omitted_count = structure.items.len().saturating_sub(shown_items.len());
        let context_applied = context_note_for_file(file_familiarity);

        files.push(SmartFile {
            path: file.display_path(),
            path_bytes: file.path_bytes_hex(),
            role: structure.role.clone(),
            language: structure.language.clone(),
            score: file_score,
            why,
            structure: SmartStructure {
                items: shown_items,
                omitted_count,
            },
            regions,
            context_applied,
        });
    }

    // `path` is unique even for lossy-colliding non-UTF-8 names because
    // display_path() appends a byte-derived suffix, so this tie-break is
    // total and portable (collect_file_entries is also pre-sorted by native
    // path bytes).
    files.sort_by(|a, b| b.score.cmp(&a.score).then_with(|| a.path.cmp(&b.path)));
    files.truncate(args.max_files);
    if shape == QueryShape::Natural {
        // After truncation: semantics may reorder the returned files but never
        // change which files are returned.
        semantic_rerank(&query.subject, &mut files);
    }

    let total_regions = files.iter().map(|f| f.regions.len()).sum();
    let best_file = files.first().map(|f| f.path.clone());

    Ok(SmartResult {
        query: query.clone(),
        root: root.display().to_string(),
        summary: SmartSummary {
            total_files: files.len(),
            total_regions,
            best_file,
        },
        files,
    })
}

fn build_regions(
    file: &TextFile,
    items: &[StructureItem],
    lower_lines: &[String],
    subject_lower: &str,
    subject_tokens: &[String],
    relation: &Relation,
    args: &SmartArgs,
    context: Option<&HarnessContext>,
    lex: &RegionLexicon<'_>,
) -> Vec<SmartRegion> {
    let relation_terms = relation_terms(relation);
    let lines = file.text.lines().collect::<Vec<_>>();

    let mut regions = Vec::new();
    for item in items {
        let start_idx = item.start_line.saturating_sub(1);
        let end_idx = item.end_line.min(lines.len());
        if start_idx >= end_idx {
            continue;
        }

        let region_lines = &lines[start_idx..end_idx];
        let region_lower = &lower_lines[start_idx..end_idx];

        let mut subject_line_hit_count = 0;
        let mut first_subject_hit = None;
        for (idx, line) in region_lower.iter().enumerate() {
            if line.contains(subject_lower) {
                subject_line_hit_count += 1;
                if first_subject_hit.is_none() {
                    first_subject_hit = Some(idx);
                }
            }
        }
        let item_label_lower = item.label.to_ascii_lowercase();
        let exact_label_match = exact_subject_label_match(&item.label, subject_lower);
        let token_label_match = subject_tokens_match_label(&item.label, subject_tokens);
        let coverage = lex.coverage(&item.label, region_lower);
        let lexical_match = lex.terms.len() >= 2 && coverage.terms >= lex.required;
        let subject_match = subject_line_hit_count > 0 || exact_label_match || token_label_match;
        if !subject_match && !lexical_match {
            continue;
        }

        let mut score = if subject_match {
            80 + (subject_line_hit_count as i32 * 10)
        } else {
            LEXICAL_REGION_BASE
        };
        let mut why = Vec::new();
        if subject_line_hit_count > 0 {
            why.push("exact subject match".to_string());
        }
        if coverage.terms > 0 && lex.total_idf > 0.0 {
            let weight = match lex.shape {
                QueryShape::Symbol => LEXICAL_REGION_WEIGHT_SYMBOL,
                QueryShape::Natural => LEXICAL_REGION_WEIGHT_NATURAL,
            };
            let lexical = (coverage.idf / lex.total_idf * weight).round() as i32
                + coverage.label_terms as i32 * LABEL_TERM_BONUS;
            score += lexical;
            why.push(format!(
                "term coverage {}/{} ({} in label): +{lexical}",
                coverage.terms,
                lex.terms.len(),
                coverage.label_terms
            ));
        }
        let relation_hit = relation_terms.iter().any(|term| {
            item_label_lower.contains(term.as_str())
                || region_lower.iter().any(|line| line.contains(term.as_str()))
        });
        if relation_hit {
            score += 30;
            why.push("relation-context aligned".to_string());
        }

        let owner_match = exact_label_match || token_label_match;
        let kind = classify_region(item, relation, owner_match);
        if exact_label_match {
            score += match relation {
                Relation::Defined | Relation::Implementation => 120,
                _ => 50,
            };
            why.push("exact subject label match".to_string());
        } else if token_label_match {
            score += match relation {
                Relation::Defined | Relation::Implementation => 90,
                _ => 35,
            };
            why.push("subject tokens match label".to_string());
        } else if matches!(relation, Relation::Defined | Relation::Implementation)
            && coverage.label_terms * 2 <= lex.terms.len()
        {
            score -= 50;
            why.push("non-owner penalty".to_string());
        }
        if owner_match && matches!(relation, Relation::Defined | Relation::Implementation) {
            if region_starts_with_pub(region_lines) {
                score += 20;
                why.push("public owner bonus".to_string());
            }
            let normalized_label = normalize_match_text(&item.label);
            if normalized_label.ends_with("tool") {
                score += 15;
                why.push("tool suffix bonus".to_string());
            }
            if normalized_label.ends_with("input") || normalized_label.ends_with("output") {
                score -= 10;
                why.push("auxiliary suffix penalty".to_string());
            }
        }
        match kind.as_str() {
            "render-site" | "definition" | "handler" | "assignment" => score += 20,
            _ => {}
        }

        if is_test_like(item, region_lines) {
            score -= 60;
            why.push("test/example penalty".to_string());
        }
        let representative_line = if let Some(first_match_idx) = first_subject_hit {
            region_lines[first_match_idx]
        } else {
            region_lines[0]
        };
        if looks_like_string_fixture(representative_line) {
            score -= 25;
            why.push("string-literal penalty".to_string());
        }
        if looks_like_cli_or_example_line(representative_line) {
            score -= 60;
            why.push("cli/example penalty".to_string());
        }

        let match_line_number = if let Some(first_match_idx) = first_subject_hit {
            item.start_line + first_match_idx
        } else {
            item.start_line
        };
        let familiarity = context
            .map(|ctx| {
                ctx.region_familiarity(
                    &file.display_path(),
                    &item.label,
                    item.start_line,
                    item.end_line,
                )
            })
            .unwrap_or_default();
        let full_region =
            should_include_full_region(item, args.full_region) && !should_prune_region(familiarity);
        let mut context_applied = None;
        let body = if full_region {
            extract_region(lines.as_slice(), item.start_line, item.end_line)
        } else if should_prune_region(familiarity) {
            context_applied = Some("compressed repeated region from harness context".to_string());
            region_lines[0].to_string()
        } else {
            lines[match_line_number - 1].to_string()
        };
        let body = compact_region_body(&body);

        regions.push(SmartRegion {
            kind,
            label: item.label.clone(),
            start_line: item.start_line,
            end_line: item.end_line,
            line_count: item.line_count,
            score,
            body,
            full_region,
            why,
            context_applied,
        });
    }

    regions
}

/// Most lexical-only candidates scored per query (by BM25).
const MAX_LEXICAL_CANDIDATES: usize = 40;
const LEXICAL_FILE_WEIGHT_SYMBOL: f64 = 40.0;
const LEXICAL_FILE_WEIGHT_NATURAL: f64 = 140.0;
const LEXICAL_REGION_BASE: i32 = 40;
const LEXICAL_REGION_WEIGHT_SYMBOL: f64 = 30.0;
const LEXICAL_REGION_WEIGHT_NATURAL: f64 = 100.0;
const LABEL_TERM_BONUS: i32 = 15;
const DEFINITION_FILE_BONUS: i32 = 50;
const COHERENCE_PER_REGION: i32 = 5;
const COHERENCE_MAX_REGIONS: usize = 3;
const NOISE_PATH_PENALTY: i32 = 40;
/// Path matches count more than body matches toward a term's frequency.
const PATH_TERM_WEIGHT: u32 = 2;

/// A file that passed pass 1, before the text is re-read for scoring.
struct Pending {
    path: std::path::PathBuf,
    relative_path: String,
    relative_raw: Option<Vec<u8>>,
    /// Passed the original whole-subject gate.
    strict: bool,
    len: usize,
    tf: Vec<u32>,
    bm25: f64,
}

/// How many of `n` query terms a lexical-only candidate must contain.
fn soft_required_terms(n: usize) -> usize {
    if n <= 2 { n } else { n.div_ceil(2) }
}

fn term_frequencies(counter: &TermCounter, relative_lower: &str, text_lower: &str) -> Vec<u32> {
    counter
        .counts(text_lower)
        .into_iter()
        .zip(counter.counts(relative_lower))
        .map(|(body, path)| body + PATH_TERM_WEIGHT * path)
        .collect()
}

struct RegionLexicon<'a> {
    terms: &'a [Term],
    idf: Vec<f64>,
    total_idf: f64,
    required: usize,
    shape: QueryShape,
}

struct Coverage {
    terms: usize,
    label_terms: usize,
    idf: f64,
}

impl RegionLexicon<'_> {
    fn coverage(&self, label: &str, region_lower: &[String]) -> Coverage {
        let label_norm = normalize_match_text(label);
        let mut out = Coverage {
            terms: 0,
            label_terms: 0,
            idf: 0.0,
        };
        for (term, idf) in self.terms.iter().zip(&self.idf) {
            let in_label = label_norm.contains(term.stem.as_str());
            if in_label
                || region_lower
                    .iter()
                    .any(|line| line.contains(term.stem.as_str()))
            {
                out.terms += 1;
                out.idf += idf;
            }
            if in_label {
                out.label_terms += 1;
            }
        }
        out
    }
}

/// How many top lexical files the semantic pass may reorder.
const SEMANTIC_RERANK_DEPTH: usize = 10;
/// Weight of the semantic rank in the fusion, relative to the lexical rank.
/// Kept below 1: at 0.7 and above the static model overrides good lexical
/// hits and the bench gets worse than with semantics off.
const SEMANTIC_RRF_WEIGHT: f64 = 0.4;

/// Fuse the lexical order with Model2Vec similarity by reciprocal rank.
///
/// Only reorders the files already found; never adds or drops one. No-op
/// unless the `semantic` feature is built and a model is configured.
fn semantic_rerank(subject: &str, files: &mut [SmartFile]) {
    // Length first: loading the model is the expensive part.
    if files.len() < 2 || !crate::semantic::available() {
        return;
    }
    let depth = files.len().min(SEMANTIC_RERANK_DEPTH);
    let head = &mut files[..depth];
    let docs: Vec<String> = head.iter().map(semantic_document).collect();
    let Some(sims) = crate::semantic::similarities(subject, &docs) else {
        return;
    };
    let sims: Vec<f64> = sims.into_iter().map(f64::from).collect();
    let sem_rank = crate::semantic::ranks_desc(&sims);
    let k = crate::semantic::RRF_K;
    let mut fused: Vec<(f64, usize)> = (0..depth)
        .map(|i| {
            let lexical = 1.0 / (k + (i + 1) as f64);
            let semantic = SEMANTIC_RRF_WEIGHT / (k + sem_rank[i] as f64);
            (lexical + semantic, i)
        })
        .collect();
    fused.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let order: Vec<usize> = fused.into_iter().map(|(_, i)| i).collect();
    let mut reordered: Vec<SmartFile> = order.iter().map(|&i| head[i].clone()).collect();
    for (i, file) in reordered.iter_mut().enumerate() {
        file.why.push(format!(
            "semantic rank {} (cos {:.3})",
            sem_rank[order[i]], sims[order[i]]
        ));
    }
    head.clone_from_slice(&reordered);
}

/// Text that represents a file to the embedding model: the label and body
/// of its best region. Whole-file text and label lists were measured and
/// rank worse (bench: natural hit@1 56-62% vs 75% with the best region).
fn semantic_document(file: &SmartFile) -> String {
    file.regions
        .first()
        .map(|r| format!("{} {}", normalize_match_text(&r.label), r.body))
        .unwrap_or_else(|| normalize_match_text(&file.path))
}

fn should_prune_region(familiarity: Familiarity) -> bool {
    familiarity.prune_confidence >= 0.7
        && familiarity.body_confidence >= 0.7
        && familiarity.current_version_confidence >= 0.6
}

fn structure_budget_for_file(familiarity: Familiarity) -> usize {
    if familiarity.focused
        && familiarity.structure_confidence >= 0.8
        && familiarity.prune_confidence >= 0.7
    {
        4
    } else if familiarity.structure_confidence >= 0.8
        && familiarity.current_version_confidence >= 0.6
        && familiarity.prune_confidence >= 0.7
    {
        6
    } else {
        10
    }
}

fn context_note_for_file(familiarity: Familiarity) -> Option<String> {
    if familiarity.focused
        && familiarity.structure_confidence >= 0.8
        && familiarity.prune_confidence >= 0.7
    {
        Some("compressed file structure from harness context".to_string())
    } else if familiarity.structure_confidence >= 0.8
        && familiarity.current_version_confidence >= 0.6
        && familiarity.prune_confidence >= 0.7
    {
        Some("reduced repeated structure from harness context".to_string())
    } else {
        None
    }
}

fn classify_region(item: &StructureItem, relation: &Relation, exact_label_match: bool) -> String {
    match relation {
        Relation::Rendered => "render-site".to_string(),
        Relation::Handled => "handler".to_string(),
        Relation::Populated => "assignment".to_string(),
        Relation::CalledFrom => "callsite".to_string(),
        Relation::Defined | Relation::Implementation => {
            if exact_label_match {
                "definition".to_string()
            } else {
                "reference".to_string()
            }
        }
        _ if item.kind == "function" => "reference".to_string(),
        _ => item.kind.clone(),
    }
}

fn exact_subject_path_match(relative_lower: &str, subject_lower: &str) -> bool {
    relative_lower.contains(&subject_lower.replace(' ', "_"))
        || relative_lower.contains(&subject_lower.replace(' ', "-"))
}

fn exact_subject_label_match(label: &str, subject_lower: &str) -> bool {
    let label_lower = label.to_ascii_lowercase();
    label_lower == subject_lower
        || label_lower == subject_lower.replace(' ', "_")
        || label_lower == subject_lower.replace(' ', "-")
}

fn tokenize_subject(subject: &str) -> Vec<String> {
    normalize_match_text(subject)
        .split_whitespace()
        .map(str::to_string)
        .collect()
}

fn subject_tokens_match_label(label: &str, subject_tokens: &[String]) -> bool {
    if subject_tokens.is_empty() {
        return false;
    }
    let normalized_label = normalize_match_text(label);
    subject_tokens
        .iter()
        .all(|token| normalized_label.contains(token.as_str()))
}

pub(crate) fn normalize_match_text(text: &str) -> String {
    let mut out = String::new();
    let mut prev_is_lower = false;
    for ch in text.chars() {
        if ch.is_ascii_uppercase() && prev_is_lower {
            out.push(' ');
        }
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            prev_is_lower = ch.is_ascii_lowercase();
        } else {
            out.push(' ');
            prev_is_lower = false;
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn relation_terms(relation: &Relation) -> Vec<String> {
    match relation {
        Relation::Rendered => vec!["render", "draw", "ui", "widget", "view"],
        Relation::CalledFrom => vec!["call", "invoke", "dispatch"],
        Relation::TriggeredFrom => vec!["trigger", "dispatch", "schedule"],
        Relation::Populated => vec!["set", "assign", "insert", "push", "build"],
        Relation::ComesFrom => vec!["source", "load", "parse", "read", "fetch"],
        Relation::Handled => vec!["handle", "handler", "event", "dispatch"],
        Relation::Defined => vec!["fn", "struct", "enum", "class", "def"],
        Relation::Implementation => vec!["impl", "register", "wire", "tool"],
        Relation::Custom(value) => vec![value.as_str()],
    }
    .into_iter()
    .map(str::to_string)
    .collect()
}

fn role_aligns(role: &str, relation: &Relation) -> bool {
    match relation {
        Relation::Rendered => role == "ui",
        Relation::Handled => role == "handler",
        Relation::ComesFrom => role == "provider" || role == "config",
        Relation::Implementation => role == "implementation" || role == "provider",
        _ => false,
    }
}

fn should_filter_kind(kind: Option<&str>, role: &str) -> bool {
    match kind {
        Some("code") => role == "docs",
        Some("docs") => role != "docs",
        Some("tests") => role != "test",
        _ => false,
    }
}

fn select_structure_items(
    items: &[StructureItem],
    regions: &[SmartRegion],
    max_items: usize,
) -> Vec<StructureItem> {
    let mut selected = Vec::new();
    for region in regions {
        if let Some(item) = items.iter().find(|item| {
            item.label == region.label
                && item.start_line == region.start_line
                && item.end_line == region.end_line
        }) && !selected.iter().any(|existing: &StructureItem| {
            existing.label == item.label
                && existing.start_line == item.start_line
                && existing.end_line == item.end_line
        }) {
            selected.push(item.clone());
        }
    }

    for item in items {
        if selected.len() >= max_items {
            break;
        }
        if !selected.iter().any(|existing| {
            existing.label == item.label
                && existing.start_line == item.start_line
                && existing.end_line == item.end_line
        }) {
            selected.push(item.clone());
        }
    }

    selected
}

fn collect_lower_lines(text: &str) -> Vec<String> {
    text.lines().map(|line| line.to_ascii_lowercase()).collect()
}

fn count_lines(lower_lines: &[String], needle: &str) -> usize {
    lower_lines
        .iter()
        .filter(|line| line.contains(needle))
        .count()
}

fn file_may_contain_subject(
    relative_lower: &str,
    text_lower: &str,
    raw_subject: &str,
    subject_lower: &str,
    subject_tokens: &[String],
) -> bool {
    if relative_lower.contains(subject_lower) || text_lower.contains(subject_lower) {
        return true;
    }

    let underscore_variant = raw_subject.to_ascii_lowercase().replace(' ', "_");
    if underscore_variant != subject_lower
        && (relative_lower.contains(&underscore_variant)
            || text_lower.contains(&underscore_variant))
    {
        return true;
    }

    let hyphen_variant = raw_subject.to_ascii_lowercase().replace(' ', "-");
    if hyphen_variant != subject_lower
        && (relative_lower.contains(&hyphen_variant) || text_lower.contains(&hyphen_variant))
    {
        return true;
    }

    !subject_tokens.is_empty()
        && subject_tokens
            .iter()
            .all(|token| relative_lower.contains(token) || text_lower.contains(token))
}

fn should_include_full_region(item: &StructureItem, mode: FullRegionMode) -> bool {
    match mode {
        FullRegionMode::Always => true,
        FullRegionMode::Never => false,
        FullRegionMode::Auto => item.line_count <= 20,
    }
}

fn extract_region(lines: &[&str], start_line: usize, end_line: usize) -> String {
    lines[start_line.saturating_sub(1)..end_line.min(lines.len())].join("\n")
}

/// Cap each body line so a single huge line (e.g. minified code) cannot
/// flood the caller's context window.
fn compact_region_body(body: &str) -> String {
    if body
        .lines()
        .all(|line| line.chars().count() <= MAX_REGION_BODY_LINE_CHARS)
    {
        return body.to_string();
    }
    body.lines()
        .map(|line| {
            let char_count = line.chars().count();
            if char_count <= MAX_REGION_BODY_LINE_CHARS {
                line.to_string()
            } else {
                let snippet: String = line.chars().take(MAX_REGION_BODY_LINE_CHARS).collect();
                format!(
                    "{} … [truncated: {} chars after]",
                    snippet,
                    char_count - MAX_REGION_BODY_LINE_CHARS
                )
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn region_starts_with_pub(region_lines: &[&str]) -> bool {
    region_lines
        .iter()
        .find_map(|line| {
            let trimmed = line.trim_start();
            (!trimmed.is_empty()).then_some(trimmed)
        })
        .is_some_and(|line| line.starts_with("pub "))
}

fn is_test_like(item: &StructureItem, region_lines: &[&str]) -> bool {
    let label = item.label.to_ascii_lowercase();
    label.contains("test")
        || region_lines.iter().any(|line| {
            line.contains("#[test]") || line.contains("assert_eq!") || line.contains("unwrap_err()")
        })
}

fn looks_like_string_fixture(line: &str) -> bool {
    let trimmed = line.trim();
    let quote_count = trimmed.matches('"').count();
    quote_count >= 2
        && (trimmed.contains("\\n")
            || trimmed.contains("subject:")
            || trimmed.contains("relation:"))
}

fn looks_like_cli_or_example_line(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.contains("agentgrep ")
        || trimmed.contains("cargo run --")
        || trimmed.contains("subject:")
        || trimmed.contains("relation:")
        || trimmed.contains("support:")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{FullRegionMode, SmartArgs};
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn smart_mode_returns_ranked_files_and_regions() {
        let dir = tempdir().unwrap();
        fs::create_dir_all(dir.path().join("src/tui")).unwrap();
        fs::create_dir_all(dir.path().join("src/auth")).unwrap();
        fs::create_dir_all(dir.path().join("docs")).unwrap();
        fs::write(
            dir.path().join("src/tui/app.rs"),
            "fn render_status_bar() {\n    let status = auth_status();\n    println!(\"{}\", status);\n}\n",
        )
        .unwrap();
        fs::write(
            dir.path().join("src/auth/mod.rs"),
            "pub fn auth_status() -> &'static str {\n    \"ok\"\n}\n",
        )
        .unwrap();
        fs::write(
            dir.path().join("docs/notes.md"),
            "# Notes\nwhere is auth_status rendered\nsubject:auth_status relation:rendered\n",
        )
        .unwrap();

        let query = SmartQuery {
            subject: "auth_status".to_string(),
            relation: Relation::Rendered,
            support: vec!["ui".to_string()],
            kind: None,
            path_hint: None,
        };
        let args = SmartArgs {
            terms: vec![],
            json: false,
            max_files: 5,
            max_regions: 5,
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
        };

        let result = run_smart(dir.path(), &query, &args).unwrap();
        assert!(!result.files.is_empty());
        assert_eq!(result.files[0].path, "src/tui/app.rs");
        assert!(!result.files[0].regions.is_empty());
        assert_eq!(result.files[0].regions[0].kind, "render-site");
        assert!(
            result
                .files
                .iter()
                .all(|file| file.path != "docs/notes.md" || file.score < result.files[0].score)
        );
    }

    #[test]
    fn smart_kind_code_filters_out_docs() {
        let dir = tempdir().unwrap();
        fs::create_dir_all(dir.path().join("src/tui")).unwrap();
        fs::create_dir_all(dir.path().join("docs")).unwrap();
        fs::write(
            dir.path().join("src/tui/app.rs"),
            "fn render_status_bar() {\n    let status = auth_status();\n}\n",
        )
        .unwrap();
        fs::write(
            dir.path().join("docs/notes.md"),
            "where is auth_status rendered\nsubject:auth_status relation:rendered\n",
        )
        .unwrap();

        let query = SmartQuery {
            subject: "auth_status".to_string(),
            relation: Relation::Rendered,
            support: vec![],
            kind: Some("code".to_string()),
            path_hint: None,
        };
        let args = SmartArgs {
            terms: vec![],
            json: false,
            max_files: 5,
            max_regions: 5,
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
        };

        let result = run_smart(dir.path(), &query, &args).unwrap();
        assert!(result.files.iter().all(|f| !f.path.ends_with(".md")));
    }

    #[test]
    fn smart_path_hint_biases_subtree() {
        let dir = tempdir().unwrap();
        fs::create_dir_all(dir.path().join("src/tui")).unwrap();
        fs::create_dir_all(dir.path().join("src/other")).unwrap();
        fs::write(
            dir.path().join("src/tui/app.rs"),
            "fn render_status_bar() {\n    let status = auth_status();\n}\n",
        )
        .unwrap();
        fs::write(
            dir.path().join("src/other/app.rs"),
            "fn render_status_bar() {\n    let status = auth_status();\n}\n",
        )
        .unwrap();

        let query = SmartQuery {
            subject: "auth_status".to_string(),
            relation: Relation::Rendered,
            support: vec![],
            kind: Some("code".to_string()),
            path_hint: Some("src/tui".to_string()),
        };
        let args = SmartArgs {
            terms: vec![],
            json: false,
            max_files: 5,
            max_regions: 5,
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
        };

        let result = run_smart(dir.path(), &query, &args).unwrap();
        assert_eq!(result.files.len(), 1);
        assert_eq!(result.files[0].path, "src/tui/app.rs");
    }

    #[test]
    fn smart_penalizes_cli_example_files() {
        let dir = tempdir().unwrap();
        fs::create_dir_all(dir.path().join("src/tui")).unwrap();
        fs::create_dir_all(dir.path().join("src")).unwrap();
        fs::write(
            dir.path().join("src/tui/app.rs"),
            "fn render_status_bar() {\n    let status = auth_status();\n}\n",
        )
        .unwrap();
        fs::write(
            dir.path().join("src/main.rs"),
            "fn main() {\n    eprintln!(\"agentgrep trace subject:auth_status relation:rendered support:ui\");\n}\n",
        )
        .unwrap();

        let query = SmartQuery {
            subject: "auth_status".to_string(),
            relation: Relation::Rendered,
            support: vec!["ui".to_string()],
            kind: Some("code".to_string()),
            path_hint: None,
        };
        let args = SmartArgs {
            terms: vec![],
            json: false,
            max_files: 5,
            max_regions: 5,
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
        };

        let result = run_smart(dir.path(), &query, &args).unwrap();
        assert_eq!(result.files[0].path, "src/tui/app.rs");
    }

    #[test]
    fn implementation_query_prefers_owner_symbols() {
        let dir = tempdir().unwrap();
        fs::create_dir_all(dir.path().join("src/tool")).unwrap();
        fs::write(
            dir.path().join("src/tool/lsp.rs"),
            "pub struct LspTool;\nimpl LspTool {}\nfn description() -> &'static str { \"LSP\" }\n",
        )
        .unwrap();

        let query = SmartQuery {
            subject: "lsp".to_string(),
            relation: Relation::Implementation,
            support: vec![],
            kind: Some("code".to_string()),
            path_hint: Some("src/tool".to_string()),
        };
        let args = SmartArgs {
            terms: vec![],
            json: false,
            max_files: 5,
            max_regions: 5,
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
        };

        let result = run_smart(dir.path(), &query, &args).unwrap();
        assert_eq!(result.files[0].path, "src/tool/lsp.rs");
        assert!(matches!(
            result.files[0].regions[0].kind.as_str(),
            "reference" | "definition"
        ));
        assert!(result.files[0].regions[0].score >= result.files[0].regions[1].score);
    }

    #[test]
    fn trace_context_can_compress_repeated_regions() {
        let dir = tempdir().unwrap();
        fs::create_dir_all(dir.path().join("src/tool")).unwrap();
        fs::write(
            dir.path().join("src/tool/lsp.rs"),
            "pub struct LspTool;\nimpl LspTool {}\nfn execute() {\n    let lsp = true;\n    println!(\"{}\", lsp);\n}\n",
        )
        .unwrap();
        let context_path = dir.path().join("context.json");
        fs::write(
            &context_path,
            r#"{
  "known_files": [
    {
      "path": "src/tool/lsp.rs",
      "structure_confidence": 0.95,
      "current_version_confidence": 0.9,
      "prune_confidence": 0.85
    }
  ],
  "known_regions": [
    {
      "path": "src/tool/lsp.rs",
      "start_line": 3,
      "end_line": 6,
      "body_confidence": 0.95,
      "current_version_confidence": 0.9,
      "prune_confidence": 0.9
    }
  ]
}"#,
        )
        .unwrap();

        let query = SmartQuery {
            subject: "lsp".to_string(),
            relation: Relation::Implementation,
            support: vec![],
            kind: Some("code".to_string()),
            path_hint: Some("src/tool".to_string()),
        };
        let args = SmartArgs {
            terms: vec![],
            json: false,
            max_files: 5,
            max_regions: 5,
            full_region: FullRegionMode::Auto,
            debug_plan: false,
            debug_score: false,
            paths_only: false,
            path: None,
            file_type: None,
            glob: None,
            hidden: false,
            no_ignore: false,
            context_json: Some(context_path.display().to_string()),
        };

        let result = run_smart(dir.path(), &query, &args).unwrap();
        assert!(result.files[0].context_applied.is_some());
        assert!(
            result.files[0]
                .regions
                .iter()
                .any(|region| region.context_applied.is_some())
        );
    }

    #[test]
    fn smart_region_bodies_cap_huge_single_lines() {
        let dir = tempdir().unwrap();
        let giant = "var x=1;".repeat(4000);
        fs::write(
            dir.path().join("minified.js"),
            format!("function handle_auth() {{{giant}}}\n"),
        )
        .unwrap();

        let query = SmartQuery {
            subject: "handle_auth".to_string(),
            relation: Relation::Defined,
            support: vec![],
            kind: None,
            path_hint: None,
        };
        let args = SmartArgs {
            terms: vec![],
            json: false,
            max_files: 5,
            max_regions: 5,
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
        };

        let result = run_smart(dir.path(), &query, &args).unwrap();
        let region = &result.files[0].regions[0];
        assert!(region.body.contains("[truncated:"), "{}", region.body);
        for line in region.body.lines() {
            assert!(
                line.chars().count() < 340,
                "region body line should be bounded, got {} chars",
                line.chars().count()
            );
        }
    }

    fn lexical_args() -> SmartArgs {
        SmartArgs {
            terms: vec![],
            json: false,
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

    fn lexical_query(subject: &str, relation: Relation) -> SmartQuery {
        SmartQuery {
            subject: subject.to_string(),
            relation,
            support: vec![],
            kind: None,
            path_hint: None,
        }
    }

    #[test]
    fn natural_language_subject_finds_code_without_every_word() {
        let dir = tempdir().unwrap();
        fs::create_dir_all(dir.path().join("src")).unwrap();
        fs::write(
            dir.path().join("src/embedder.rs"),
            "pub fn maybe_unload_if_idle(idle_for: Duration) -> bool {\n    \
             // drop the embedder after a period without use\n    true\n}\n",
        )
        .unwrap();
        fs::write(
            dir.path().join("src/other.rs"),
            "pub fn render() {\n    println!(\"hello\");\n}\n",
        )
        .unwrap();

        // "when" is a stopword; "embedder" never appears next to "unload",
        // so the old all-words gate returned nothing.
        let query = lexical_query("unload embedder when idle", Relation::Implementation);
        let result = run_smart(dir.path(), &query, &lexical_args()).unwrap();
        assert_eq!(
            result.files.first().map(|f| f.path.as_str()),
            Some("src/embedder.rs")
        );
        assert_eq!(result.files[0].regions[0].label, "maybe_unload_if_idle");
    }

    #[test]
    fn defining_file_outranks_file_that_mentions_subject_more() {
        let dir = tempdir().unwrap();
        fs::create_dir_all(dir.path().join("src")).unwrap();
        fs::write(
            dir.path().join("src/config.rs"),
            "pub enum CompactionMode {\n    Reactive,\n    Semantic,\n}\n",
        )
        .unwrap();
        fs::write(
            dir.path().join("src/ui.rs"),
            "pub fn set_compaction_mode(mode: CompactionMode) {\n    \
             let _ = CompactionMode::Reactive;\n    let _ = CompactionMode::Semantic;\n}\n",
        )
        .unwrap();

        let query = lexical_query("CompactionMode", Relation::Defined);
        let result = run_smart(dir.path(), &query, &lexical_args()).unwrap();
        assert_eq!(result.files[0].path, "src/config.rs");
    }

    #[test]
    fn stub_copy_ranks_below_real_implementation() {
        let dir = tempdir().unwrap();
        fs::create_dir_all(dir.path().join("src")).unwrap();
        // The stub mentions the subject more often, so without the noise
        // penalty it would win on hit counts alone.
        fs::write(
            dir.path().join("src/embedding_stub.rs"),
            "// cosine similarity stub\n\
             pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {\n    0.0\n}\n\
             pub fn batch_cosine_similarity() {}\n",
        )
        .unwrap();
        fs::write(
            dir.path().join("src/embedding.rs"),
            "pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {\n    \
             a.iter().zip(b).map(|(x, y)| x * y).sum()\n}\n",
        )
        .unwrap();

        let query = lexical_query("cosine similarity", Relation::Implementation);
        let result = run_smart(dir.path(), &query, &lexical_args()).unwrap();
        assert_eq!(result.files[0].path, "src/embedding.rs");
    }

    #[test]
    fn single_word_subject_keeps_strict_gate() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("a.rs"), "fn alpha() {}\n").unwrap();
        fs::write(dir.path().join("b.rs"), "fn beta() {}\n").unwrap();

        let query = lexical_query("gamma", Relation::Defined);
        let result = run_smart(dir.path(), &query, &lexical_args()).unwrap();
        assert!(result.files.is_empty());
    }
}

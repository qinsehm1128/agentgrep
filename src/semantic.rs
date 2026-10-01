//! Optional semantic re-ranking for `smart` mode (feature `semantic`).
//!
//! Uses a Model2Vec static embedding model: a token-embedding lookup plus
//! mean pooling, no transformer forward pass, so a query costs about a
//! millisecond on CPU.
//!
//! This is a **re-ranker, not a recall channel**. It only reorders the files
//! smart mode is about to return, so turning it on never changes which files
//! are returned, only their order.
//!
//! The model is never downloaded here. It is loaded from a local directory
//! (containing `tokenizer.json`, `model.safetensors`, `config.json`) named
//! by `AGENTGREP_SEMANTIC_MODEL`. Without that variable, or if loading
//! fails, semantic re-ranking is silently off.

use std::sync::OnceLock;

/// Environment variable naming the local Model2Vec model directory.
pub const MODEL_ENV: &str = "AGENTGREP_SEMANTIC_MODEL";

#[cfg(feature = "semantic")]
mod imp {
    use super::MODEL_ENV;
    use model2vec_rs::model::StaticModel;
    use std::path::Path;

    pub struct Model(StaticModel);

    impl Model {
        pub fn load() -> Option<Self> {
            let dir = std::env::var_os(MODEL_ENV)?;
            let path = Path::new(&dir);
            if !path.join("model.safetensors").is_file() || !assets_consistent(path) {
                return None;
            }
            StaticModel::from_pretrained(path, None, Some(true), None)
                .ok()
                .map(Model)
        }

        /// `None` if encoding panics. model2vec-rs indexes the embedding
        /// table without bounds checks and `expect`s on tokenizer errors, so
        /// a damaged model must not take the search down with it.
        pub fn embed(&self, texts: &[String]) -> Option<Vec<Vec<f32>>> {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.0.encode_with_args(texts, Some(256), 256)
            }))
            .ok()
        }
    }

    #[cfg(test)]
    pub fn assets_consistent_for_test(dir: &Path) -> bool {
        assets_consistent(dir)
    }

    /// Every token id the tokenizer can emit must index an embedding row
    /// (directly, or through the optional `mapping` tensor).
    fn assets_consistent(dir: &Path) -> bool {
        let Ok(bytes) = std::fs::read(dir.join("model.safetensors")) else {
            return false;
        };
        let Ok(tensors) = safetensors::SafeTensors::deserialize(&bytes) else {
            return false;
        };
        let Ok(embeddings) = tensors.tensor("embeddings") else {
            return false;
        };
        let rows = match embeddings.shape() {
            [rows, _cols] => *rows,
            _ => return false,
        };
        let Ok(tokenizer) = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")) else {
            return false;
        };
        let vocab = tokenizer.get_vocab_size(true);
        match tensors.tensor("mapping") {
            Ok(mapping) => {
                let entries = mapping.shape().first().copied().unwrap_or(0);
                entries >= vocab && mapping_within(&mapping, rows)
            }
            Err(_) => vocab <= rows,
        }
    }

    fn mapping_within(mapping: &safetensors::tensor::TensorView<'_>, rows: usize) -> bool {
        use safetensors::Dtype;
        let data = mapping.data();
        let ok = |v: u64| (v as usize) < rows;
        match mapping.dtype() {
            Dtype::I64 | Dtype::U64 => data
                .chunks_exact(8)
                .all(|c| ok(u64::from_le_bytes(c.try_into().unwrap_or([0xff; 8])))),
            Dtype::I32 | Dtype::U32 => data
                .chunks_exact(4)
                .all(|c| ok(u32::from_le_bytes(c.try_into().unwrap_or([0xff; 4])) as u64)),
            _ => false,
        }
    }
}

#[cfg(not(feature = "semantic"))]
mod imp {
    pub struct Model;

    impl Model {
        pub fn load() -> Option<Self> {
            None
        }

        pub fn embed(&self, _texts: &[String]) -> Option<Vec<Vec<f32>>> {
            None
        }
    }
}

/// The process-wide model, loaded on first use.
pub fn model() -> Option<&'static imp::Model> {
    static MODEL: OnceLock<Option<imp::Model>> = OnceLock::new();
    MODEL.get_or_init(imp::Model::load).as_ref()
}

/// Whether semantic re-ranking is available in this process.
pub fn available() -> bool {
    model().is_some()
}

/// Cosine similarity of the query against each document text.
///
/// Returns `None` when no model is available. Vectors are L2-normalized by
/// the model, so the dot product is the cosine.
pub fn similarities(query: &str, docs: &[String]) -> Option<Vec<f32>> {
    let model = model()?;
    if docs.is_empty() {
        return Some(Vec::new());
    }
    let mut texts = Vec::with_capacity(docs.len() + 1);
    texts.push(query.to_string());
    texts.extend(docs.iter().cloned());
    let vectors = model.embed(&texts)?;
    if vectors.len() != texts.len() {
        return None;
    }
    let (q, rest) = vectors.split_first()?;
    Some(
        rest.iter()
            .map(|d| q.iter().zip(d).map(|(a, b)| a * b).sum())
            .collect(),
    )
}

/// Reciprocal-rank fusion constant, as in the original RRF paper.
pub const RRF_K: f64 = 60.0;

/// Ranks (1-based) of `scores` in descending order; ties share order of index.
pub fn ranks_desc(scores: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    let mut ranks = vec![0; scores.len()];
    for (rank, idx) in order.into_iter().enumerate() {
        ranks[idx] = rank + 1;
    }
    ranks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_desc_orders_by_score() {
        assert_eq!(ranks_desc(&[0.1, 0.9, 0.5]), vec![3, 1, 2]);
        assert_eq!(ranks_desc(&[]), Vec::<usize>::new());
    }

    #[test]
    fn similarities_absent_without_model() {
        if std::env::var_os(MODEL_ENV).is_none() {
            assert!(similarities("q", &["d".to_string()]).is_none());
        }
    }

    #[cfg(feature = "semantic")]
    fn write_model(dir: &std::path::Path, vocab: &[&str], rows: usize) {
        use std::fmt::Write as _;
        let mut v = String::new();
        for (i, t) in vocab.iter().enumerate() {
            let _ = write!(v, "{}\"{t}\":{i}", if i == 0 { "" } else { "," });
        }
        let tokenizer = format!(
            r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{{"type":"Whitespace"}},"post_processor":null,"decoder":null,"model":{{"type":"WordLevel","vocab":{{{v}}},"unk_token":"[UNK]"}}}}"#
        );
        std::fs::write(dir.join("tokenizer.json"), tokenizer).unwrap();
        std::fs::write(dir.join("config.json"), r#"{"normalize": true}"#).unwrap();
        let cols = 2usize;
        let data: Vec<u8> = (0..rows * cols)
            .flat_map(|i| (i as f32 + 1.0).to_le_bytes())
            .collect();
        let header = format!(
            r#"{{"embeddings":{{"dtype":"F32","shape":[{rows},{cols}],"data_offsets":[0,{}]}}}}"#,
            data.len()
        );
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&data);
        std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
    }

    #[cfg(feature = "semantic")]
    #[test]
    fn inconsistent_model_assets_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        // Tokenizer emits ids 0..3, but the table has a single row: encoding
        // would index out of bounds and panic inside model2vec-rs.
        write_model(dir.path(), &["[UNK]", "idle", "unload"], 1);
        assert!(!imp::assets_consistent_for_test(dir.path()));

        write_model(dir.path(), &["[UNK]", "idle", "unload"], 3);
        assert!(imp::assets_consistent_for_test(dir.path()));
    }
}

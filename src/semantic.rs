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

    pub struct Model(StaticModel);

    impl Model {
        pub fn load() -> Option<Self> {
            let dir = std::env::var_os(MODEL_ENV)?;
            let path = std::path::Path::new(&dir);
            if !path.join("model.safetensors").is_file() {
                return None;
            }
            StaticModel::from_pretrained(path, None, Some(true), None)
                .ok()
                .map(Model)
        }

        pub fn embed(&self, texts: &[String]) -> Vec<Vec<f32>> {
            self.0.encode_with_args(texts, Some(256), 256)
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

        pub fn embed(&self, _texts: &[String]) -> Vec<Vec<f32>> {
            Vec::new()
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
    let vectors = model.embed(&texts);
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
}

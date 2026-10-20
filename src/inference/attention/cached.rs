// src/inference/attention/cached.rs

use crate::tensor::Tensor;

/*
TODO potentially remove having the cache as an input to a constructor?
Maybe every constructor shouldn't care about the cache
*/

use super::{AttentionBackend, AttentionError, AttentionInput, AttentionMask};

use super::kv_cache::{KvCacheError, StandardKvCache};

pub struct CachedAttention {
    cache: StandardKvCache,
}

impl CachedAttention {
    pub fn new(cache: StandardKvCache) -> Self {
        Self { cache }
    }

    pub fn cache(&self) -> &StandardKvCache {
        &self.cache
    }

    pub fn cache_mut(&mut self) -> &mut StandardKvCache {
        &mut self.cache
    }

    pub fn with_capacity_defaults(key_dim: usize, value_dim: usize) -> Result<Self, KvCacheError> {
        let cache = StandardKvCache::new(key_dim, value_dim)?;
        Ok(Self::new(cache))
    }
}

impl AttentionBackend for CachedAttention {
    fn forward(&mut self, input: AttentionInput<'_>) -> Result<Tensor, AttentionError> {
        let q_shape = input.q.shape();
        let k_shape = input.k.shape();
        let v_shape = input.v.shape();

        if q_shape.len() != 2 || k_shape.len() != 2 || v_shape.len() != 2 {
            return Err(AttentionError::InvalidShape);
        }

        let seq_q = q_shape[0];
        let d_k = q_shape[1];
        let seq_new_k = k_shape[0];
        let d_v = v_shape[1];

        if seq_q == 0
            || d_k == 0
            || d_v == 0
            || k_shape[1] != d_k
            || v_shape[0] != seq_new_k
            || self.cache.key_dim() != d_k
            || self.cache.value_dim() != d_v
        {
            return Err(AttentionError::InvalidShape);
        }

        // Append only the new K/V rows.
        self.cache.append(input.k, input.v)?;

        let seq_k = self.cache.seq_len();

        if seq_k == 0 || seq_q > seq_k {
            return Err(AttentionError::InvalidShape);
        }

        // Borrow the cache directly. No copying of cached K/V data.
        let keys = self.cache.keys_data();
        let values = self.cache.values_data();

        // Scores = QK^T / sqrt(d_k)
        let scale = 1.0 / (d_k as f32).sqrt();
        let causal_offset = seq_k - seq_q;

        let mut scores = Tensor::zeros(vec![seq_q, seq_k])?;

        for i in 0..seq_q {
            for j in 0..seq_k {
                let mut dot = 0.0;

                for d in 0..d_k {
                    dot += input.q.data()[i * d_k + d] * keys[j * d_k + d];
                }

                let score = if input.mask == AttentionMask::Causal && j > causal_offset + i {
                    f32::NEG_INFINITY
                } else {
                    dot * scale
                };

                scores
                    .set(&[i, j], score)
                    .ok_or(AttentionError::InvalidTensorIndex)?;
            }
        }

        // Probabilities = softmax(scores)
        let probabilities = scores.softmax_last_dim()?;

        // Output = probabilities * V
        let mut output = Tensor::zeros(vec![seq_q, d_v])?;

        for i in 0..seq_q {
            for d in 0..d_v {
                let mut sum = 0.0;

                for j in 0..seq_k {
                    sum += probabilities.data()[i * seq_k + j] * values[j * d_v + d];
                }

                output
                    .set(&[i, d], sum)
                    .ok_or(AttentionError::InvalidTensorIndex)?;
            }
        }

        Ok(output)
    }
}

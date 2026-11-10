// src/inference/attention/cached.rs

use crate::tensor::Tensor;

/*
TODO potentially remove having the cache as an input to a constructor?
Maybe every constructor shouldn't care about the cache
*/

use super::{AttentionBackend, AttentionError, AttentionInput, AttentionMask};

use super::error::KvCacheError;
use super::kv_cache::StandardKvCache;

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

    pub fn with_capacity_defaults(
        key_dim: usize,
        value_dim: usize,
        capacity: usize,
    ) -> Result<Self, KvCacheError> {
        let cache = StandardKvCache::new(key_dim, value_dim, capacity)?;
        Ok(Self::new(cache))
    }
}

impl AttentionBackend for CachedAttention {
    // The cached version can receive less data, since it uses the cache
    // Standard version must receive the full K and V matrices on each .forward() call
    fn forward(&mut self, input: AttentionInput<'_>) -> Result<Tensor, AttentionError> {
        let [query_sequence_length, key_vector_dimension]: &[usize] = input.q.shape() else {
            return Err(AttentionError::InvalidShape);
        };

        let [input_key_sequence_length, key_dimension]: &[usize] = input.k.shape() else {
            return Err(AttentionError::InvalidShape);
        };

        let [input_value_sequence_length, value_vector_dimension]: &[usize] = input.v.shape()
        else {
            return Err(AttentionError::InvalidShape);
        };

        if key_vector_dimension != key_dimension
            || input_key_sequence_length != input_value_sequence_length
            || *key_vector_dimension != self.cache.key_dim()
            || *value_vector_dimension != self.cache.value_dim()
        {
            return Err(AttentionError::InvalidShape);
        }

        // Append only the new K/V rows.
        self.cache.append(input.k.data(), input.v.data())?;

        let key_sequence_length = self.cache.seq_len();

        if key_sequence_length == 0 || *query_sequence_length > key_sequence_length {
            return Err(AttentionError::InvalidShape);
        }

        // Scores = QK^T / sqrt(key_vector_dimension)
        let cached_keys = Tensor::new(
            self.cache.keys_data().to_vec(),
            vec![key_sequence_length, *key_vector_dimension],
        )?;
        let cached_values = Tensor::new(
            self.cache.values_data().to_vec(),
            vec![key_sequence_length, *value_vector_dimension],
        )?;
        let cached_keys_transposed = cached_keys.transpose()?;
        let scores = input.q.matmul(&cached_keys_transposed)?;
        let scale = 1.0 / (*key_vector_dimension as f32).sqrt();
        let mut scores = scores.scale(scale)?;

        if input.mask == AttentionMask::Causal {
            let causal_offset = key_sequence_length - *query_sequence_length;

            for i in 0..*query_sequence_length {
                let last_visible_key = causal_offset + i;

                for j in (last_visible_key + 1)..key_sequence_length {
                    scores
                        .set(&[i, j], f32::NEG_INFINITY)
                        .ok_or(AttentionError::InvalidTensorIndex)?;
                }
            }
        }

        let probabilities = scores.softmax_last_dim()?;

        let output = probabilities.matmul(&cached_values)?;

        Ok(output)
    }
}

// src/inference/attention/paged.rs

use crate::tensor::Tensor;

use super::error::PagedKvCacheError;
use super::paged_cache::PagedKvCache;
use super::{AttentionBackend, AttentionError, AttentionInput, AttentionMask};

// Attention over a paged KV cache.

// Processes the cache blockwise without flattening or copying the cached
// keys and values into contiguous tensors.

// Uses online softmax to combine attention contributions across blocks,
// avoiding the need to store scores for the entire sequence.
pub struct PagedAttention {
    cache: PagedKvCache,
}

impl PagedAttention {
    pub fn new(cache: PagedKvCache) -> Self {
        Self { cache }
    }

    pub fn cache(&self) -> &PagedKvCache {
        &self.cache
    }

    pub fn cache_mut(&mut self) -> &mut PagedKvCache {
        &mut self.cache
    }

    pub fn with_block_size(
        key_dim: usize,
        value_dim: usize,
        block_size: usize,
    ) -> Result<Self, PagedKvCacheError> {
        let cache = PagedKvCache::new(key_dim, value_dim, block_size)?;
        Ok(Self::new(cache))
    }
}

impl AttentionBackend for PagedAttention {
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

        // Append the new K/V rows to the paged cache.
        self.cache.append(input.k.data(), input.v.data())?;

        let key_sequence_length = self.cache.seq_len();

        if *query_sequence_length > key_sequence_length {
            return Err(AttentionError::InvalidShape);
        }

        let num_blocks = self.cache.num_blocks();
        let scale = 1.0 / (*key_vector_dimension as f32).sqrt();
        let causal_offset = key_sequence_length - *query_sequence_length;

        // The output allocation is derived from caller-provided tensor dimensions.
        let output_len = query_sequence_length
            .checked_mul(*value_vector_dimension)
            .ok_or(AttentionError::InvalidShape)?;

        let mut output = Vec::new();

        // Preserve the attention error contract if the output cannot be allocated.
        output
            .try_reserve_exact(output_len)
            .map_err(|_| AttentionError::InvalidShape)?;

        // Process each query independently, visiting KV blocks in order.
        for i in 0..*query_sequence_length {
            let query_start = i * *key_vector_dimension;
            let query = &input.q.data()[query_start..(query_start + *key_vector_dimension)];

            // Online softmax state:
            // max_score: maximum score seen so far.
            // denominator: running softmax denominator.
            // accumulator: running weighted sum of value vectors.
            let mut max_score = f32::NEG_INFINITY;
            let mut denominator = 0.0_f32;
            let mut accumulator = vec![0.0_f32; *value_vector_dimension];

            // for each block
            for block_idx in 0..num_blocks {
                let (keys, values, valid_tokens, block_start) = self
                    .cache
                    .block(block_idx)
                    .ok_or(AttentionError::InvalidShape)?;

                // Compute attention scores for this block.
                let mut scores = Vec::new();

                // Scores are temporary, but allocation failure must still be reported through
                // the fallible attention API.
                scores
                    .try_reserve_exact(valid_tokens)
                    .map_err(|_| AttentionError::InvalidShape)?;

                let mut block_max = f32::NEG_INFINITY;

                for local_j in 0..valid_tokens {
                    let token_idx = block_start + local_j;

                    // Exclude future tokens for causal attention.
                    // if the key token is newer than the 
                    // current query is allowed to see
                    // then hide it
                    if input.mask == AttentionMask::Causal && token_idx > causal_offset + i {
                        scores.push(f32::NEG_INFINITY);
                        continue;
                    }

                    let key_start = local_j * *key_vector_dimension;
                    let key = &keys[key_start..key_start + *key_vector_dimension];

                    let mut dot = 0.0_f32;

                    // compute the dot product for this key
                    for d in 0..*key_vector_dimension {
                        dot += query[d] * key[d];
                    }

                    let score = dot * scale;

                    block_max = block_max.max(score);
                    scores.push(score);
                }

                // A fully masked block contributes nothing.
                if block_max == f32::NEG_INFINITY {
                    continue;
                }

                // Update the running maximum and rescale the accumulated
                // denominator and output to the new maximum.
                let new_max = max_score.max(block_max);

                let old_scale = if denominator == 0.0 {
                    0.0
                } else {
                    (max_score - new_max).exp()
                };

                denominator *= old_scale;

                for value in &mut accumulator {
                    *value *= old_scale;
                }

                // Incorporate this block into the online softmax.
                for (local_j, &score) in scores.iter().enumerate() {
                    if score == f32::NEG_INFINITY {
                        continue;
                    }

                    let weight = (score - new_max).exp();

                    denominator += weight;

                    let value_start = local_j * *value_vector_dimension;
                    let value = &values[value_start..value_start + *value_vector_dimension];

                    for d in 0..*value_vector_dimension {
                        accumulator[d] += weight * value[d];
                    }
                }

                max_score = new_max;
            }

            if denominator == 0.0 || !denominator.is_finite() {
                return Err(AttentionError::InvalidShape);
            }

            // Normalize the accumulated weighted sum.
            for value in &mut accumulator {
                *value /= denominator;
            }

            output.extend_from_slice(&accumulator);
        }

        Ok(Tensor::new(
            output,
            vec![*query_sequence_length, *value_vector_dimension],
        )?)
    }
}


use crate::tensor::Tensor;

use super::paged_cache::{PagedKvCache, PagedKvCacheError};
use super::{AttentionBackend, AttentionError, AttentionInput, AttentionMask};

// Attention over a paged KV cache.
//
// Processes the cache blockwise without flattening or copying the cached
// keys and values into contiguous tensors.
//
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
    fn forward(
        &mut self,
        input: AttentionInput<'_>,
    ) -> Result<Tensor, AttentionError> {
        let q_shape = input.q.shape();
        let k_shape = input.k.shape();
        let v_shape = input.v.shape();

        if q_shape.len() != 2
            || k_shape.len() != 2
            || v_shape.len() != 2
        {
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

        // Append the new K/V rows to the paged cache.
        self.cache.append(input.k, input.v)?;

        let seq_k = self.cache.seq_len();

        if seq_q > seq_k {
            return Err(AttentionError::InvalidShape);
        }

        let block_size = self.cache.block_size();
        let num_blocks = self.cache.num_blocks();
        let scale = 1.0 / (d_k as f32).sqrt();
        let causal_offset = seq_k - seq_q;

        let output_len = seq_q
            .checked_mul(d_v)
            .ok_or(AttentionError::InvalidShape)?;

        let mut output = Vec::new();

        output
            .try_reserve_exact(output_len)
            .map_err(|_| AttentionError::InvalidShape)?;

        // Process each query independently, visiting KV blocks in order.
        for i in 0..seq_q {
            let query_start = i * d_k;
            let query = &input.q.data()[query_start..query_start + d_k];

            // Online softmax state:
            // max_score: maximum score seen so far.
            // denominator: running softmax denominator.
            // accumulator: running weighted sum of value vectors.
            let mut max_score = f32::NEG_INFINITY;
            let mut denominator = 0.0_f32;
            let mut accumulator = vec![0.0_f32; d_v];

            for block_idx in 0..num_blocks {
                let keys = self.cache
                    .key_block(block_idx)
                    .ok_or(AttentionError::InvalidShape)?;

                let values = self.cache
                    .value_block(block_idx)
                    .ok_or(AttentionError::InvalidShape)?;

                let valid_tokens = self.cache
                    .valid_tokens_in_block(block_idx)
                    .ok_or(AttentionError::InvalidShape)?;

                let block_start = block_idx * block_size;

                // Compute attention scores for this block.
                let mut scores = Vec::new();

                scores
                    .try_reserve_exact(valid_tokens)
                    .map_err(|_| AttentionError::InvalidShape)?;

                let mut block_max = f32::NEG_INFINITY;

                for local_j in 0..valid_tokens {
                    let token_idx = block_start + local_j;

                    // Exclude future tokens for causal attention.
                    if input.mask == AttentionMask::Causal
                        && token_idx > causal_offset + i
                    {
                        scores.push(f32::NEG_INFINITY);
                        continue;
                    }

                    let key_start = local_j * d_k;
                    let key = &keys[key_start..key_start + d_k];

                    let mut dot = 0.0_f32;

                    for d in 0..d_k {
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

                    let value_start = local_j * d_v;
                    let value = &values[value_start..value_start + d_v];

                    for d in 0..d_v {
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

        Ok(Tensor::new(output, vec![seq_q, d_v])?)
    }
}

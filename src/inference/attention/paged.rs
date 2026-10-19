
use crate::tensor::Tensor;

use super::{
    AttentionBackend, AttentionError, AttentionInput, AttentionMask,
};
use super::paged_cache::{PagedKvCache, PagedKvCacheError};

// Attention over a paged KV cache.
//
// This is a straightforward CPU implementation. It computes attention
// directly from the cache's blocks, without flattening the cache into
// contiguous tensors.
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

        let scale = 1.0 / (d_k as f32).sqrt();
        let causal_offset = seq_k - seq_q;

        let mut output = Vec::with_capacity(seq_q * d_v);

        for i in 0..seq_q {
            // Compute QK^T directly from the paged cache.
            let mut scores = Vec::with_capacity(seq_k);

            for j in 0..seq_k {
                if input.mask == AttentionMask::Causal
                    && j > causal_offset + i
                {
                    scores.push(f32::NEG_INFINITY);
                    continue;
                }

                let key = self.cache.key_at(j)
                    .ok_or(AttentionError::InvalidShape)?;

                let mut dot = 0.0_f32;

                for d in 0..d_k {
                    dot += input.q.data()[i * d_k + d] * key[d];
                }

                scores.push(dot * scale);
            }

            // Numerically stable softmax.
            let max_score = scores
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);

            let mut exp_scores: Vec<f32> = scores
                .iter()
                .map(|&score| (score - max_score).exp())
                .collect();

            let denominator: f32 = exp_scores.iter().sum();

            if denominator == 0.0 || !denominator.is_finite() {
                return Err(AttentionError::InvalidShape);
            }

            for weight in &mut exp_scores {
                *weight /= denominator;
            }

            // Compute the weighted sum of V directly from the blocks.
            let mut row_output = vec![0.0_f32; d_v];

            for (j, &weight) in exp_scores.iter().enumerate() {
                if weight == 0.0 {
                    continue;
                }

                let value = self.cache.value_at(j)
                    .ok_or(AttentionError::InvalidShape)?;

                for d in 0..d_v {
                    row_output[d] += weight * value[d];
                }
            }

            output.extend(row_output);
        }

        Ok(Tensor::new(output, vec![seq_q, d_v])?)
    }
}

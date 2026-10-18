use crate::tensor::Tensor;

use super::{
    AttentionBackend,
    AttentionError,
    AttentionInput,
    AttentionMask,
};

use super::kv_cache::{KvCache, KvCacheError, StandardKvCache};

pub struct CachedAttention {
    cache: StandardKvCache,
}

impl CachedAttention {
    pub fn new(
        key_dim: usize,
        value_dim: usize,
    ) -> Result<Self, KvCacheError> {
        Ok(Self {
            cache: StandardKvCache::new(key_dim, value_dim)?,
        })
    }
}

impl AttentionBackend for CachedAttention {
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
        let d_v = v_shape[1];

        if d_k == 0
            || d_v == 0
            || k_shape[1] != d_k
            || k_shape[0] != v_shape[0]
        {
            return Err(AttentionError::InvalidShape);
        }

        // Append only the new K/V rows.
        self.cache.append(input.k, input.v)?;

        let seq_k = self.cache.seq_len();

        if seq_q > seq_k {
            return Err(AttentionError::InvalidShape);
        }

        let causal_offset = seq_k - seq_q;
        let scale = 1.0 / (d_k as f32).sqrt();
        let mut output = Tensor::zeros(vec![seq_q, d_v])?;

        for i in 0..seq_q {
            let visible_keys = match input.mask {
                AttentionMask::Causal => causal_offset + i + 1,
                AttentionMask::None => seq_k,
            };

            let mut scores = Vec::with_capacity(visible_keys);
            let mut max_score = f32::NEG_INFINITY;

            // QK^T, reading keys directly from cache storage.
            for j in 0..visible_keys {
                let mut dot = 0.0_f32;

                for d in 0..d_k {
                    let q = input.q
                        .get(&[i, d])
                        .ok_or(AttentionError::InvalidTensorIndex)?;

                    let k = self.cache
                        .key_at(j, d)
                        .ok_or(AttentionError::InvalidTensorIndex)?;

                    dot += q * k;
                }

                let score = dot * scale;

                if !score.is_finite() {
                    return Err(AttentionError::InvalidShape);
                }

                max_score = max_score.max(score);
                scores.push(score);
            }

            // Numerically stable softmax.
            let mut denominator = 0.0_f32;

            for score in &mut scores {
                *score = (*score - max_score).exp();
                denominator += *score;
            }

            if !denominator.is_finite() || denominator == 0.0 {
                return Err(AttentionError::InvalidShape);
            }

            // softmax(QK^T)V, reading values directly from cache.
            for d in 0..d_v {
                let mut result = 0.0_f32;

                for (j, &score) in scores.iter().enumerate() {
                    let v = self.cache
                        .value_at(j, d)
                        .ok_or(AttentionError::InvalidTensorIndex)?;

                    result += (score / denominator) * v;
                }

                output
                    .set(&[i, d], result)
                    .ok_or(AttentionError::InvalidTensorIndex)?;
            }
        }

        Ok(output)
    }
}
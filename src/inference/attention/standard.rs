use crate::tensor::Tensor;

use super::{AttentionBackend, AttentionError, AttentionInput, AttentionMask};

pub struct StandardAttention;

impl AttentionBackend for StandardAttention {
    fn forward(&mut self, input: AttentionInput<'_>) -> Result<Tensor, AttentionError> {
        let q_shape = input.q.shape();
        let k_shape = input.k.shape();
        let v_shape = input.v.shape();

        // This implementation supports 2D, single-head attention.
        if q_shape.len() != 2 || k_shape.len() != 2 || v_shape.len() != 2 {
            return Err(AttentionError::InvalidShape);
        }

        let seq_q = q_shape[0];
        let d_k = q_shape[1];
        let seq_k = k_shape[0];
        let d_v = v_shape[1];

        if seq_q == 0
            || seq_k == 0
            || d_k == 0
            || d_v == 0
            || k_shape[1] != d_k
            || v_shape[0] != seq_k
        {
            return Err(AttentionError::InvalidShape);
        }

        // Scores = QK^T / sqrt(d_k)
        let k_transposed = input.k.transpose()?;
        let scores = input.q.matmul(&k_transposed)?;
        let scale = 1.0 / (d_k as f32).sqrt();
        let mut scores = scores.scale(scale)?;

        let causal_offset = seq_k - seq_q;

        if input.mask == AttentionMask::Causal {
            for i in 0..seq_q {
                let last_visible_key = causal_offset + i;

                for j in (last_visible_key + 1)..seq_k {
                    scores
                        .set(&[i, j], f32::NEG_INFINITY)
                        .ok_or(AttentionError::InvalidTensorIndex)?;
                }
            }
        }

        // Probabilities = softmax(scores)
        let probabilities = scores.softmax_last_dim()?;

        // Output = probabilities * V
        let output = probabilities.matmul(input.v)?;

        Ok(output)
    }
}

use crate::tensor::Tensor;

use super::{
    AttentionBackend,
    AttentionError,
    AttentionInput,
    AttentionMask,
};

pub struct StandardAttention;

impl StandardAttention {
    fn read(
        tensor: &Tensor,
        indices: &[usize],
    ) -> Result<f32, AttentionError> {
        tensor
            .get(indices)
            .ok_or(AttentionError::InvalidTensorIndex)
    }

    fn write(
        tensor: &mut Tensor,
        indices: &[usize],
        value: f32,
    ) -> Result<(), AttentionError> {
        tensor
            .set(indices, value)
            .map(|_| ())
            .ok_or(AttentionError::InvalidTensorIndex)
    }
}

impl AttentionBackend for StandardAttention {
    fn forward(
        &mut self,
        input: AttentionInput<'_>,
    ) -> Result<Tensor, AttentionError> {
        let q_shape = input.q.shape();
        let k_shape = input.k.shape();
        let v_shape = input.v.shape();

        // This implementation supports 2D, single-head attention.
        if q_shape.len() != 2
            || k_shape.len() != 2
            || v_shape.len() != 2
        {
            return Err(AttentionError::InvalidShape);
        }

        let seq_q = q_shape[0];
        let d_k = q_shape[1];
        let seq_k = k_shape[0];
        let d_k_actual = k_shape[1];
        let seq_v = v_shape[0];
        let d_v = v_shape[1];

        if d_k == 0
            || d_k != d_k_actual
            || seq_k != seq_v
        {
            return Err(AttentionError::InvalidShape);
        }

        // Scores = QK^T / sqrt(d_k)
        let mut scores = Tensor::zeros(vec![seq_q, seq_k])?;
        let scale = 1.0 / (d_k as f32).sqrt();

        for i in 0..seq_q {
            for j in 0..seq_k {
                let mut dot_product = 0.0;

                for d in 0..d_k {
                    let q_value = Self::read(input.q, &[i, d])?;
                    let k_value = Self::read(input.k, &[j, d])?;

                    dot_product += q_value * k_value;
                }

                Self::write(
                    &mut scores,
                    &[i, j],
                    dot_product * scale,
                )?;
            }
        }

        // Apply causal masking.
        // Assumes query and key positions share the same starting index.
        if input.mask == AttentionMask::Causal {
            for i in 0..seq_q {
                for j in (i + 1)..seq_k {
                    Self::write(
                        &mut scores,
                        &[i, j],
                        f32::NEG_INFINITY,
                    )?;
                }
            }
        }

        // Softmax each row, subtracting the maximum for numerical stability.
        let mut probabilities = Tensor::zeros(vec![seq_q, seq_k])?;

        for i in 0..seq_q {
            let mut max_score = f32::NEG_INFINITY;

            for j in 0..seq_k {
                let score = Self::read(&scores, &[i, j])?;
                max_score = max_score.max(score);
            }

            if !max_score.is_finite() {
                return Err(AttentionError::InvalidShape);
            }

            let mut denominator = 0.0;

            for j in 0..seq_k {
                let score = Self::read(&scores, &[i, j])?;
                let probability = (score - max_score).exp();

                Self::write(
                    &mut probabilities,
                    &[i, j],
                    probability,
                )?;

                denominator += probability;
            }

            if !denominator.is_finite() || denominator == 0.0 {
                return Err(AttentionError::InvalidShape);
            }

            for j in 0..seq_k {
                let probability =
                    Self::read(&probabilities, &[i, j])? / denominator;

                Self::write(
                    &mut probabilities,
                    &[i, j],
                    probability,
                )?;
            }
        }

        // Output = softmax(scores) V
        let mut output = Tensor::zeros(vec![seq_q, d_v])?;

        for i in 0..seq_q {
            for d in 0..d_v {
                let mut value = 0.0;

                for j in 0..seq_k {
                    let probability =
                        Self::read(&probabilities, &[i, j])?;
                    let v_value = Self::read(input.v, &[j, d])?;

                    value += probability * v_value;
                }

                Self::write(&mut output, &[i, d], value)?;
            }
        }

        Ok(output)
    }
}
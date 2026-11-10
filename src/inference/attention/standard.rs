// src/inference/attention/standard.rs

use crate::tensor::Tensor;

use super::{AttentionBackend, AttentionError, AttentionInput, AttentionMask};

pub struct StandardAttention;

impl AttentionBackend for StandardAttention {
    fn forward(&mut self, input: AttentionInput<'_>) -> Result<Tensor, AttentionError> {
        let [query_sequence_length, key_vector_dimension]: &[usize] = input.q.shape() else {
            return Err(AttentionError::InvalidShape);
        };

        let [key_sequence_length, key_dimension]: &[usize] = input.k.shape() else {
            return Err(AttentionError::InvalidShape);
        };

        let [value_sequence_length, _value_vector_dimension]: &[usize] = input.v.shape() else {
            return Err(AttentionError::InvalidShape);
        };

        // Q: query_sequence_length x key_vector_dimension
        // K^T: key_vector_dimension x key_sequence_length,
        // so K: key_sequence_length x key_vector_dimension
        // V: key_sequence_length x value_vector_dimension

        // q_seq_length can be different to key_sequence_length
        // for example one new token can attend to all the previous tokens
        if key_vector_dimension != key_dimension || key_sequence_length != value_sequence_length {
            return Err(AttentionError::InvalidShape);
        }

        // Scores = QK^T / sqrt(key_vector_dimension)
        let k_transposed = input.k.transpose()?;
        let scores = input.q.matmul(&k_transposed)?;

        // keep the dot product scores from becoming too large
        // From Attention Is All You Need: 
        // "[for high d_k] the dot products grow large in magnitude, pushing the softmax function into regions where it has
        //  extremely small gradients"
        
        // let each component be q_i or k_i
        // when you add components, the variances add
        // so the variance is d_k multiples of the variance of a single component
        // dividing by sqrt(d_k) keeps the variance stable
        let scale = 1.0 / (*key_vector_dimension as f32).sqrt();
        let mut scores = scores.scale(scale)?;


        if input.mask == AttentionMask::Causal {
            // shift the key sequence length
            let causal_offset = key_sequence_length - query_sequence_length;

            for i in 0..*query_sequence_length {
                let last_visible_key = causal_offset + i;

                for j in (last_visible_key + 1)..*key_sequence_length {
                    scores
                    // negative infinity will lead to 0 softmax score
                        .set(&[i, j], f32::NEG_INFINITY) 
                        .ok_or(AttentionError::InvalidTensorIndex)?;
                }
            }
        }

        let probabilities = scores.softmax_last_dim()?;

        let output = probabilities.matmul(input.v)?;

        Ok(output)
    }
}

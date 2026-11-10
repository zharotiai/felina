// src/inference/attention/types.rs

use crate::tensor::Tensor;

use super::error::AttentionError;

pub struct AttentionInput<'a> {
    pub q: &'a Tensor,
    pub k: &'a Tensor,
    pub v: &'a Tensor,
    pub mask: AttentionMask,
}

pub trait AttentionBackend {
    fn forward(&mut self, input: AttentionInput<'_>) -> Result<Tensor, AttentionError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionMask {
    Causal,
    None,
}


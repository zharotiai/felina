use std::fmt;
use crate::tensor::{Tensor, TensorError};
use super::kv_cache::KvCacheError;

pub struct AttentionInput<'a> {
    pub q: &'a Tensor,
    pub k: &'a Tensor,
    pub v: &'a Tensor,
    pub mask: AttentionMask,
}

pub trait AttentionBackend {
    fn forward(
        &mut self, 
        input: AttentionInput<'_>, 
    ) -> Result<Tensor, AttentionError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionMask {
    Causal,
    None,
}

#[derive(Debug)]
pub enum AttentionError {
    InvalidShape,
    InvalidTensorIndex,
    Tensor(TensorError),
    KvCache(KvCacheError),
}

impl From<TensorError> for AttentionError {
    fn from(err: TensorError) -> Self {
        AttentionError::Tensor(err)
    }
}

impl From<KvCacheError> for AttentionError {
    fn from(err: KvCacheError) -> Self {
        Self::KvCache(err)
    }
}

impl std::error::Error for AttentionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AttentionError::Tensor(err) => Some(err),
            AttentionError::InvalidShape
            | AttentionError::InvalidTensorIndex => None,
            AttentionError::KvCache(err) => Some(err),
        }
    }
}

impl fmt::Display for AttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AttentionError::InvalidShape => {
                write!(f, "Invalid tensor shapes for attention")
            }
            AttentionError::InvalidTensorIndex => {
                write!(f, "Invalid tensor index during attention computation")
            }
            AttentionError::Tensor(err) => {
                write!(f, "Tensor error: {err}")
            }
            AttentionError::KvCache(err) => {
                write!(f, "KV cache error: {err}")
            }
        }
    }
}




// src/inference/attention/error.rs

use std::fmt;

use super::kv_cache::KvCacheError;
use super::paged_cache::PagedKvCacheError;
use super::types::AttentionError;
use crate::tensor::TensorError;

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

impl From<PagedKvCacheError> for AttentionError {
    fn from(err: PagedKvCacheError) -> Self {
        Self::PagedKvCache(err)
    }
}

impl std::error::Error for AttentionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AttentionError::Tensor(err) => Some(err),
            AttentionError::InvalidShape | AttentionError::InvalidTensorIndex => None,
            AttentionError::KvCache(err) => Some(err),
            AttentionError::PagedKvCache(err) => Some(err),
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
            AttentionError::PagedKvCache(err) => {
                write!(f, "Paged KV cache error: {err}")
            }
        }
    }
}

// src/inference/attention/error.rs

use std::fmt;

use crate::tensor::TensorError;

#[derive(Debug)]
pub enum KvCacheError {
    InvalidShape,
    CapacityOverflow,
    AllocationFailed,
    Tensor(TensorError),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PagedKvCacheError {
    InvalidShape,
    ShapeOverflow,
    AllocationFailed,
}

#[derive(Debug)]
pub enum AttentionError {
    InvalidShape,
    InvalidTensorIndex,
    Tensor(TensorError),
    KvCache(KvCacheError),
    PagedKvCache(PagedKvCacheError),
}

impl From<TensorError> for KvCacheError {
    fn from(err: TensorError) -> Self {
        Self::Tensor(err)
    }
}

impl fmt::Display for KvCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => write!(f, "Invalid shapes for KV cache"),
            Self::CapacityOverflow => write!(f, "KV cache capacity overflow"),
            Self::AllocationFailed => write!(f, "Failed to allocate KV cache"),
            Self::Tensor(err) => write!(f, "Tensor error: {err}"),
        }
    }
}

impl std::error::Error for KvCacheError {}

impl fmt::Display for PagedKvCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => write!(f, "Invalid key/value tensor shape"),
            Self::ShapeOverflow => write!(f, "Paged KV cache shape overflow"),
            Self::AllocationFailed => write!(f, "Failed to allocate paged KV cache"),
        }
    }
}

impl std::error::Error for PagedKvCacheError {}

impl From<TensorError> for AttentionError {
    fn from(err: TensorError) -> Self {
        Self::Tensor(err)
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

impl std::error::Error for AttentionError {}

impl fmt::Display for AttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => write!(f, "Invalid tensor shapes for attention"),
            Self::InvalidTensorIndex => {
                write!(f, "Invalid tensor index during attention computation")
            }
            Self::Tensor(err) => write!(f, "Tensor error: {err}"),
            Self::KvCache(err) => write!(f, "KV cache error: {err}"),
            Self::PagedKvCache(err) => write!(f, "Paged KV cache error: {err}"),
        }
    }
}


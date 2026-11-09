// src/inference/attention/types.rs

use super::kv_cache::KvCacheError;
use super::paged_cache::PagedKvCacheError;
use crate::tensor::{Tensor, TensorError};

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

#[derive(Debug)]
pub enum AttentionError {
    InvalidShape,
    InvalidTensorIndex,
    Tensor(TensorError),
    KvCache(KvCacheError),
    PagedKvCache(PagedKvCacheError),
}

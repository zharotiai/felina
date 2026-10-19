use crate::tensor::{Tensor, TensorError};
use std::fmt;

/*
Standard implementation of attention with a KV cache.
*/

// Initial number of token slots allocated by the standard cache.
pub const INITIAL_KV_CACHE_CAPACITY: usize = 512;

#[derive(Debug)]
pub enum KvCacheError {
    InvalidShape,
    CapacityOverflow,
    Tensor(TensorError),
}

impl From<TensorError> for KvCacheError {
    fn from(err: TensorError) -> Self {
        Self::Tensor(err)
    }
}

impl fmt::Display for KvCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KvCacheError::InvalidShape => {
                write!(f, "Invalid shapes for KV cache")
            }
            KvCacheError::CapacityOverflow => {
                write!(f, "KV cache capacity overflow")
            }
            KvCacheError::Tensor(err) => {
                write!(f, "Tensor error: {err}")
            }
        }
    }
}

impl std::error::Error for KvCacheError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            KvCacheError::Tensor(err) => Some(err),
            KvCacheError::InvalidShape
            | KvCacheError::CapacityOverflow => None,
        }
    }
}

/// A contiguous KV cache with dynamically growing capacity.
pub struct StandardKvCache {
    keys: Tensor,
    values: Tensor,
    seq_len: usize,
    capacity: usize,
    key_dim: usize,
    value_dim: usize,
}

impl StandardKvCache {
    pub fn new(
        key_dim: usize,
        value_dim: usize,
    ) -> Result<Self, KvCacheError> {
        if key_dim == 0 || value_dim == 0 {
            return Err(KvCacheError::InvalidShape);
        }

        Ok(Self {
            keys: Tensor::zeros(vec![
                INITIAL_KV_CACHE_CAPACITY,
                key_dim,
            ])?,
            values: Tensor::zeros(vec![
                INITIAL_KV_CACHE_CAPACITY,
                value_dim,
            ])?,
            seq_len: 0,
            capacity: INITIAL_KV_CACHE_CAPACITY,
            key_dim,
            value_dim,
        })
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    pub fn key_dim(&self) -> usize {
        self.key_dim
    }

    pub fn value_dim(&self) -> usize {
        self.value_dim
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the populated key data as a borrowed contiguous slice.
    pub fn keys_data(&self) -> &[f32] {
        &self.keys.data()[..self.seq_len * self.key_dim]
    }

    /// Returns the populated value data as a borrowed contiguous slice.
    pub fn values_data(&self) -> &[f32] {
        &self.values.data()[..self.seq_len * self.value_dim]
    }

    fn ensure_capacity(
        &mut self,
        required: usize,
    ) -> Result<(), KvCacheError> {
        if required <= self.capacity {
            return Ok(());
        }

        let new_capacity = self
            .capacity
            .checked_mul(2)
            .filter(|&capacity| capacity >= required)
            .unwrap_or(required);

        let mut new_keys =
            Tensor::zeros(vec![new_capacity, self.key_dim])?;
        let mut new_values =
            Tensor::zeros(vec![new_capacity, self.value_dim])?;

        for token in 0..self.seq_len {
            for d in 0..self.key_dim {
                let value = self.keys
                    .get(&[token, d])
                    .ok_or(KvCacheError::InvalidShape)?;

                new_keys
                    .set(&[token, d], value)
                    .ok_or(KvCacheError::InvalidShape)?;
            }

            for d in 0..self.value_dim {
                let value = self.values
                    .get(&[token, d])
                    .ok_or(KvCacheError::InvalidShape)?;

                new_values
                    .set(&[token, d], value)
                    .ok_or(KvCacheError::InvalidShape)?;
            }
        }

        self.keys = new_keys;
        self.values = new_values;
        self.capacity = new_capacity;

        Ok(())
    }

    pub fn append(
        &mut self,
        keys: &Tensor,
        values: &Tensor,
    ) -> Result<(), KvCacheError> {
        let key_shape = keys.shape();
        let value_shape = values.shape();

        if key_shape.len() != 2
            || value_shape.len() != 2
            || key_shape[1] != self.key_dim
            || value_shape[1] != self.value_dim
            || key_shape[0] != value_shape[0]
        {
            return Err(KvCacheError::InvalidShape);
        }

        let new_seq_len = self.seq_len
            .checked_add(key_shape[0])
            .ok_or(KvCacheError::CapacityOverflow)?;

        self.ensure_capacity(new_seq_len)?;

        for token in 0..key_shape[0] {
            for d in 0..self.key_dim {
                let value = keys
                    .get(&[token, d])
                    .ok_or(KvCacheError::InvalidShape)?;

                self.keys
                    .set(&[self.seq_len + token, d], value)
                    .ok_or(KvCacheError::InvalidShape)?;
            }

            for d in 0..self.value_dim {
                let value = values
                    .get(&[token, d])
                    .ok_or(KvCacheError::InvalidShape)?;

                self.values
                    .set(&[self.seq_len + token, d], value)
                    .ok_or(KvCacheError::InvalidShape)?;
            }
        }

        self.seq_len = new_seq_len;
        Ok(())
    }

    pub fn key_at(&self, token: usize, dim: usize) -> Option<f32> {
        if token >= self.seq_len || dim >= self.key_dim {
            return None;
        }

        self.keys.get(&[token, dim])
    }

    pub fn value_at(&self, token: usize, dim: usize) -> Option<f32> {
        if token >= self.seq_len || dim >= self.value_dim {
            return None;
        }

        self.values.get(&[token, dim])
    }
}

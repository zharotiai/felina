// src/inference/attention/kv_cache.rs

use super::error::KvCacheError;

// A contiguous KV cache with dynamically growing capacity.
pub struct StandardKvCache {
    keys: Vec<f32>,
    values: Vec<f32>,
    seq_len: usize,
    capacity: usize,
    // key and value vector can have different dimensions
    key_dim: usize,
    value_dim: usize,
}

impl StandardKvCache {
    pub fn new(
        key_dim: usize,
        value_dim: usize,
        capacity: usize,
    ) -> Result<Self, KvCacheError> {
        if key_dim == 0 || value_dim == 0 || capacity == 0 {
            return Err(KvCacheError::InvalidShape);
        }

        Ok(Self {
            keys: Self::allocate_buffer(capacity, key_dim)?,
            values: Self::allocate_buffer(capacity, value_dim)?,
            seq_len: 0,
            capacity,
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

    fn allocate_buffer(capacity: usize, dimension: usize) -> Result<Vec<f32>, KvCacheError> {
        // allocates a buffer of capacity * dimension size
        let element_count = capacity
            .checked_mul(dimension)
            .ok_or(KvCacheError::CapacityOverflow)?;
        
        let mut buffer = Vec::new();
        buffer
            .try_reserve_exact(element_count)
            .map_err(|_| KvCacheError::AllocationFailed)?;
        buffer.resize(element_count, 0.0);
        Ok(buffer)
    }

    // Returns the populated key data as a borrowed contiguous slice.
    pub fn keys_data(&self) -> &[f32] {
        &self.keys[..self.seq_len * self.key_dim]
    }

    // Returns the populated value data as a borrowed contiguous slice.
    pub fn values_data(&self) -> &[f32] {
        &self.values[..self.seq_len * self.value_dim]
    }

    fn ensure_capacity(&mut self, required: usize) -> Result<(), KvCacheError> {
        if required <= self.capacity {
            return Ok(());
        }

        let new_capacity = self
            .capacity
            // double capacity if not big enough
            .checked_mul(2)
            .filter(|&capacity| capacity >= required)
            .unwrap_or(required);

        let mut new_keys = Self::allocate_buffer(new_capacity, self.key_dim)?;
        let mut new_values = Self::allocate_buffer(new_capacity, self.value_dim)?;
        let key_data_length = self
            .seq_len
            .checked_mul(self.key_dim)
            .ok_or(KvCacheError::CapacityOverflow)?;
        let value_data_length = self
            .seq_len
            .checked_mul(self.value_dim)
            .ok_or(KvCacheError::CapacityOverflow)?;

        new_keys[..key_data_length].copy_from_slice(&self.keys[..key_data_length]);
        new_values[..value_data_length].copy_from_slice(&self.values[..value_data_length]);

        self.keys = new_keys;
        self.values = new_values;
        self.capacity = new_capacity;

        Ok(())
    }

    // appending the raw data
    pub fn append(&mut self, keys: &[f32], values: &[f32]) -> Result<(), KvCacheError> {
        // make sure data is split into correct size vectors
        if keys.len() % self.key_dim != 0 || values.len() % self.value_dim != 0 {
            return Err(KvCacheError::InvalidShape);
        }

        let key_sequence_length = keys.len() / self.key_dim;
        let value_sequence_length = values.len() / self.value_dim;

        if key_sequence_length != value_sequence_length {
            return Err(KvCacheError::InvalidShape);
        }

        let new_seq_len = self
            .seq_len
            .checked_add(key_sequence_length)
            .ok_or(KvCacheError::CapacityOverflow)?;

        self.ensure_capacity(new_seq_len)?;

        let key_data_start = self
            .seq_len
            .checked_mul(self.key_dim)
            .ok_or(KvCacheError::CapacityOverflow)?;
        let value_data_start = self
            .seq_len
            .checked_mul(self.value_dim)
            .ok_or(KvCacheError::CapacityOverflow)?;

        // copy the data over
        self.keys[key_data_start..(key_data_start + keys.len())].copy_from_slice(keys);
        self.values[value_data_start..(value_data_start + values.len())].copy_from_slice(values);

        self.seq_len = new_seq_len;
        Ok(())
    }

    pub fn key_at(&self, token: usize, dim: usize) -> Option<f32> {
        if token >= self.seq_len || dim >= self.key_dim {
            return None;
        }

        self.keys.get(token * self.key_dim + dim).copied()
    }

    pub fn value_at(&self, token: usize, dim: usize) -> Option<f32> {
        if token >= self.seq_len || dim >= self.value_dim {
            return None;
        }

        self.values.get(token * self.value_dim + dim).copied()
    }
}

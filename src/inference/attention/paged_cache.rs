// src/inference/attention/paged_cache.rs

use super::error::PagedKvCacheError;

// A fixed-size block containing rows of key or value vectors.

// The data is stored contiguously in a single heap allocation.
// Each row contains `feature_dim` floating-point elements.
struct KvBlock {
    data: Vec<f32>,
    row_count: usize,
    feature_dim: usize,
}

impl KvBlock {
    // Creates a block containing zero-initialised elements.
    fn new(row_count: usize, feature_dim: usize) -> Result<Self, PagedKvCacheError> {
        let elements = row_count
            .checked_mul(feature_dim)
            .ok_or(PagedKvCacheError::ShapeOverflow)?;
        let mut data = Vec::new();

        data.try_reserve_exact(elements)
            .map_err(|_| PagedKvCacheError::AllocationFailed)?;

        data.resize(elements, 0.0);

        Ok(Self {
            data,
            row_count,
            feature_dim,
        })
    }

    // borrowed slice
    fn row(&self, row_offset: usize) -> Option<&[f32]> {
        if row_offset >= self.row_count {
            return None;
        }

        let start = row_offset.checked_mul(self.feature_dim)?;
        let end = start.checked_add(self.feature_dim)?;

        self.data.get(start..end)
    }

    // Copies a row into the block.
    fn write_row(&mut self, row_offset: usize, values: &[f32]) -> Option<()> {
        if row_offset >= self.row_count || values.len() != self.feature_dim {
            return None;
        }

        let start = row_offset.checked_mul(self.feature_dim)?;
        let end = start.checked_add(self.feature_dim)?;

        self.data.get_mut(start..end)?.copy_from_slice(values);
        Some(())
    }
}

// Stores KV entries in fixed-size blocks rather than one contiguous allocation.
pub struct PagedKvCache {
    key_dim: usize,
    value_dim: usize,
    block_size: usize,
    seq_len: usize,
    key_blocks: Vec<KvBlock>,
    value_blocks: Vec<KvBlock>,
}

impl PagedKvCache {
    pub fn new(
        key_dim: usize,
        value_dim: usize,
        block_size: usize,
    ) -> Result<Self, PagedKvCacheError> {
        if key_dim == 0 || value_dim == 0 || block_size == 0 {
            return Err(PagedKvCacheError::InvalidShape);
        }

        block_size
            .checked_mul(key_dim)
            .ok_or(PagedKvCacheError::ShapeOverflow)?;

        block_size
            .checked_mul(value_dim)
            .ok_or(PagedKvCacheError::ShapeOverflow)?;

        Ok(Self {
            key_dim,
            value_dim,
            block_size,
            seq_len: 0,
            key_blocks: Vec::new(),
            value_blocks: Vec::new(),
        })
    }

    pub fn key_dim(&self) -> usize {
        self.key_dim
    }

    pub fn value_dim(&self) -> usize {
        self.value_dim
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    pub fn num_blocks(&self) -> usize {
        self.key_blocks.len()
    }

    // Returns the block index and row offset for a cached token.
    pub fn block_location(&self, token_index: usize) -> Option<(usize, usize)> {
        if token_index >= self.seq_len {
            return None;
        }

        Some((token_index / self.block_size, token_index % self.block_size))
    }

    // Returns the key vector for a cached token.
    pub fn key_at(&self, token_index: usize) -> Option<&[f32]> {
        let (block_index, row_offset) = self.block_location(token_index)?;

        self.key_blocks
            .get(block_index)?
            .row(row_offset)
    }

    // Returns the value vector for a cached token.
    pub fn value_at(&self, token_index: usize) -> Option<&[f32]> {
        let (block_index, row_offset) = self.block_location(token_index)?;

        self.value_blocks
            .get(block_index)?
            .row(row_offset)
    }

    // Appends new key/value rows to the cache.
    // Allocates additional blocks only when the current blocks are full.
    pub fn append(&mut self, keys: &[f32], values: &[f32]) -> Result<(), PagedKvCacheError> {
        if keys.len() % self.key_dim != 0 || values.len() % self.value_dim != 0 {
            return Err(PagedKvCacheError::InvalidShape);
        }

        let key_sequence_length = keys.len() / self.key_dim;
        let value_sequence_length = values.len() / self.value_dim;

        if key_sequence_length != value_sequence_length {
            return Err(PagedKvCacheError::InvalidShape);
        }

        let rows = key_sequence_length;

        if rows == 0 {
            return Ok(());
        }

        let new_seq_len = self
            .seq_len
            .checked_add(rows)
            .ok_or(PagedKvCacheError::ShapeOverflow)?;

        let required_blocks = new_seq_len
            .checked_add(self.block_size - 1)
            .ok_or(PagedKvCacheError::ShapeOverflow)?
            / self.block_size;

        let additional_blocks = required_blocks - self.key_blocks.len();

        // Allocate all new blocks before modifying the logical cache contents.
        let mut new_key_blocks = Vec::new();
        let mut new_value_blocks = Vec::new();

        new_key_blocks
            .try_reserve_exact(additional_blocks)
            .map_err(|_| PagedKvCacheError::AllocationFailed)?;

        new_value_blocks
            .try_reserve_exact(additional_blocks)
            .map_err(|_| PagedKvCacheError::AllocationFailed)?;

        for _ in 0..additional_blocks {
            new_key_blocks.push(KvBlock::new(self.block_size, self.key_dim)?);
            new_value_blocks.push(KvBlock::new(self.block_size, self.value_dim)?);
        }

        self.key_blocks
            .try_reserve_exact(additional_blocks)
            .map_err(|_| PagedKvCacheError::AllocationFailed)?;

        self.value_blocks
            .try_reserve_exact(additional_blocks)
            .map_err(|_| PagedKvCacheError::AllocationFailed)?;

        self.key_blocks.extend(new_key_blocks);
        self.value_blocks.extend(new_value_blocks);

        for row in 0..rows {
            let token_index = self.seq_len + row;
            let block_index = token_index / self.block_size;
            let row_offset = token_index % self.block_size;

            let key_start = row * self.key_dim;
            let value_start = row * self.value_dim;

            self.key_blocks[block_index]
                .write_row(row_offset, &keys[key_start..key_start + self.key_dim])
                .ok_or(PagedKvCacheError::InvalidShape)?;

            self.value_blocks[block_index]
                .write_row(row_offset, &values[value_start..value_start + self.value_dim])
                .ok_or(PagedKvCacheError::InvalidShape)?;
        }

        self.seq_len = new_seq_len;

        Ok(())
    }

    // Returns the valid key data for a block, without copying.

    pub fn key_block(&self, block_index: usize) -> Option<&[f32]> {
        let valid_tokens = self.valid_tokens_in_block(block_index)?;
        let elements = valid_tokens.checked_mul(self.key_dim)?;

        self.key_blocks.get(block_index)?.data.get(..elements)
    }

    // Returns the valid value data for a block, without copying.

    // The returned slice contains `valid_tokens * value_dim` elements.
    / The final block may be only partially filled.
    pub fn value_block(&self, block_index: usize) -> Option<&[f32]> {
        let valid_tokens = self.valid_tokens_in_block(block_index)?;
        let elements = valid_tokens.checked_mul(self.value_dim)?;

        self.value_blocks.get(block_index)?.data.get(..elements)
    }

    /// Returns the number of valid tokens in a block.
    ///
    /// Returns None if the block index is invalid.
    pub fn valid_tokens_in_block(&self, block_index: usize) -> Option<usize> {
        if block_index >= self.num_blocks() {
            return None;
        }

        let remaining = self
            .seq_len
            .checked_sub(block_index.checked_mul(self.block_size)?)?;

        Some(remaining.min(self.block_size))
    }
}

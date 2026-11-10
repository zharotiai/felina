// src/tensor/basic_ops.rs

use super::{
    TILE_SIZE,
    types::{Tensor, TensorError, TensorView},
};

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, TensorError> {
        Self::validate_num_elements(&shape, data.len())?;
        Ok(Self { shape, data })
    }

    pub fn zeros(shape: Vec<usize>) -> Result<Self, TensorError> {
        Self::filled(shape, 0.0)
    }

    pub fn ones(shape: Vec<usize>) -> Result<Self, TensorError> {
        Self::filled(shape, 1.0)
    }

    fn filled(shape: Vec<usize>, value: f32) -> Result<Self, TensorError> {
        let n = Self::get_expected_num_elements(&shape)?;

        let mut data = Vec::new();
        // Preserve the Result-based allocation error instead of letting resize panic on OOM.
        data.try_reserve_exact(n)
            .map_err(|_| TensorError::AllocationFailed)?;

        data.resize(n, value);

        Ok(Self { shape, data })
    }

    pub(super) fn get_expected_num_elements(shape: &[usize]) -> Result<usize, TensorError> {
        // Shapes come from callers, so reject products that cannot be represented.
        shape
            .iter()
            // maybe the multiplication could overflow
            // handle that just in case
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or(TensorError::ShapeOverflow)
    }

    fn validate_num_elements(shape: &[usize], actual: usize) -> Result<(), TensorError> {
        let expected = Self::get_expected_num_elements(shape)?;

        if expected != actual {
            return Err(TensorError::ShapeMismatch { expected, actual });
        }

        Ok(())
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn num_elements(&self) -> usize {
        self.data.len()
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn get(&self, indices: &[usize]) -> Option<f32> {
        let offset = self.get_offset(indices)?;
        Some(self.data[offset])
    }

    pub fn set(&mut self, indices: &[usize], value: f32) -> Option<f32> {
        let offset = self.get_offset(indices)?;

        Some(std::mem::replace(&mut self.data[offset], value))
    }

    fn get_offset(&self, indices: &[usize]) -> Option<usize> {
        if indices.len() != self.shape.len() {
            return None;
        }

        let mut offset: usize = 0;

        for (&index, &dim) in indices.iter().zip(&self.shape) {
            if index >= dim {
                return None;
            }

            // Shape validation guarantees this offset fits in the backing data.
            offset = offset * dim + index;
        }

        Some(offset)
    }

    pub fn reshape(&mut self, new_shape: Vec<usize>) -> Result<(), TensorError> {
        Self::validate_num_elements(&new_shape, self.num_elements())?;
        self.shape = new_shape;
        Ok(())
    }

    // Copy-based transpose
    pub fn transpose(&self) -> Result<Self, TensorError> {
        if self.shape.len() != 2 {
            return Err(TensorError::InvalidDimensions {
                expected: 2,
                actual: self.shape.len(),
            });
        }

        let rows = self.shape[0];
        let cols = self.shape[1];

        let mut transposed = Self::zeros(vec![cols, rows])?;

        // A more safe approach might be to fill this with None as a sentinel
        // then input the values, and at the end we can see whether we missed anything
        // Although this might be overkill, the tests will test the correctness of this
        for row_start in (0..rows).step_by(TILE_SIZE) {
            for col_start in (0..cols).step_by(TILE_SIZE) {
                let row_end = (row_start + TILE_SIZE).min(rows);
                let col_end = (col_start + TILE_SIZE).min(cols);

                for row in row_start..row_end {
                    for col in col_start..col_end {
                        transposed.data[col * rows + row] = self.data[row * cols + col];
                    }
                }
            }
        }

        Ok(transposed)
    }

    pub fn view(&self) -> Result<TensorView<'_>, TensorError> {
        let strides = TensorView::contiguous_strides(&self.shape)?;

        Ok(TensorView {
            data: &self.data,
            shape: self.shape.clone(),
            strides,
            offset: 0,
        })
    }
}

impl<'a> TensorView<'a> {
    pub fn from_slice(data: &'a [f32], shape: Vec<usize>) -> Result<Self, TensorError> {
        Tensor::validate_num_elements(&shape, data.len())?;

        let strides = Self::contiguous_strides(&shape)?;

        Ok(Self {
            data,
            shape,
            strides,
            offset: 0,
        })
    }

    pub(super) fn contiguous_strides(shape: &[usize]) -> Result<Vec<usize>, TensorError> {
        // If we have a A x B x C X D tensor
        // the contiguous strides are [B * C * D, C * D, D, 1] for respective dimensions
        let mut strides = vec![0; shape.len()];
        let mut stride = 1usize;

        for i in (0..shape.len()).rev() {
            strides[i] = stride;

            stride *= shape[i];
        }

        Ok(strides)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn backing_data(&self) -> &'a [f32] {
        self.data
    }

    pub fn is_contiguous(&self) -> bool {
        // checks whether the view is the same as the one in memory
        // by validating the stride against the shape
        let mut expected_stride = 1usize;

        for i in (0..self.shape.len()).rev() {
            if self.shape[i] > 1 && self.strides[i] != expected_stride {
                return false;
            }

            expected_stride *= self.shape[i];
        }

        true
    }

    pub fn as_slice(&self) -> Option<&'a [f32]> {
        if !self.is_contiguous() {
            return None;
        }

        let end = self.offset + self.num_elements();
        self.data.get(self.offset..end)
    }

    pub fn get(&self, indices: &[usize]) -> Option<f32> {
        if indices.len() != self.shape.len() {
            return None;
        }

        let mut physical_offset = self.offset;

        for ((&index, &dim), &stride) in indices.iter().zip(&self.shape).zip(&self.strides) {
            // ((0, dim_0), stride_0)
            if index >= dim {
                return None;
            }

            physical_offset += index * stride;
        }

        self.data.get(physical_offset).copied()
    }

    pub fn transpose(&self) -> Result<Self, TensorError> {
        if self.shape.len() != 2 {
            return Err(TensorError::InvalidDimensions {
                expected: 2,
                actual: self.shape.len(),
            });
        }

        Ok(Self {
            data: self.data,
            shape: vec![self.shape[1], self.shape[0]],
            strides: vec![self.strides[1], self.strides[0]],
            offset: self.offset,
        })
    }

    // TODO: Potentially to_tensor and slice()
    // Slice would return a new view of the tensor from a different offset
}

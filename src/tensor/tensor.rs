use super::TILE_SIZE;
use std::fmt;

// Could make generic at some point:
// Tensor<T: Multiply>

// TODO: Could have types.rs and all the iplmementations in separated files
// like implementations in ops.rs

#[derive(Clone, PartialEq, Debug)]
pub struct Tensor {
    pub(super) shape: Vec<usize>,

    // Contiguous storage for cache locality.
    pub(super) data: Vec<f32>,
}

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
        data.try_reserve_exact(n)
            .map_err(|_| TensorError::AllocationFailed)?;
        data.resize(n, value);

        Ok(Self { shape, data })
    }

    pub(super) fn get_expected_num_elements(shape: &[usize]) -> Result<usize, TensorError> {
        shape
            .iter()
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

    // Replaces the value and returns the previous value.
    pub fn set(&mut self, indices: &[usize], value: f32) -> Option<f32> {
        let offset = self.get_offset(indices)?;

        Some(std::mem::replace(&mut self.data[offset], value))
    }

    fn get_offset(&self, indices: &[usize]) -> Option<usize> {
        if indices.len() != self.shape.len() {
            return None;
        }

        let mut offset = 0;

        for (&index, &dim) in indices.iter().zip(&self.shape) {
            if index >= dim {
                return None;
            }

            offset = offset * dim + index;
        }

        Some(offset)
    }

    // Changes the shape without changing the underlying data.
    pub fn reshape(&mut self, new_shape: Vec<usize>) -> Result<(), TensorError> {
        Self::validate_num_elements(&new_shape, self.num_elements())?;
        self.shape = new_shape;
        Ok(())
    }

    pub fn transpose(&self) -> Result<Self, TensorError> {
        if self.shape.len() != 2 {
            return Err(TensorError::InvalidDimensions {
                expected: 2,
                actual: self.shape.len(),
            });
        }

        let rows = self.shape[0];
        let cols = self.shape[1];

        let mut data = Vec::new();
        data.try_reserve_exact(self.data.len())
            .map_err(|_| TensorError::AllocationFailed)?;
        data.resize(self.data.len(), 0.0);

        // Used tiling for cache optimisation
        // Source: Cornell CS 3410: Cache Optimization
        // https://www.cs.cornell.edu/courses/cs3410/2024fa/assignments/cacheblock/instructions.html

        for row_start in (0..rows).step_by(TILE_SIZE) {
            for col_start in (0..cols).step_by(TILE_SIZE) {
                let row_end = (row_start + TILE_SIZE).min(rows);
                let col_end = (col_start + TILE_SIZE).min(cols);

                for row in row_start..row_end {
                    for col in col_start..col_end {
                        data[col * rows + row] = self.data[row * cols + col];
                    }
                }
            }
        }

        Self::new(data, vec![cols, rows])
    }
}

// A read-only view into tensor data.
//
// The view borrows the original allocation. Transposing or slicing a view
// changes its metadata without copying the underlying elements.
#[derive(Clone, Debug)]
pub struct TensorView<'a> {
    data: &'a [f32],
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

impl Tensor {
    /// Creates a view over this tensor without copying its data.
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
    /// Creates a view over borrowed, contiguous row-major data.
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

    /// Computes row-major strides for a contiguous tensor.
    fn contiguous_strides(shape: &[usize]) -> Result<Vec<usize>, TensorError> {
        let mut strides = vec![0; shape.len()];
        let mut stride = 1usize;

        for i in (0..shape.len()).rev() {
            strides[i] = stride;

            stride = stride
                .checked_mul(shape[i])
                .ok_or(TensorError::ShapeOverflow)?;
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
        // The shape is validated when the view is created.
        self.shape.iter().product()
    }

    /// Returns the entire backing allocation, not necessarily the
    /// logical elements of this view in their iteration order.
    pub fn backing_data(&self) -> &'a [f32] {
        self.data
    }

    /// Returns true when the logical elements occupy a contiguous
    /// row-major region of the backing allocation.
    pub fn is_contiguous(&self) -> bool {
        let mut expected_stride = 1usize;

        for i in (0..self.shape.len()).rev() {
            if self.shape[i] > 1 && self.strides[i] != expected_stride {
                return false;
            }

            match expected_stride.checked_mul(self.shape[i]) {
                Some(next) => expected_stride = next,
                None => return false,
            }
        }

        true
    }

    /// Returns a contiguous slice of the logical elements, if possible.
    pub fn as_slice(&self) -> Option<&'a [f32]> {
        if !self.is_contiguous() {
            return None;
        }

        let end = self.offset.checked_add(self.num_elements())?;
        self.data.get(self.offset..end)
    }

    /// Reads an element using logical coordinates.
    pub fn get(&self, indices: &[usize]) -> Option<f32> {
        if indices.len() != self.shape.len() {
            return None;
        }

        let mut physical_offset = self.offset;

        for ((&index, &dim), &stride) in indices.iter().zip(&self.shape).zip(&self.strides) {
            if index >= dim {
                return None;
            }

            physical_offset = physical_offset.checked_add(index.checked_mul(stride)?)?;
        }

        self.data.get(physical_offset).copied()
    }

    /// Transposes a two-dimensional view without copying its data.
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

    /// Selects the half-open range [start, end) along an axis.
    ///
    /// Returns None for an invalid axis or range.
    /// This operation does not copy tensor data.
    pub fn slice(&self, axis: usize, start: usize, end: usize) -> Option<Self> {
        if axis >= self.shape.len() || start > end || end > self.shape[axis] {
            return None;
        }

        let offset_delta = start.checked_mul(self.strides[axis])?;
        let offset = self.offset.checked_add(offset_delta)?;

        let mut shape = self.shape.clone();
        shape[axis] = end - start;

        Some(Self {
            data: self.data,
            shape,
            strides: self.strides.clone(),
            offset,
        })
    }

    /// Materializes this view into a new owned, contiguous tensor.
    pub fn to_tensor(&self) -> Result<Tensor, TensorError> {
        let n = self.num_elements();

        let mut data = Vec::new();
        data.try_reserve_exact(n)
            .map_err(|_| TensorError::AllocationFailed)?;

        // Convert each logical row-major index into physical coordinates.
        for linear_index in 0..n {
            let mut remainder = linear_index;
            let mut physical_offset = self.offset;

            for i in (0..self.shape.len()).rev() {
                let dim = self.shape[i];
                let coordinate = remainder % dim;
                remainder /= dim;

                physical_offset = physical_offset
                    .checked_add(
                        coordinate
                            .checked_mul(self.strides[i])
                            .ok_or(TensorError::ShapeOverflow)?,
                    )
                    .ok_or(TensorError::ShapeOverflow)?;
            }

            data.push(
                *self
                    .data
                    .get(physical_offset)
                    .ok_or(TensorError::ShapeOverflow)?,
            );
        }

        Tensor::new(data, self.shape.clone())
    }
}

#[derive(Debug)]
pub enum TensorError {
    ShapeOverflow,
    AllocationFailed,
    ShapeMismatch { expected: usize, actual: usize },
    InvalidDimensions { expected: usize, actual: usize },
    MatmulShapeMismatch { left: Vec<usize>, right: Vec<usize> },
    ElementwiseShapeMismatch { left: Vec<usize>, right: Vec<usize> },
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::ShapeOverflow => {
                write!(f, "Tensor shape exceeds the maximum representable size")
            }
            TensorError::AllocationFailed => {
                write!(f, "Failed to allocate memory for tensor")
            }
            TensorError::ShapeMismatch { expected, actual } => {
                write!(
                    f,
                    "Data length mismatch: expected {} elements from shape, got {}",
                    expected, actual
                )
            }
            TensorError::InvalidDimensions { expected, actual } => {
                write!(
                    f,
                    "Expected a tensor with {} dimensions, got {}",
                    expected, actual
                )
            }
            TensorError::MatmulShapeMismatch { left, right } => {
                write!(
                    f,
                    "Cannot multiply tensors with shapes {left:?} and {right:?}"
                )
            }
            TensorError::ElementwiseShapeMismatch { left, right } => write!(
                f,
                "Cannot perform elementwise operation on tensors with shapes {left:?} and {right:?}"
            ),
        }
    }
}

impl std::error::Error for TensorError {}

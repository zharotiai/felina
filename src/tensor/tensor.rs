
use std::fmt;
use super::TILE_SIZE;

// Could make generic at some point:
// Tensor<T: Multiply>

// TODO: Could have types.rs and all the iplmementations in separated files



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

    fn validate_num_elements(
        shape: &[usize],
        actual: usize,
    ) -> Result<(), TensorError> {
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

#[derive(Debug)]
pub enum TensorError {
    ShapeOverflow,
    AllocationFailed,
    ShapeMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidDimensions {
        expected: usize,
        actual: usize,
    },
    MatmulShapeMismatch {
        left: Vec<usize>,
        right: Vec<usize>,
    },
    ElementwiseShapeMismatch {
        left: Vec<usize>,
        right: Vec<usize>,
    },
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::ShapeOverflow => {
                write!(f, "Tensor shape exceeds the maximum representable size")
            },
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
            TensorError::ElementwiseShapeMismatch { left, right } => 
                write!(
                    f,
                    "Cannot perform elementwise operation on tensors with shapes {left:?} and {right:?}"
                ),
        }
    }
}

impl std::error::Error for TensorError {}

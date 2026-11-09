// src/tensor/types.rs

use std::fmt;

#[derive(Clone, PartialEq, Debug)]
pub struct Tensor {
    pub(super) shape: Vec<usize>,

    // Contiguous storage for cache locality.
    pub(super) data: Vec<f32>,
}

// A read-only view into tensor data.
//
// The view borrows the original allocation. Transposing or slicing a view
// changes its metadata without copying the underlying elements.
#[derive(Clone, Debug)]
pub struct TensorView<'a> {
    pub(super) data: &'a [f32],
    pub(super) shape: Vec<usize>,
    pub(super) strides: Vec<usize>,
    pub(super) offset: usize,
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

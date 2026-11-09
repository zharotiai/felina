// src/tensor/error.rs

use std::fmt;

use super::TensorError;

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
            TensorError::InsufficientDimensions { expected, actual } => {
                write!(
                    f,
                    "Expected at least {} dimension(s), got {}",
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

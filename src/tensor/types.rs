// src/tensor/types.rs

#[derive(Clone, PartialEq, Debug)]
pub struct Tensor {
    pub(super) shape: Vec<usize>,
    pub(super) data: Vec<f32>,
}

// A read-only view into tensor data.
//
// The view borrows the original allocation. Transposing or slicing a view
// changes its metadata without copying the underlying elements.
// Potentially for future operations where viewing is more efficient than copying,
// but not in use as of now.
#[derive(Clone, Debug)]
pub struct TensorView<'a> {
    pub(super) data: &'a [f32],
    pub(super) shape: Vec<usize>,

    // the data may be stored in memory with respect to a different shape
    // so the view need to know how to traverse the Tensor

    // For example if we have a 5x2 matrix, and the original stride is [2, 1]
    // but if we have a 2x5 view of it, then we still need to remember how to
    // traverse it with the [2, 1] stride, rather than inferring the new stride
    // which would be [5, 1]
    pub(super) strides: Vec<usize>,

    // where the view starts
    pub(super) offset: usize,
}

#[derive(Debug)]
pub enum TensorError {
    ShapeOverflow,
    AllocationFailed,
    ShapeMismatch { expected: usize, actual: usize },
    InvalidDimensions { expected: usize, actual: usize },
    InsufficientDimensions { expected: usize, actual: usize },
    MatmulShapeMismatch { left: Vec<usize>, right: Vec<usize> },
    ElementwiseShapeMismatch { left: Vec<usize>, right: Vec<usize> },
}

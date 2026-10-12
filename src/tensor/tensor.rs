
use std::fmt;

// Can make generic lter
// Maybe Tensor<T: Multiply> or something

pub struct Tensor {
    // Storing data in a contiguous Vec for cache locality
    data: Vec<f32>,

    // Using the shape to index into it.
    shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, TensorError> {
        let expected: usize = shape.iter().product();

        // Can't make a Tensor with a different shape to the data inputted
        if data.len() != expected {
            return Err(TensorError::ShapeMismatch {
                expected,
                actual: data.len(),
            });
        }

        Ok(Self {
            data,
            shape,
        })
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        todo!()
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        todo!()
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
}

#[derive(Debug)]
pub enum TensorError {
    ShapeOverflow,
    AllocationFailed,
    ShapeMismatch {
        expected: usize,
        actual: usize,
    },
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
        }
    }
}

impl std::error::Error for TensorError {}


use std::fmt;



// Could make generic at some point
// Tensor<T: Multiply>
pub struct Tensor {
    shape: Vec<usize>,

    // Contiguous storage for cache locality.
    data: Vec<f32>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, TensorError> {
        let expected = Self::get_expected_num_elements(&shape)?;

        // Ensure the data length matches the shape.
        if data.len() != expected {
            return Err(TensorError::ShapeMismatch {
                expected,
                actual: data.len(),
            });
        }

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

    fn get_expected_num_elements(shape: &[usize]) -> Result<usize, TensorError> {
        shape.iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or(TensorError::ShapeOverflow)
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

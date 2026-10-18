
use std::fmt;

// Could make generic at some point:
// Tensor<T: Multiply>

const TILE_SIZE: usize = 32;

#[derive(Clone, PartialEq, Debug)]
pub struct Tensor {
    shape: Vec<usize>,

    // Contiguous storage for cache locality.
    data: Vec<f32>,
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

    fn get_expected_num_elements(shape: &[usize]) -> Result<usize, TensorError> {
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

    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, TensorError> { 
        let left_shape = &self.shape; 
        let right_shape = &other.shape; 

        if left_shape.is_empty() || right_shape.is_empty() {
            return Err(TensorError::MatmulShapeMismatch {
                left: left_shape.clone(),
                right: right_shape.clone(),
            });
        }

        // Matching PyTorch semantics
        // Vectors are treated differently from n x 1 matrices
        // As are scalars different from 1x1 matrices
        let left_is_vector = left_shape.len() == 1;
        let right_is_vector = right_shape.len() == 1;

        // Treat vectors as matrices internally:
        // [K] becomes [1, K] on the left and [K, 1] on the right.

        // In general tensor multiplication
        // We contract the second last dimension of the right tensor
        // with the last dimension of the left one
        let (m, k) = if left_is_vector {
            (1, left_shape[0])
        } else {
            (
                left_shape[left_shape.len() - 2],
                left_shape[left_shape.len() - 1],
            )
        };

        let (other_k, n) = if right_is_vector {
            (right_shape[0], 1)
        } else {
            (
                right_shape[right_shape.len() - 2],
                right_shape[right_shape.len() - 1],
            )
        };

        if k != other_k {
            return Err(TensorError::MatmulShapeMismatch {
                left: left_shape.clone(),
                right: right_shape.clone(),
            });
        }

        // Dimensions preceding the final two are batch dimensions.
        let left_batch = if left_is_vector {
            &[][..]
        } else {
            &left_shape[..left_shape.len() - 2]
        };

        let right_batch = if right_is_vector {
            &[][..]
        } else {
            &right_shape[..right_shape.len() - 2]
        };

        // Batching treats everything as batches of 2d matrices

        // Right-align batch dimensions and apply broadcasting.
        let batch_shape =
            Self::broadcast_batch_shapes(left_batch, right_batch).ok_or_else(|| {
                TensorError::MatmulShapeMismatch {
                    left: left_shape.clone(),
                    right: right_shape.clone(),
                }
            })?;
        
        let mut output_shape = batch_shape.clone();
        output_shape.push(m);
        output_shape.push(n);

        let mut output = Self::zeros(output_shape)?;

        let batch_count = Self::get_expected_num_elements(&batch_shape)?;

        let left_matrix_size =
            m.checked_mul(k).ok_or(TensorError::ShapeOverflow)?;

        let right_matrix_size =
            k.checked_mul(n).ok_or(TensorError::ShapeOverflow)?;

        let output_matrix_size =
            m.checked_mul(n).ok_or(TensorError::ShapeOverflow)?;

        // Calculate each batch independently.
        for batch_idx in 0..batch_count {
            let left_batch_idx = Self::broadcast_batch_offset(
                batch_idx,
                &batch_shape,
                left_batch,
            );

            let right_batch_idx = Self::broadcast_batch_offset(
                batch_idx,
                &batch_shape,
                right_batch,
            );

            let left_base = left_batch_idx * left_matrix_size;
            let right_base = right_batch_idx * right_matrix_size;
            let output_base = batch_idx * output_matrix_size;

            // Tiled matrix multiplication.
            for row_start in (0..m).step_by(TILE_SIZE) {
                for col_start in (0..n).step_by(TILE_SIZE) {
                    for inner_start in (0..k).step_by(TILE_SIZE) {
                        let row_end = (row_start + TILE_SIZE).min(m);
                        let col_end = (col_start + TILE_SIZE).min(n);
                        let inner_end = (inner_start + TILE_SIZE).min(k);

                        for row in row_start..row_end {
                            for inner in inner_start..inner_end {
                                let a =
                                    self.data[left_base + row * k + inner];

                                for col in col_start..col_end {
                                    let b = other.data[
                                        right_base + inner * n + col
                                    ];

                                    output.data[
                                        output_base + row * n + col
                                    ] += a * b;
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut result_shape = batch_shape; 

        if !left_is_vector {
            result_shape.push(m);
        }

        if !right_is_vector {
            result_shape.push(n);
        }

        output.shape = result_shape;

        Ok(output)
    }

    fn broadcast_batch_shapes(
        left: &[usize], 
        right: &[usize],
    ) -> Option<Vec<usize>> {
        let rank = left.len().max(right.len());
        let mut result = vec![1; rank];

        for i in 0..rank {
            let left_idx = i.checked_sub(rank - left.len());
            let right_idx = i.checked_sub(rank - right.len());

            let a = left_idx.map_or(1, |idx| left[idx]);
            let b = right_idx.map_or(1, |idx| right[idx]);

            result[i] = if a == b {
                a
            } else if a == 1 {
                b
            } else if b == 1 {
                a
            } else {
                return None;
            };
        }

        Some(result)
    }


    fn broadcast_batch_offset(
        flat_index: usize,
        output_batch_shape: &[usize],
        input_batch_shape: &[usize],
    ) -> usize {
        // Decode the output batch's flat index into coordinates.
        let mut coordinates = vec![0; output_batch_shape.len()];
        let mut remaining = flat_index;

        for axis in (0..output_batch_shape.len()).rev() {
            coordinates[axis] = remaining % output_batch_shape[axis];
            remaining /= output_batch_shape[axis];
        }

        // Convert output coordinates into the input's batch offset.
        // Broadcast dimensions of size 1 always use coordinate 0.
        let mut input_offset = 0;
        let mut input_stride = 1;

        for input_axis in (0..input_batch_shape.len()).rev() {
            let output_axis =
                output_batch_shape.len() - input_batch_shape.len()
                    + input_axis;

            let coordinate = if input_batch_shape[input_axis] == 1 {
                0
            } else {
                coordinates[output_axis]
            };

            input_offset += coordinate * input_stride;
            input_stride *= input_batch_shape[input_axis];
        }

        input_offset
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
        }
    }
}

impl std::error::Error for TensorError {}

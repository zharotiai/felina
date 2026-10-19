
use super::{Tensor, TensorError, TILE_SIZE};

impl Tensor {
    /// Matrix multiplication using two tensors.
    ///
    /// Preserves the existing API while delegating to `matmul_slices`.
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        Self::matmul_slices(
            &self.data,
            &self.shape,
            &other.data,
            &other.shape,
        )
    }

    /// Matrix multiplication over borrowed, contiguous row-major data.
    ///
    /// The input slices are not copied. Only the output tensor is allocated.
    ///
    /// `left_shape` and `right_shape` describe the corresponding data slices.
    pub fn matmul_slices(
        left_data: &[f32],
        left_shape: &[usize],
        right_data: &[f32],
        right_shape: &[usize],
    ) -> Result<Tensor, TensorError> {
        if left_shape.is_empty() || right_shape.is_empty() {
            return Err(TensorError::MatmulShapeMismatch {
                left: left_shape.to_vec(),
                right: right_shape.to_vec(),
            });
        }

        // Validate that each slice contains exactly the number of elements
        // described by its shape.
        let expected_left = Self::get_expected_num_elements(left_shape)?;
        let expected_right = Self::get_expected_num_elements(right_shape)?;

        if left_data.len() != expected_left {
            return Err(TensorError::ShapeMismatch {
                expected: expected_left,
                actual: left_data.len(),
            });
        }

        if right_data.len() != expected_right {
            return Err(TensorError::ShapeMismatch {
                expected: expected_right,
                actual: right_data.len(),
            });
        }

        // Matching PyTorch semantics:
        // vectors are treated differently from n x 1 matrices.
        let left_is_vector = left_shape.len() == 1;
        let right_is_vector = right_shape.len() == 1;

        // Treat vectors as matrices internally:
        // [K] becomes [1, K] on the left and [K, 1] on the right.
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
                left: left_shape.to_vec(),
                right: right_shape.to_vec(),
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

        // Right-align batch dimensions and apply broadcasting.
        let batch_shape =
            Self::broadcast_batch_shapes(left_batch, right_batch).ok_or_else(
                || TensorError::MatmulShapeMismatch {
                    left: left_shape.to_vec(),
                    right: right_shape.to_vec(),
                },
            )?;

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
                                    left_data[left_base + row * k + inner];

                                for col in col_start..col_end {
                                    let b = right_data[
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

        // Restore vector-specific output dimensions.
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

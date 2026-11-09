// src/tensor/matmul.rs

use super::{TILE_SIZE, Tensor, TensorError};

impl Tensor {
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        Self::matmul_slices(&self.data, &self.shape, &other.data, &other.shape)
    }

    // Matrix multiplication over borrowed, contiguous row-major data.

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

        let left_is_vector = left_shape.len() == 1;
        let right_is_vector = right_shape.len() == 1;

        // Treat vectors as matrices internally to make multiplication cleaner
        // [K] becomes [1, K] on the left and [K, 1] on the right.

        // Modelling the matrix multiplication with dimensions:
        // (...., m, k) @ (....., other_k, n)

        // k must be equal to other_k
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

        // All other dimenstions are batch dimensions
        let left_batch = if left_is_vector {
            &[][..] // a slice of a reference to an empty array
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
            Self::broadcast_batch_shapes(left_batch, right_batch).ok_or_else(|| {
                TensorError::MatmulShapeMismatch {
                    left: left_shape.to_vec(),
                    right: right_shape.to_vec(),
                }
            })?;

        // output dims = [...(broadcast_dims)..., m, n]
        let mut output_shape = batch_shape.clone();
        output_shape.push(m);
        output_shape.push(n);

        let mut output = Self::zeros(output_shape)?;

        let batch_count = Self::get_expected_num_elements(&batch_shape)?;

        let left_batch_offsets: Vec<usize> = (0..batch_count)
            .map(|batch_idx| Self::broadcast_batch_offset(batch_idx, &batch_shape, left_batch))
            .collect();
        let right_batch_offsets: Vec<usize> = (0..batch_count)
            .map(|batch_idx| Self::broadcast_batch_offset(batch_idx, &batch_shape, right_batch))
            .collect();

        let left_matrix_size = m.checked_mul(k).ok_or(TensorError::ShapeOverflow)?;
        let right_matrix_size = k.checked_mul(n).ok_or(TensorError::ShapeOverflow)?;
        let output_matrix_size = m.checked_mul(n).ok_or(TensorError::ShapeOverflow)?;

        // Calculate each batch independently.
        for batch_idx in 0..batch_count {
            let left_batch_idx = left_batch_offsets[batch_idx];
            let right_batch_idx = right_batch_offsets[batch_idx];

            // on batch 0, we are starting from the start of the matrix
            // on batch 4 we are starting from 4 matrices after the start
            // remember everything is contiguous in memory at the end of the day
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

                        // Imagine einstein notation (summing over inner):
                        // output[row][col] = left_data[row][inner] * right_data[inner][col] 
                        for row in row_start..row_end {
                            for inner in inner_start..inner_end {
                                // row major layout, k elements in each row
                                // essentially you look at the next dimension
                                // for the cardinality of the current dimension
                                let a = left_data[left_base + row * k + inner];


                                for col in col_start..col_end {
                                    // n elements for each "inner"
                                    let b = right_data[right_base + inner * n + col];

                                    output.data[output_base + row * n + col] += a * b;
                                }
                            }
                        }
                    }
                }
            }

            // Equivalent readable version for one 2D matrix, but slower because
            // get and set validate indices and recalculate offsets each time:
            //
            // for row in 0..m {
            //     for inner in 0..k {
            //         let a = left.get(&[row, inner]).unwrap();
            //
            //         for col in 0..n {
            //             let b = right.get(&[inner, col]).unwrap();
            //             let current = output.get(&[row, col]).unwrap();
            //             output.set(&[row, col], current + a * b);
            //         }
            //     }
            // }
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

    fn broadcast_batch_shapes(left: &[usize], right: &[usize]) -> Option<Vec<usize>> {
        /*
        Two tensor shapes are compatible if after right aligning each pairs
        satisfies the following rules

        1. they are equal
        2. One of them is a 1
        3. One of them is missing (can put a 1 there)
        */

        let num_dims = left.len().max(right.len());
        let mut result = vec![1; num_dims];
        let rank = num_dims; // Declare rank based on the number of dimensions

        for i in 0..rank {
            let left_idx = i.checked_sub(rank - left.len());
            let right_idx = i.checked_sub(rank - right.len());

            let a = left_idx.map_or(1, |idx| left[idx]);
            let b = right_idx.map_or(1, |idx| right[idx]);

            // Implement the rules above
            result[i] = if a == b {
                a
            } else if a == 1 {
                b
            } else if b == 1 {
                a
            } else {
                // we have failed the broadcase
                return None;
            };
        }

        Some(result)
    }

    fn broadcast_batch_offset(
        flat_index: usize, // index of the current batch
        output_batch_shape: &[usize],
        input_batch_shape: &[usize],
    ) -> usize {
        // Example: output_batch_shape [2, 3], input_batch_shape [1, 3], flat_index = 4.
        let mut coordinates = vec![0; output_batch_shape.len()];
        let mut remaining = flat_index;

        // 4 -> [1, 1]
        for axis in (0..output_batch_shape.len()).rev() {
            coordinates[axis] = remaining % output_batch_shape[axis];
            remaining /= output_batch_shape[axis];
        }

        // Map output coordinates [1, 1] to input coordinates [0, 1].
        // The input's first dimension has size 1, so it is broadcast and
        // always uses coordinate 0.
        let mut input_offset = 0;
        let mut input_stride = 1;

        for input_axis in (0..input_batch_shape.len()).rev() {
            // handle right alignment
            // imagine:
            // output = [2, 3]
            // input = [3]
            // input_axis = 0
            // output_axis = 2 - 1 + 1 = 1
            let output_axis = output_batch_shape.len() - input_batch_shape.len() + input_axis;

            // use the output coord unless the input coord is 1
            let coordinate = if input_batch_shape[input_axis] == 1 {
                // 1 is broadcast
                0
            } else {
                coordinates[output_axis]
            };

            // Convert [0, 1] into the input's flat offset: 1.
            input_offset += coordinate * input_stride;
            input_stride *= input_batch_shape[input_axis];
        }

        // Output batch 4 therefore reuses input batch 1.
        input_offset
    }
}

// src/tensor/elementwise.rs

use super::{Tensor, TensorError};

impl Tensor {
    // Adds two tensors element by element.
    // Both tensors must have the same shape.
    pub fn add(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        self.elementwise_binary(other, |a, b| a + b)
    }

    // Multiplies two tensors element by element.
    // Both tensors must have the same shape.
    pub fn mul(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        self.elementwise_binary(other, |a, b| a * b)
    }

    // Multiplies every element by a scalar.
    pub fn scale(&self, scalar: f32) -> Result<Tensor, TensorError> {
        let mut data = Vec::with_capacity(self.data.len());

        for &value in &self.data {
            data.push(value * scalar);
        }

        Tensor::new(data, self.shape.clone())
    }

    // Sums all elements of the tensor.
    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    // Returns the largest element, or None if the tensor is empty.
    pub fn max(&self) -> Option<f32> {
        self.data.iter().copied().reduce(f32::max)
    }

    // Applies numerically stable softmax along the last dimension (rows for a 2d matrix)
    //
    // For a tensor with shape [d_0, d_1, ..., d_n], softmax is
    // applied independently to each group of d_n elements.
    pub fn softmax_last_dim(&self) -> Result<Tensor, TensorError> {
        let Some(&last_dim) = self.shape.last() else {
            return Err(TensorError::InsufficientDimensions {
                expected: 1,
                actual: 0,
            });
        };

        let mut data = Vec::with_capacity(self.data.len());

        // An empty last dimension means there are no elements to process.
        if last_dim == 0 {
            return Tensor::new(data, self.shape.clone());
        }

        // Each chunk represents one row along the last dimension.
        // Iterate over slices, each containing the last last_dim elements each
        for row in self.data.chunks(last_dim) {
            // Subtract the maximum to improve numerical stability.
            // Walks through the row one at a time, doesn't copy the full row
            let max_value = row.iter().copied().reduce(f32::max).unwrap();

            // where we start the next row
            let start = data.len();

            // Compute exp(x - max) for each element.
            let mut sum = 0.0;
            for &value in row {
                // subtracting the maximum here keeps it numerically stable
                let exp_value = (value - max_value).exp();
                data.push(exp_value);
                sum += exp_value;
            }

            // Normalise so the values in each row sum to approximately 1.
            for value in &mut data[start..] {
                *value /= sum;
            }
        }

        Tensor::new(data, self.shape.clone())
    }

    /// Shared implementation for elementwise binary operations.
    fn elementwise_binary(
        &self,
        other: &Tensor,
        operation: impl Fn(f32, f32) -> f32,
    ) -> Result<Tensor, TensorError> {
        if self.shape != other.shape {
            return Err(TensorError::ElementwiseShapeMismatch {
                left: self.shape.clone(),
                right: other.shape.clone(),
            });
        }

        let mut data = Vec::with_capacity(self.data.len());

        for (&a, &b) in self.data.iter().zip(&other.data) {
            data.push(operation(a, b));
        }

        Tensor::new(data, self.shape.clone())
    }
}

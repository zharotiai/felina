use felina::tensor::{Tensor, TensorError};

// Basic matrix multiplication

#[test]
fn matmul_multiplies_non_square_matrices() {
    // [2, 3] @ [3, 2] -> [2, 2]
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    let b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]).unwrap();

    let result = a.matmul(&b).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.data(), &[58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn matmul_multiplies_square_matrices() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();

    let result = a.matmul(&b).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.data(), &[19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn matmul_by_identity_preserves_matrix() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    let identity = Tensor::new(
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        vec![3, 3],
    )
    .unwrap();

    let result = a.matmul(&identity).unwrap();

    assert_eq!(result.shape(), a.shape());
    assert_eq!(result.data(), a.data());
}

// Vector semantics

#[test]
fn matmul_multiplies_two_vectors_into_a_scalar() {
    // [3] @ [3] -> scalar, shape []
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]).unwrap();

    let result = a.matmul(&b).unwrap();

    assert_eq!(result.shape(), &[]);
    assert_eq!(result.num_elements(), 1);
    assert_eq!(result.data(), &[32.0]);
}

#[test]
fn matmul_multiplies_matrix_by_vector() {
    // [2, 3] @ [3] -> [2]
    let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    let vector = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]).unwrap();

    let result = matrix.matmul(&vector).unwrap();

    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.data(), &[20.0, 47.0]);
}

#[test]
fn matmul_multiplies_vector_by_matrix() {
    // [2] @ [2, 3] -> [3]
    let vector = Tensor::new(vec![2.0, 3.0], vec![2]).unwrap();
    let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

    let result = vector.matmul(&matrix).unwrap();

    assert_eq!(result.shape(), &[3]);
    assert_eq!(result.data(), &[14.0, 19.0, 24.0]);
}

// Batched matrix multiplication

#[test]
fn matmul_multiplies_batched_matrices() {
    // [2, 2, 3] @ [2, 3, 2] -> [2, 2, 2]
    let a = Tensor::new(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        vec![2, 2, 3],
    )
    .unwrap();

    let b = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![2, 3, 2],
    )
    .unwrap();

    let result = a.matmul(&b).unwrap();

    assert_eq!(result.shape(), &[2, 2, 2]);
    assert_eq!(
        result.data(),
        &[22.0, 28.0, 49.0, 64.0, 16.0, 17.0, 22.0, 23.0,]
    );
}

#[test]
fn matmul_broadcasts_a_single_batch_dimension() {
    // [2, 2, 3] @ [3, 2] -> [2, 2, 2]
    let a = Tensor::new(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        vec![2, 2, 3],
    )
    .unwrap();

    let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();

    let result = a.matmul(&b).unwrap();

    assert_eq!(result.shape(), &[2, 2, 2]);
    assert_eq!(
        result.data(),
        &[22.0, 28.0, 49.0, 64.0, 76.0, 100.0, 103.0, 136.0,]
    );
}

#[test]
fn matmul_broadcasts_multiple_batch_dimensions() {
    // [2, 1, 2, 3] @ [1, 4, 3, 2] -> [2, 4, 2, 2]
    let a = Tensor::new(vec![1.0; 12], vec![2, 1, 2, 3]).unwrap();
    let b = Tensor::new(vec![1.0; 24], vec![1, 4, 3, 2]).unwrap();

    let result = a.matmul(&b).unwrap();

    assert_eq!(result.shape(), &[2, 4, 2, 2]);
    assert_eq!(result.num_elements(), 32);
    assert!(result.data().iter().all(|&x| x == 3.0));
}

// Invalid dimensions and shapes

#[test]
fn matmul_rejects_incompatible_inner_dimensions() {
    // [2, 3] @ [4, 2] is invalid because 3 != 4.
    let a = Tensor::new(vec![1.0; 6], vec![2, 3]).unwrap();
    let b = Tensor::new(vec![1.0; 8], vec![4, 2]).unwrap();

    assert!(matches!(
        a.matmul(&b),
        Err(TensorError::MatmulShapeMismatch { .. })
    ));
}

#[test]
fn matmul_rejects_incompatible_batch_dimensions() {
    // Batch dimensions 2 and 4 cannot broadcast.
    let a = Tensor::new(vec![1.0; 12], vec![2, 2, 3]).unwrap();
    let b = Tensor::new(vec![1.0; 24], vec![4, 3, 2]).unwrap();

    assert!(a.matmul(&b).is_err());
}

#[test]
fn matmul_rejects_scalar_operands() {
    let scalar = Tensor::new(vec![2.0], vec![]).unwrap();
    let matrix = Tensor::new(vec![1.0; 4], vec![2, 2]).unwrap();

    assert!(scalar.matmul(&matrix).is_err());
    assert!(matrix.matmul(&scalar).is_err());
}

#[test]
fn matmul_rejects_vectors_with_different_lengths() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    let b = Tensor::new(vec![4.0, 5.0], vec![2]).unwrap();

    assert!(a.matmul(&b).is_err());
}

// Degenerate dimensions

#[test]
fn matmul_handles_zero_inner_dimension() {
    // [2, 0] @ [0, 3] -> [2, 3], filled with zeros.
    let a = Tensor::new(vec![], vec![2, 0]).unwrap();
    let b = Tensor::new(vec![], vec![0, 3]).unwrap();

    let result = a.matmul(&b).unwrap();

    assert_eq!(result.shape(), &[2, 3]);
    assert_eq!(result.data(), &[0.0; 6]);
}

#[test]
fn matmul_handles_zero_sized_output() {
    // [0, 3] @ [3, 2] -> [0, 2].
    let a = Tensor::new(vec![], vec![0, 3]).unwrap();
    let b = Tensor::new(vec![1.0; 6], vec![3, 2]).unwrap();

    let result = a.matmul(&b).unwrap();

    assert_eq!(result.shape(), &[0, 2]);
    assert!(result.data().is_empty());
}

// Tile-boundary regression test

#[test]
fn matmul_handles_matrices_larger_than_one_tile() {
    // Dimensions exceed TILE_SIZE = 32.
    let n = 35;
    let a = Tensor::new(vec![1.0; n * n], vec![n, n]).unwrap();
    let b = Tensor::new(vec![2.0; n * n], vec![n, n]).unwrap();

    let result = a.matmul(&b).unwrap();

    assert_eq!(result.shape(), &[n, n]);
    assert!(result.data().iter().all(|&x| x == 70.0));
}

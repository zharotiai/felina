use felina::tensor::{Tensor, TensorError};

#[test]
fn test_add() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]).unwrap();

    let result = a.add(&b).unwrap();

    assert_eq!(result.shape(), &[3]);
    assert_eq!(result.data(), &[5.0, 7.0, 9.0]);
}

#[test]
fn test_mul() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]).unwrap();

    let result = a.mul(&b).unwrap();

    assert_eq!(result.data(), &[4.0, 10.0, 18.0]);
}

#[test]
fn test_add_multidimensional() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();

    let result = a.add(&b).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.data(), &[6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn test_elementwise_shape_mismatch() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();

    // Same number of elements, but different shapes.
    assert!(matches!(
        a.add(&b),
        Err(TensorError::ElementwiseShapeMismatch { .. })
    ));

    assert!(matches!(
        a.mul(&b),
        Err(TensorError::ElementwiseShapeMismatch { .. })
    ));
}

#[test]
fn test_scale() {
    let tensor = Tensor::new(vec![1.0, -2.0, 3.5], vec![3]).unwrap();

    let result = tensor.scale(2.0).unwrap();

    assert_eq!(result.shape(), &[3]);
    assert_eq!(result.data(), &[2.0, -4.0, 7.0]);
}

#[test]
fn test_scale_by_zero() {
    let tensor = Tensor::new(vec![1.0, -2.0, 3.0], vec![3]).unwrap();

    let result = tensor.scale(0.0).unwrap();

    assert_eq!(result.data(), &[0.0, 0.0, 0.0]);
}

#[test]
fn test_sum() {
    let tensor = Tensor::new(vec![1.0, -2.0, 3.0, 4.0], vec![2, 2]).unwrap();

    assert_eq!(tensor.sum(), 6.0);
}

#[test]
fn test_sum_empty_tensor() {
    let tensor = Tensor::new(vec![], vec![0]).unwrap();

    assert_eq!(tensor.sum(), 0.0);
}

#[test]
fn test_max() {
    let tensor = Tensor::new(vec![-5.0, 3.0, 1.0, 8.0], vec![4]).unwrap();

    assert_eq!(tensor.max(), Some(8.0));
}

#[test]
fn test_max_empty_tensor() {
    let tensor = Tensor::new(vec![], vec![0]).unwrap();

    assert_eq!(tensor.max(), None);
}

#[test]
fn test_softmax_vector() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();

    let result = tensor.softmax_last_dim().unwrap();
    let values = result.data();

    // Softmax values should sum to approximately 1.
    let sum: f32 = values.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);

    // Softmax preserves ordering.
    assert!(values[0] < values[1]);
    assert!(values[1] < values[2]);
}

#[test]
fn test_softmax_multidimensional() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], vec![2, 3]).unwrap();

    let result = tensor.softmax_last_dim().unwrap();
    let values = result.data();

    // Each row should independently sum to 1.
    for row in values.chunks(3) {
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    // Identical input rows should produce identical output rows.
    assert_eq!(&values[0..3], &values[3..6]);
}

#[test]
fn test_softmax_numerical_stability() {
    let tensor = Tensor::new(vec![1000.0, 1001.0, 1002.0], vec![3]).unwrap();

    let result = tensor.softmax_last_dim().unwrap();

    // All outputs should be finite, even for large input values.
    assert!(result.data().iter().all(|x| x.is_finite()));

    let sum: f32 = result.data().iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn test_softmax_empty_last_dimension() {
    let tensor = Tensor::new(vec![], vec![2, 0]).unwrap();

    let result = tensor.softmax_last_dim().unwrap();

    assert_eq!(result.shape(), &[2, 0]);
    assert!(result.data().is_empty());
}

#[test]
fn test_softmax_scalar_rejected() {
    // An empty shape represents a scalar in this tensor implementation.
    let tensor = Tensor::new(vec![5.0], vec![]).unwrap();

    assert!(matches!(
        tensor.softmax_last_dim(),
        Err(TensorError::InvalidDimensions { .. })
    ));
}

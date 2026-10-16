
use felina::tensor::{Tensor, TensorError};
use std::error::Error;

// --------------------------------------------------
// Tensor::new: valid construction
// --------------------------------------------------

#[test]
fn new_creates_one_dimensional_tensor() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();

    assert_eq!(tensor.shape(), &[3]);
    assert_eq!(tensor.num_elements(), 3);
    assert_eq!(tensor.data(), &[1.0, 2.0, 3.0]);
}

#[test]
fn new_creates_two_dimensional_tensor() {
    let tensor =
        Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

    assert_eq!(tensor.shape(), &[2, 3]);
    assert_eq!(tensor.num_elements(), 6);
    assert_eq!(tensor.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn new_creates_three_dimensional_tensor() {
    let values: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let tensor = Tensor::new(values.clone(), vec![2, 3, 4]).unwrap();

    assert_eq!(tensor.shape(), &[2, 3, 4]);
    assert_eq!(tensor.num_elements(), 24);
    assert_eq!(tensor.data(), values.as_slice());
}

#[test]
fn new_preserves_negative_values() {
    let tensor = Tensor::new(vec![-1.0, -2.5, 0.0, 3.5], vec![2, 2]).unwrap();

    assert_eq!(tensor.data(), &[-1.0, -2.5, 0.0, 3.5]);
}

#[test]
fn new_preserves_fractional_values() {
    let tensor =
        Tensor::new(vec![0.1, 0.25, 1.75], vec![3]).unwrap();

    assert_eq!(tensor.data(), &[0.1, 0.25, 1.75]);
}

#[test]
fn new_preserves_nan_and_infinity() {
    let values = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
    let tensor = Tensor::new(values, vec![3]).unwrap();

    assert!(tensor.data()[0].is_nan());
    assert_eq!(tensor.data()[1], f32::INFINITY);
    assert_eq!(tensor.data()[2], f32::NEG_INFINITY);
}

// --------------------------------------------------
// Tensor::new: shape validation
// --------------------------------------------------

#[test]
fn new_rejects_too_little_data() {
    let result = Tensor::new(vec![1.0, 2.0], vec![2, 2]);

    assert!(matches!(
        result,
        Err(TensorError::ShapeMismatch {
            expected: 4,
            actual: 2,
        })
    ));
}

#[test]
fn new_rejects_too_much_data() {
    let result = Tensor::new(vec![1.0, 2.0, 3.0], vec![2]);

    assert!(matches!(
        result,
        Err(TensorError::ShapeMismatch {
            expected: 2,
            actual: 3,
        })
    ));
}

#[test]
fn new_rejects_empty_data_for_nonempty_shape() {
    let result = Tensor::new(vec![], vec![2, 3]);

    assert!(matches!(
        result,
        Err(TensorError::ShapeMismatch {
            expected: 6,
            actual: 0,
        })
    ));
}

#[test]
fn new_accepts_empty_data_for_zero_sized_shape() {
    let tensor = Tensor::new(vec![], vec![2, 0, 3]).unwrap();

    assert_eq!(tensor.shape(), &[2, 0, 3]);
    assert_eq!(tensor.num_elements(), 0);
    assert!(tensor.data().is_empty());
}

#[test]
fn new_accepts_zero_dimension_at_start() {
    let tensor = Tensor::new(vec![], vec![0, 2, 3]).unwrap();

    assert_eq!(tensor.shape(), &[0, 2, 3]);
    assert_eq!(tensor.num_elements(), 0);
}

#[test]
fn new_accepts_zero_dimension_at_end() {
    let tensor = Tensor::new(vec![], vec![2, 3, 0]).unwrap();

    assert_eq!(tensor.shape(), &[2, 3, 0]);
    assert_eq!(tensor.num_elements(), 0);
}

#[test]
fn new_accepts_multiple_zero_dimensions() {
    let tensor = Tensor::new(vec![], vec![0, 2, 0, 3]).unwrap();

    assert_eq!(tensor.shape(), &[0, 2, 0, 3]);
    assert_eq!(tensor.num_elements(), 0);
}

#[test]
fn new_accepts_scalar_shape() {
    let tensor = Tensor::new(vec![42.0], vec![]).unwrap();

    assert_eq!(tensor.shape(), &[]);
    assert_eq!(tensor.num_elements(), 1);
    assert_eq!(tensor.data(), &[42.0]);
}

#[test]
fn new_rejects_empty_data_for_scalar_shape() {
    let result = Tensor::new(vec![], vec![]);

    assert!(matches!(
        result,
        Err(TensorError::ShapeMismatch {
            expected: 1,
            actual: 0,
        })
    ));
}

#[test]
fn new_rejects_shape_overflow() {
    let result = Tensor::new(vec![], vec![usize::MAX, 2]);

    assert!(matches!(
        result,
        Err(TensorError::ShapeOverflow)
    ));
}

#[test]
fn new_rejects_overflow_after_several_dimensions() {
    let result = Tensor::new(
        vec![],
        vec![2, usize::MAX, 2],
    );

    assert!(matches!(
        result,
        Err(TensorError::ShapeOverflow)
    ));
}

#[test]
fn new_accepts_shape_with_large_dimension_if_product_fits() {
    // The shape product fits in usize, even though allocating
    // a tensor of this size would be impractical.
    let result = Tensor::new(vec![], vec![usize::MAX]);

    assert!(matches!(
        result,
        Err(TensorError::ShapeMismatch {
            expected: usize::MAX,
            actual: 0,
        })
    ));
}

// --------------------------------------------------
// Tensor::zeros
// --------------------------------------------------

#[test]
fn zeros_creates_correct_shape() {
    let tensor = Tensor::zeros(vec![2, 3]).unwrap();

    assert_eq!(tensor.shape(), &[2, 3]);
}

#[test]
fn zeros_creates_correct_number_of_elements() {
    let tensor = Tensor::zeros(vec![2, 3, 4]).unwrap();

    assert_eq!(tensor.num_elements(), 24);
}

#[test]
fn zeros_initialises_every_element_to_zero() {
    let tensor = Tensor::zeros(vec![3, 4]).unwrap();

    assert!(tensor.data().iter().all(|&x| x == 0.0));
}

#[test]
fn zeros_creates_one_dimensional_tensor() {
    let tensor = Tensor::zeros(vec![4]).unwrap();

    assert_eq!(tensor.data(), &[0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn zeros_creates_scalar() {
    let tensor = Tensor::zeros(vec![]).unwrap();

    assert_eq!(tensor.shape(), &[]);
    assert_eq!(tensor.data(), &[0.0]);
}

#[test]
fn zeros_supports_zero_sized_dimensions() {
    let tensor = Tensor::zeros(vec![2, 0, 3]).unwrap();

    assert_eq!(tensor.shape(), &[2, 0, 3]);
    assert!(tensor.data().is_empty());
}

#[test]
fn zeros_rejects_shape_overflow() {
    let result = Tensor::zeros(vec![usize::MAX, 2]);

    assert!(matches!(
        result,
        Err(TensorError::ShapeOverflow)
    ));
}

// --------------------------------------------------
// Tensor::ones
// --------------------------------------------------

#[test]
fn ones_creates_correct_shape() {
    let tensor = Tensor::ones(vec![2, 3]).unwrap();

    assert_eq!(tensor.shape(), &[2, 3]);
}

#[test]
fn ones_creates_correct_number_of_elements() {
    let tensor = Tensor::ones(vec![2, 3, 4]).unwrap();

    assert_eq!(tensor.num_elements(), 24);
}

#[test]
fn ones_initialises_every_element_to_one() {
    let tensor = Tensor::ones(vec![3, 4]).unwrap();

    assert!(tensor.data().iter().all(|&x| x == 1.0));
}

#[test]
fn ones_creates_one_dimensional_tensor() {
    let tensor = Tensor::ones(vec![3]).unwrap();

    assert_eq!(tensor.data(), &[1.0, 1.0, 1.0]);
}

#[test]
fn ones_creates_scalar() {
    let tensor = Tensor::ones(vec![]).unwrap();

    assert_eq!(tensor.shape(), &[]);
    assert_eq!(tensor.data(), &[1.0]);
}

#[test]
fn ones_supports_zero_sized_dimensions() {
    let tensor = Tensor::ones(vec![2, 0, 3]).unwrap();

    assert_eq!(tensor.shape(), &[2, 0, 3]);
    assert!(tensor.data().is_empty());
}

#[test]
fn ones_rejects_shape_overflow() {
    let result = Tensor::ones(vec![usize::MAX, 2]);

    assert!(matches!(
        result,
        Err(TensorError::ShapeOverflow)
    ));
}

// --------------------------------------------------
// Accessors
// --------------------------------------------------

#[test]
fn shape_returns_dimensions_in_order() {
    let tensor = Tensor::zeros(vec![2, 3, 4]).unwrap();

    assert_eq!(tensor.shape(), &[2, 3, 4]);
}

#[test]
fn num_elements_matches_data_length() {
    let tensor =
        Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

    assert_eq!(tensor.num_elements(), tensor.data().len());
}

#[test]
fn data_preserves_row_major_storage_order() {
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::new(values.clone(), vec![2, 3]).unwrap();

    assert_eq!(tensor.data(), values.as_slice());
}

#[test]
fn data_returns_empty_slice_for_empty_tensor() {
    let tensor = Tensor::zeros(vec![0, 5]).unwrap();

    assert!(tensor.data().is_empty());
    assert_eq!(tensor.num_elements(), 0);
}

#[test]
fn accessors_do_not_change_tensor() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();

    let _ = tensor.shape();
    let _ = tensor.data();
    let _ = tensor.num_elements();

    assert_eq!(tensor.shape(), &[3]);
    assert_eq!(tensor.data(), &[1.0, 2.0, 3.0]);
}

// --------------------------------------------------
// Error types and messages
// --------------------------------------------------

#[test]
fn shape_overflow_has_display_message() {
    let error = TensorError::ShapeOverflow;

    assert_eq!(
        error.to_string(),
        "Tensor shape exceeds the maximum representable size"
    );
}

#[test]
fn allocation_failed_has_display_message() {
    let error = TensorError::AllocationFailed;

    assert_eq!(
        error.to_string(),
        "Failed to allocate memory for tensor"
    );
}

#[test]
fn shape_mismatch_has_display_message() {
    let error = TensorError::ShapeMismatch {
        expected: 6,
        actual: 4,
    };

    assert_eq!(
        error.to_string(),
        "Data length mismatch: expected 6 elements from shape, got 4"
    );
}

#[test]
fn tensor_error_implements_debug() {
    let error = TensorError::ShapeOverflow;

    let debug = format!("{error:?}");

    assert!(debug.contains("ShapeOverflow"));
}

#[test]
fn tensor_error_implements_standard_error() {
    let error = TensorError::ShapeOverflow;
    let _: &dyn Error = &error;
}

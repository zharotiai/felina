use felina::tensor::{Tensor, TensorError};
use std::error::Error;

// Construction and shape validation

#[test]
fn new_preserves_shape_and_row_major_data() {
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::new(values.clone(), vec![2, 3]).unwrap();

    assert_eq!(tensor.shape(), &[2, 3]);
    assert_eq!(tensor.num_elements(), 6);
    assert_eq!(tensor.data(), values.as_slice());
}

#[test]
fn new_supports_higher_rank_tensors() {
    let values: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let tensor = Tensor::new(values.clone(), vec![2, 3, 4]).unwrap();

    assert_eq!(tensor.shape(), &[2, 3, 4]);
    assert_eq!(tensor.num_elements(), 24);
    assert_eq!(tensor.data(), values.as_slice());
}

#[test]
fn new_preserves_negative_fractional_and_non_finite_values() {
    let tensor = Tensor::new(
        vec![-2.5, 0.25, f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
        vec![5],
    )
    .unwrap();

    assert_eq!(tensor.data()[0], -2.5);
    assert_eq!(tensor.data()[1], 0.25);
    assert!(tensor.data()[2].is_nan());
    assert_eq!(tensor.data()[3], f32::INFINITY);
    assert_eq!(tensor.data()[4], f32::NEG_INFINITY);
}

#[test]
fn new_rejects_data_length_mismatches() {
    for (data, shape, expected, actual) in [
        (vec![1.0, 2.0], vec![2, 2], 4, 2),
        (vec![1.0, 2.0, 3.0], vec![2], 2, 3),
        (vec![], vec![2, 3], 6, 0),
        (vec![], vec![], 1, 0),
    ] {
        assert!(matches!(
            Tensor::new(data, shape),
            Err(TensorError::ShapeMismatch { expected: e, actual: a })
                if e == expected && a == actual
        ));
    }
}

#[test]
fn new_accepts_scalar_and_zero_sized_shapes() {
    let scalar = Tensor::new(vec![42.0], vec![]).unwrap();
    assert_eq!(scalar.shape(), &[]);
    assert_eq!(scalar.num_elements(), 1);
    assert_eq!(scalar.data(), &[42.0]);

    for shape in [vec![2, 0, 3], vec![0, 2, 3], vec![2, 3, 0], vec![0, 2, 0]] {
        let tensor = Tensor::new(vec![], shape.clone()).unwrap();
        assert_eq!(tensor.shape(), shape.as_slice());
        assert_eq!(tensor.num_elements(), 0);
        assert!(tensor.data().is_empty());
    }
}

#[test]
fn new_reports_shape_product_overflow() {
    for shape in [vec![usize::MAX, 2], vec![2, usize::MAX, 2]] {
        assert!(matches!(
            Tensor::new(vec![], shape),
            Err(TensorError::ShapeOverflow)
        ));
    }
}

#[test]
fn new_distinguishes_large_valid_shape_from_overflow() {
    // The product fits in usize, but the provided data length does not match.
    assert!(matches!(
        Tensor::new(vec![], vec![usize::MAX]),
        Err(TensorError::ShapeMismatch {
            expected: usize::MAX,
            actual: 0
        })
    ));
}

// Constructors

#[test]
fn zeros_initialises_requested_shape_and_values() {
    let tensor = Tensor::zeros(vec![2, 3, 4]).unwrap();

    assert_eq!(tensor.shape(), &[2, 3, 4]);
    assert_eq!(tensor.num_elements(), 24);
    assert!(tensor.data().iter().all(|&x| x == 0.0));
}

#[test]
fn ones_initialises_requested_shape_and_values() {
    let tensor = Tensor::ones(vec![2, 3]).unwrap();

    assert_eq!(tensor.shape(), &[2, 3]);
    assert_eq!(tensor.num_elements(), 6);
    assert!(tensor.data().iter().all(|&x| x == 1.0));
}

#[test]
fn zeros_and_ones_support_scalar_and_empty_shapes() {
    let zero_scalar = Tensor::zeros(vec![]).unwrap();
    let one_scalar = Tensor::ones(vec![]).unwrap();
    assert_eq!(zero_scalar.data(), &[0.0]);
    assert_eq!(one_scalar.data(), &[1.0]);

    let zero_empty = Tensor::zeros(vec![2, 0, 3]).unwrap();
    let one_empty = Tensor::ones(vec![2, 0, 3]).unwrap();
    assert_eq!(zero_empty.shape(), &[2, 0, 3]);
    assert_eq!(one_empty.shape(), &[2, 0, 3]);
    assert!(zero_empty.data().is_empty());
    assert!(one_empty.data().is_empty());
}

#[test]
fn constructors_report_shape_overflow() {
    assert!(matches!(
        Tensor::zeros(vec![usize::MAX, 2]),
        Err(TensorError::ShapeOverflow)
    ));
    assert!(matches!(
        Tensor::ones(vec![usize::MAX, 2]),
        Err(TensorError::ShapeOverflow)
    ));
}

// Accessors and indexing

#[test]
fn accessors_do_not_mutate_tensor() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();

    let _ = tensor.shape();
    let _ = tensor.data();
    let _ = tensor.num_elements();

    assert_eq!(tensor.shape(), &[3]);
    assert_eq!(tensor.data(), &[1.0, 2.0, 3.0]);
    assert_eq!(tensor.num_elements(), 3);
}

#[test]
fn get_reads_values_at_multidimensional_indices() {
    let tensor = Tensor::new(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![2, 3]).unwrap();

    assert_eq!(tensor.get(&[0, 0]), Some(10.0));
    assert_eq!(tensor.get(&[0, 2]), Some(30.0));
    assert_eq!(tensor.get(&[1, 0]), Some(40.0));
    assert_eq!(tensor.get(&[1, 2]), Some(60.0));
}

#[test]
fn get_returns_none_for_invalid_indices() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

    assert_eq!(tensor.get(&[2, 0]), None); // Out of bounds.
    assert_eq!(tensor.get(&[0, 2]), None); // Out of bounds.
    assert_eq!(tensor.get(&[0]), None); // Wrong rank.
    assert_eq!(tensor.get(&[0, 0, 0]), None); // Wrong rank.
}

#[test]
fn set_updates_only_the_selected_element() {
    let mut tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

    tensor.set(&[1, 0], 99.0);

    assert_eq!(tensor.data(), &[1.0, 2.0, 99.0, 4.0]);
    assert_eq!(tensor.get(&[1, 0]), Some(99.0));
}

// Clone and equality

#[test]
fn clone_copies_shape_and_data_independently() {
    let mut original = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let cloned = original.clone();

    original.set(&[0, 0], 99.0);

    assert_eq!(cloned.shape(), &[2, 2]);
    assert_eq!(cloned.data(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(original.get(&[0, 0]), Some(99.0));
    assert_eq!(cloned.get(&[0, 0]), Some(1.0));
}

#[test]
fn equality_compares_both_shape_and_data() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let same = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let different_data = Tensor::new(vec![1.0, 2.0, 3.0, 5.0], vec![2, 2]).unwrap();
    let different_shape = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();

    assert_eq!(a, same);
    assert_ne!(a, different_data);
    assert_ne!(a, different_shape);
}

// Transpose

#[test]
fn transpose_reorders_non_square_matrix() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

    let transposed = tensor.transpose().unwrap();

    assert_eq!(transposed.shape(), &[3, 2]);
    assert_eq!(transposed.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert_eq!(tensor.shape(), &[2, 3]);
    assert_eq!(tensor.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn transpose_handles_square_and_degenerate_matrices() {
    let square = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])
        .unwrap()
        .transpose()
        .unwrap();
    assert_eq!(square.shape(), &[2, 2]);
    assert_eq!(square.data(), &[1.0, 3.0, 2.0, 4.0]);

    let row = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3])
        .unwrap()
        .transpose()
        .unwrap();
    assert_eq!(row.shape(), &[3, 1]);
    assert_eq!(row.data(), &[1.0, 2.0, 3.0]);

    let empty = Tensor::new(vec![], vec![0, 3])
        .unwrap()
        .transpose()
        .unwrap();
    assert_eq!(empty.shape(), &[3, 0]);
    assert!(empty.data().is_empty());
}

// Resilience / scale regression tests

#[test]
fn large_tensor_construction_preserves_length_and_endpoints() {
    // A deterministic scale test, not a wall-clock benchmark.
    let len = 1_000_000;
    let values: Vec<f32> = (0..len).map(|i| i as f32).collect();
    let tensor = Tensor::new(values, vec![1_000, 1_000]).unwrap();

    assert_eq!(tensor.num_elements(), len);
    assert_eq!(tensor.data().first(), Some(&0.0));
    assert_eq!(tensor.data().last(), Some(&999_999.0));
}

#[test]
fn large_tensor_transpose_preserves_element_count_and_samples() {
    // Catches indexing/stride regressions at a larger size without flaky timing assertions.
    let rows = 256;
    let cols = 384;
    let values: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
    let tensor = Tensor::new(values, vec![rows, cols]).unwrap();

    let transposed = tensor.transpose().unwrap();

    assert_eq!(transposed.shape(), &[cols, rows]);
    assert_eq!(transposed.num_elements(), rows * cols);
    assert_eq!(transposed.get(&[0, 0]), Some(0.0));
    assert_eq!(
        transposed.get(&[cols - 1, rows - 1]),
        Some((rows * cols - 1) as f32)
    );
    assert_eq!(transposed.get(&[17, 23]), Some((23 * cols + 17) as f32));
}

// Error contract

#[test]
fn tensor_errors_have_stable_display_messages() {
    assert_eq!(
        TensorError::ShapeOverflow.to_string(),
        "Tensor shape exceeds the maximum representable size"
    );
    assert_eq!(
        TensorError::AllocationFailed.to_string(),
        "Failed to allocate memory for tensor"
    );
    assert_eq!(
        TensorError::ShapeMismatch {
            expected: 6,
            actual: 4
        }
        .to_string(),
        "Data length mismatch: expected 6 elements from shape, got 4"
    );
}

#[test]
fn tensor_error_implements_standard_error() {
    let error = TensorError::ShapeOverflow;
    let _: &dyn Error = &error;
}

#[test]
fn view_does_not_copy_data() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

    let view = tensor.view().unwrap();

    assert_eq!(view.shape(), &[2, 3]);
    assert_eq!(view.strides(), &[3, 1]);
    assert!(std::ptr::eq(
        view.backing_data().as_ptr(),
        tensor.data().as_ptr(),
    ));
}

#[test]
fn transposed_view_does_not_copy_data() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

    let view = tensor.view().unwrap();
    let transposed = view.transpose().unwrap();

    assert_eq!(transposed.shape(), &[3, 2]);
    assert_eq!(transposed.strides(), &[1, 3]);
    assert_eq!(transposed.get(&[0, 1]), Some(4.0));
    assert_eq!(transposed.get(&[2, 1]), Some(6.0));
    assert!(transposed.as_slice().is_none());
}

#[test]
fn sliced_view_reads_correct_elements() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

    let view = tensor.view().unwrap();
    let sliced = view.slice(1, 1, 3).unwrap();

    assert_eq!(sliced.shape(), &[2, 2]);
    assert_eq!(sliced.get(&[0, 0]), Some(2.0));
    assert_eq!(sliced.get(&[0, 1]), Some(3.0));
    assert_eq!(sliced.get(&[1, 0]), Some(5.0));
    assert_eq!(sliced.get(&[1, 1]), Some(6.0));
}

#[test]
fn materializing_transposed_view_copies_in_logical_order() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

    let transposed = tensor.view().unwrap().transpose().unwrap();
    let owned = transposed.to_tensor().unwrap();

    assert_eq!(owned.shape(), &[3, 2]);
    assert_eq!(owned.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

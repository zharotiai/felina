use felina::inference::attention::{
    AttentionBackend,
    AttentionInput,
    AttentionMask,
    StandardAttention,
};

use felina::tensor::Tensor;

#[test]
fn attention_computes_expected_output() {
    // Q = [[1, 0]]
    let q = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();

    // K = [[1, 0], [0, 1]]
    let k = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();

    // V = [[10, 0], [0, 10]]
    let v = Tensor::new(vec![10.0, 0.0, 0.0, 10.0], vec![2, 2]).unwrap();

    let attention = StandardAttention;

    let output = attention
        .forward(AttentionInput {
            q: &q,
            k: &k,
            v: &v,
            mask: AttentionMask::None,
        })
        .unwrap();

    // Scores are [1, 0] / sqrt(2).
    // Softmax weights are approximately [0.6698, 0.3302].
    assert_eq!(output.shape(), &[1, 2]);
    assert!((output.data()[0] - 6.698).abs() < 0.01);
    assert!((output.data()[1] - 3.302).abs() < 0.01);
}

#[test]
fn attention_handles_single_token() {
    let q = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
    let k = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
    let v = Tensor::new(vec![5.0, 7.0], vec![1, 2]).unwrap();

    let output = StandardAttention
        .forward(AttentionInput {
            q: &q,
            k: &k,
            v: &v,
            mask: AttentionMask::None,
        })
        .unwrap();

    // Softmax of a single score is 1, so output must equal V.
    assert_eq!(output.shape(), &[1, 2]);
    assert!((output.data()[0] - 5.0).abs() < 1e-6);
    assert!((output.data()[1] - 7.0).abs() < 1e-6);
}

#[test]
fn attention_rejects_incompatible_query_and_key_dimensions() {
    let q = Tensor::ones(vec![1, 3]).unwrap();
    let k = Tensor::ones(vec![2, 4]).unwrap();
    let v = Tensor::ones(vec![2, 2]).unwrap();

    let result = StandardAttention.forward(AttentionInput {
        q: &q,
        k: &k,
        v: &v,
        mask: AttentionMask::None,
    });

    assert!(result.is_err());
}

#[test]
fn attention_rejects_incompatible_key_and_value_lengths() {
    let q = Tensor::ones(vec![1, 2]).unwrap();
    let k = Tensor::ones(vec![3, 2]).unwrap();
    let v = Tensor::ones(vec![2, 2]).unwrap();

    let result = StandardAttention.forward(AttentionInput {
        q: &q,
        k: &k,
        v: &v,
        mask: AttentionMask::None,
    });

    assert!(result.is_err());
}

#[test]
fn causal_attention_ignores_future_tokens() {
    let q = Tensor::new(vec![1.0, 0.0, 1.0, 0.0], vec![2, 2]).unwrap();
    let k = Tensor::new(
        vec![1.0, 0.0, 1.0, 0.0],
        vec![2, 2],
    )
    .unwrap();

    // Change the second value token dramatically.
    let v1 = Tensor::new(vec![2.0, 4.0, 100.0, 200.0], vec![2, 2]).unwrap();
    let v2 = Tensor::new(vec![2.0, 4.0, -100.0, -200.0], vec![2, 2]).unwrap();

    let output1 = StandardAttention
        .forward(AttentionInput {
            q: &q,
            k: &k,
            v: &v1,
            mask: AttentionMask::Causal,
        })
        .unwrap();

    let output2 = StandardAttention
        .forward(AttentionInput {
            q: &q,
            k: &k,
            v: &v2,
            mask: AttentionMask::Causal,
        })
        .unwrap();

    // At query position 0, future key position 1 must be masked.
    assert!((output1.data()[0] - output2.data()[0]).abs() < 1e-6);
    assert!((output1.data()[1] - output2.data()[1]).abs() < 1e-6);
}

#[test]
fn attention_output_is_finite_for_large_inputs() {
    let q = Tensor::new(vec![1000.0, 1000.0], vec![1, 2]).unwrap();
    let k = Tensor::new(
        vec![1000.0, 1000.0, 999.0, 999.0],
        vec![2, 2],
    )
    .unwrap();
    let v = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

    let output = StandardAttention
        .forward(AttentionInput {
            q: &q,
            k: &k,
            v: &v,
            mask: AttentionMask::None,
        })
        .unwrap();

    assert!(output.data().iter().all(|x| x.is_finite()));
}

#[test]
fn attention_output_shape_matches_query_and_value_dimensions() {
    let q = Tensor::ones(vec![3, 4]).unwrap();
    let k = Tensor::ones(vec![5, 4]).unwrap();
    let v = Tensor::ones(vec![5, 7]).unwrap();

    let output = StandardAttention
        .forward(AttentionInput {
            q: &q,
            k: &k,
            v: &v,
            mask: AttentionMask::None,
        })
        .unwrap();

    assert_eq!(output.shape(), &[3, 7]);
}
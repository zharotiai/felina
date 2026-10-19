use felina::inference::attention::{
    AttentionBackend,
    AttentionInput,
    AttentionMask,
    CachedAttention,
    StandardAttention,
};
use felina::tensor::Tensor;

fn make_cached_attention() -> CachedAttention {
    CachedAttention::with_capacity_defaults(2, 2).unwrap()
}

fn test_attention_computes_expected_output<A: AttentionBackend>(
    mut attention: A,
) {
    let q = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
    let k = Tensor::new(
        vec![1.0, 0.0, 0.0, 1.0],
        vec![2, 2],
    )
    .unwrap();
    let v = Tensor::new(
        vec![2.0, 4.0, 6.0, 8.0],
        vec![2, 2],
    )
    .unwrap();

    let output = attention
        .forward(AttentionInput {
            q: &q,
            k: &k,
            v: &v,
            mask: AttentionMask::None,
        })
        .unwrap();

    assert_eq!(output.shape(), &[1, 2]);
    assert!(output.data()[0].is_finite());
    assert!(output.data()[1].is_finite());

    // The first key has a higher attention score, so the output
    // should be closer to the first value vector.
    assert!(output.data()[0] > 2.0);
    assert!(output.data()[0] < 6.0);
    assert!(output.data()[1] > 4.0);
    assert!(output.data()[1] < 8.0);
}

#[test]
fn standard_attention_computes_expected_output() {
    test_attention_computes_expected_output(StandardAttention);
}

#[test]
fn cached_attention_computes_expected_output() {
    test_attention_computes_expected_output(make_cached_attention());
}

fn test_single_token_attention<A: AttentionBackend>(
    mut attention: A,
) {
    let q = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
    let k = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
    let v = Tensor::new(vec![3.0, 7.0], vec![1, 2]).unwrap();

    let output = attention
        .forward(AttentionInput {
            q: &q,
            k: &k,
            v: &v,
            mask: AttentionMask::None,
        })
        .unwrap();

    assert_eq!(output.shape(), &[1, 2]);
    assert!((output.data()[0] - 3.0).abs() < 1e-6);
    assert!((output.data()[1] - 7.0).abs() < 1e-6);
}

#[test]
fn standard_attention_handles_single_token() {
    test_single_token_attention(StandardAttention);
}

#[test]
fn cached_attention_handles_single_token() {
    test_single_token_attention(make_cached_attention());
}

fn test_attention_produces_finite_output<A: AttentionBackend>(
    mut attention: A,
) {
    let q = Tensor::new(
        vec![1.0, 0.0, 0.0, 1.0],
        vec![2, 2],
    )
    .unwrap();
    let k = Tensor::new(
        vec![1.0, 0.0, 0.0, 1.0],
        vec![2, 2],
    )
    .unwrap();
    let v = Tensor::new(
        vec![2.0, 4.0, 6.0, 8.0],
        vec![2, 2],
    )
    .unwrap();

    let output = attention
        .forward(AttentionInput {
            q: &q,
            k: &k,
            v: &v,
            mask: AttentionMask::None,
        })
        .unwrap();

    assert_eq!(output.shape(), &[2, 2]);
    assert!(output.data().iter().all(|x| x.is_finite()));
}

#[test]
fn standard_attention_produces_finite_output() {
    test_attention_produces_finite_output(StandardAttention);
}

#[test]
fn cached_attention_produces_finite_output() {
    test_attention_produces_finite_output(make_cached_attention());
}

fn test_attention_rejects_mismatched_dimensions<A: AttentionBackend>(
    mut attention: A,
) {
    let q = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
    let k = Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap();
    let v = Tensor::new(vec![2.0, 4.0], vec![1, 2]).unwrap();

    let result = attention.forward(AttentionInput {
        q: &q,
        k: &k,
        v: &v,
        mask: AttentionMask::None,
    });

    assert!(result.is_err());
}

#[test]
fn standard_attention_rejects_mismatched_dimensions() {
    test_attention_rejects_mismatched_dimensions(StandardAttention);
}

#[test]
fn cached_attention_rejects_mismatched_dimensions() {
    test_attention_rejects_mismatched_dimensions(make_cached_attention());
}

fn test_causal_attention_ignores_future_tokens<A: AttentionBackend>(
    mut make_attention: impl FnMut() -> A,
) {
    let q = Tensor::new(
        vec![1.0, 0.0, 1.0, 0.0],
        vec![2, 2],
    )
    .unwrap();
    let k = Tensor::new(
        vec![1.0, 0.0, 1.0, 0.0],
        vec![2, 2],
    )
    .unwrap();

    let v1 = Tensor::new(
        vec![2.0, 4.0, 100.0, 200.0],
        vec![2, 2],
    )
    .unwrap();
    let v2 = Tensor::new(
        vec![2.0, 4.0, -100.0, -200.0],
        vec![2, 2],
    )
    .unwrap();

    // Use independent backends so CachedAttention's state from one
    // forward call cannot affect the other.
    let output1 = make_attention()
        .forward(AttentionInput {
            q: &q,
            k: &k,
            v: &v1,
            mask: AttentionMask::Causal,
        })
        .unwrap();

    let output2 = make_attention()
        .forward(AttentionInput {
            q: &q,
            k: &k,
            v: &v2,
            mask: AttentionMask::Causal,
        })
        .unwrap();

    // The first query must not attend to the second (future) token.
    assert!((output1.data()[0] - output2.data()[0]).abs() < 1e-6);
    assert!((output1.data()[1] - output2.data()[1]).abs() < 1e-6);
}

#[test]
fn standard_causal_attention_ignores_future_tokens() {
    test_causal_attention_ignores_future_tokens(|| StandardAttention);
}

#[test]
fn cached_causal_attention_ignores_future_tokens() {
    test_causal_attention_ignores_future_tokens(make_cached_attention);
}
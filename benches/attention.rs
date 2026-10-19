use criterion::{
    criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
use std::hint::black_box;

use felina::inference::attention::{
    AttentionBackend, AttentionInput, AttentionMask, CachedAttention, PagedAttention,
    StandardAttention,
};
use felina::tensor::Tensor;

// Projection-plus-attention benchmark for autoregressive decoding.
//
// uncached_standard recomputes K/V projections for the entire growing context
// on every generated token.
// cached projects the prompt once, then projects only each new token and stores
// K/V in the contiguous KV cache.
// paged_* does the same projection work as cached, but stores K/V in the
// paged cache with the requested block size.
//
// Prompt prefill is excluded from the timed interval for cached and paged
// variants. This measures decode-phase cost after the prompt has been cached.
// Inputs and projection matrices are identical across all variants.

const HIDDEN_SIZE: usize = 128;
const HEAD_DIM: usize = 64;
const GENERATED_TOKENS: usize = 16;
const CONTEXT_LENGTHS: [usize; 4] = [16, 64, 128, 256];
const BLOCK_SIZES: [usize; 3] = [16, 32, 64];

fn deterministic_data(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as u32)
                .wrapping_mul(1_664_525)
                .wrapping_add(seed)
                .wrapping_add(1_013_904_223);
            ((x % 2_000) as f32 - 1_000.0) / 1_000.0
        })
        .collect()
}

struct Inputs {
    // Prompt hidden states followed by hidden states for generated tokens.
    hidden_states: Vec<f32>,
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    context_tokens: usize,
}

impl Inputs {
    fn new(context_tokens: usize) -> Self {
        let total_tokens = context_tokens + GENERATED_TOKENS;
        Self {
            hidden_states: deterministic_data(total_tokens * HIDDEN_SIZE, 7),
            wq: Tensor::new(
                deterministic_data(HIDDEN_SIZE * HEAD_DIM, 11),
                vec![HIDDEN_SIZE, HEAD_DIM],
            )
            .expect("valid Wq"),
            wk: Tensor::new(
                deterministic_data(HIDDEN_SIZE * HEAD_DIM, 13),
                vec![HIDDEN_SIZE, HEAD_DIM],
            )
            .expect("valid Wk"),
            wv: Tensor::new(
                deterministic_data(HIDDEN_SIZE * HEAD_DIM, 17),
                vec![HIDDEN_SIZE, HEAD_DIM],
            )
            .expect("valid Wv"),
            context_tokens,
        }
    }

    fn hidden_tensor(&self, token_count: usize) -> Tensor {
        Tensor::new(
            self.hidden_states[..token_count * HIDDEN_SIZE].to_vec(),
            vec![token_count, HIDDEN_SIZE],
        )
        .expect("valid hidden-state tensor")
    }

    fn current_hidden_tensor(&self, token_index: usize) -> Tensor {
        let start = token_index * HIDDEN_SIZE;
        Tensor::new(
            self.hidden_states[start..start + HIDDEN_SIZE].to_vec(),
            vec![1, HIDDEN_SIZE],
        )
        .expect("valid single-token tensor")
    }

    fn prefix_kv(&self) -> (Tensor, Tensor) {
        let prompt = self.hidden_tensor(self.context_tokens);
        let keys = prompt.matmul(&self.wk).expect("prompt K projection");
        let values = prompt.matmul(&self.wv).expect("prompt V projection");
        (keys, values)
    }
}

fn project_q(hidden: &Tensor, wq: &Tensor) -> Tensor {
    hidden.matmul(wq).expect("Q projection")
}

/// No cache: every decode step recomputes K and V for the complete prefix.
fn decode_without_kv_cache(inputs: &Inputs) {
    for step in 0..GENERATED_TOKENS {
        let token_count = inputs.context_tokens + step + 1;
        let all_hidden = inputs.hidden_tensor(token_count);
        let keys = all_hidden.matmul(&inputs.wk).expect("K projection");
        let values = all_hidden.matmul(&inputs.wv).expect("V projection");

        let current = inputs.current_hidden_tensor(token_count - 1);
        let q = project_q(&current, &inputs.wq);
        let mut attention = StandardAttention;
        let output = attention
            .forward(AttentionInput {
                q: black_box(&q),
                k: black_box(&keys),
                v: black_box(&values),
                mask: AttentionMask::Causal,
            })
            .expect("standard attention");
        black_box(output);
    }
}

/// Cached decode: K/V for the prompt are already in the cache. Only the new
/// token's K/V are projected at each step.
fn decode_with_kv_cache(inputs: &Inputs, attention: &mut CachedAttention) {
    for step in 0..GENERATED_TOKENS {
        let token_index = inputs.context_tokens + step;
        let current = inputs.current_hidden_tensor(token_index);
        let q = project_q(&current, &inputs.wq);
        let k = current.matmul(&inputs.wk).expect("new K projection");
        let v = current.matmul(&inputs.wv).expect("new V projection");

        let output = attention
            .forward(AttentionInput {
                q: black_box(&q),
                k: black_box(&k),
                v: black_box(&v),
                mask: AttentionMask::Causal,
            })
            .expect("cached attention");
        black_box(output);
    }
}

/// Paged cached decode: same projection work as contiguous caching, but K/V
/// are stored and read through PagedAttention's block-based cache.
fn decode_with_paged_kv_cache(inputs: &Inputs, attention: &mut PagedAttention) {
    for step in 0..GENERATED_TOKENS {
        let token_index = inputs.context_tokens + step;
        let current = inputs.current_hidden_tensor(token_index);
        let q = project_q(&current, &inputs.wq);
        let k = current.matmul(&inputs.wk).expect("new K projection");
        let v = current.matmul(&inputs.wv).expect("new V projection");

        let output = attention
            .forward(AttentionInput {
                q: black_box(&q),
                k: black_box(&k),
                v: black_box(&v),
                mask: AttentionMask::Causal,
            })
            .expect("paged attention");
        black_box(output);
    }
}

fn kv_cache_benefit(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_cache_projection_benefit");
    group.sample_size(10);
    group.throughput(Throughput::Elements(GENERATED_TOKENS as u64));

    for context_tokens in CONTEXT_LENGTHS {
        let inputs = Inputs::new(context_tokens);
        // This prompt projection is shared setup for the cached variants and
        // is deliberately outside the measured decode interval.
        let (prefix_k, prefix_v) = inputs.prefix_kv();

        group.bench_with_input(
            BenchmarkId::new("uncached_standard", context_tokens),
            &context_tokens,
            |b, _| {
                b.iter(|| decode_without_kv_cache(black_box(&inputs)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("cached_contiguous", context_tokens),
            &context_tokens,
            |b, _| {
                b.iter_batched(
                    || {
                        let mut attention =
                            CachedAttention::with_capacity_defaults(HEAD_DIM, HEAD_DIM)
                                .expect("create KV cache");
                        attention
                            .cache_mut()
                            .append(&prefix_k, &prefix_v)
                            .expect("prefill contiguous cache");
                        attention
                    },
                    |mut attention| {
                        decode_with_kv_cache(black_box(&inputs), &mut attention);
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        for block_size in BLOCK_SIZES {
            group.bench_with_input(
                BenchmarkId::new(format!("paged_block_{block_size}"), context_tokens),
                &context_tokens,
                |b, _| {
                    b.iter_batched(
                        || {
                            let mut attention = PagedAttention::with_block_size(
                                HEAD_DIM,
                                HEAD_DIM,
                                block_size,
                            )
                            .expect("create paged KV cache");
                            attention
                                .cache_mut()
                                .append(&prefix_k, &prefix_v)
                                .expect("prefill paged cache");
                            attention
                        },
                        |mut attention| {
                            decode_with_paged_kv_cache(black_box(&inputs), &mut attention);
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, kv_cache_benefit);
criterion_main!(benches);

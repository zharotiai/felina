// benches/attention.rs

use criterion::{
    BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
};
use std::hint::black_box;

use felina::inference::attention::{
    AttentionBackend, AttentionInput, AttentionMask, CachedAttention, PagedAttention,
    StandardAttention,
};
use felina::tensor::Tensor;

const HIDDEN_SIZE: usize = 128;
const HEAD_DIM: usize = 64;
const GENERATED_TOKENS: usize = 16;
const CONTEXT_LENGTHS: [usize; 6] = [16, 64, 128, 256, 512, 1024];
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

/// Recomputes K/V projections for the entire growing prefix.
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

/// Reuses a contiguous KV cache and projects only each new token.
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

/// Reuses a paged KV cache and projects only each new token.
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

// Reports allocated KV payload capacity after each token.
//
// This reports capacity-derived payload bytes, not process RSS or allocator
// overhead. The standard cache allocates 512 slots initially and grows when
// needed. The paged cache allocates whole blocks on demand.
fn report_memory_growth() {
    let bytes_per_token =
        (HEAD_DIM + HEAD_DIM) * std::mem::size_of::<f32>();

    println!("\nKV cache allocated payload during decode");
    println!("Bytes per token (K + V): {bytes_per_token}");
    println!(
        "{:>7} {:>7} {:>12} {:>12} {:>12} {:>12}",
        "Context",
        "Tokens",
        "Std slots",
        "Std KiB",
        "Paged blocks",
        "Paged KiB"
    );

    for context_tokens in CONTEXT_LENGTHS {
        let inputs = Inputs::new(context_tokens);
        let (prefix_k, prefix_v) = inputs.prefix_kv();

        let mut standard =
            CachedAttention::with_capacity_defaults(HEAD_DIM, HEAD_DIM)
                .expect("create contiguous cache");

        standard
            .cache_mut()
            .append(&prefix_k, &prefix_v)
            .expect("prefill contiguous cache");

        // Use block size 16 for the token-by-token memory report.
        let block_size = 16;
        let mut paged =
            PagedAttention::with_block_size(HEAD_DIM, HEAD_DIM, block_size)
                .expect("create paged cache");

        paged
            .cache_mut()
            .append(&prefix_k, &prefix_v)
            .expect("prefill paged cache");

        print_memory_row(
            context_tokens,
            context_tokens,
            standard.cache().capacity(),
            paged.cache().num_blocks(),
            block_size,
            bytes_per_token,
        );

        for step in 0..GENERATED_TOKENS {
            let token_index = context_tokens + step;
            let current = inputs.current_hidden_tensor(token_index);
            let k = current.matmul(&inputs.wk).expect("new K projection");
            let v = current.matmul(&inputs.wv).expect("new V projection");

            standard
                .cache_mut()
                .append(&k, &v)
                .expect("append contiguous cache");

            paged
                .cache_mut()
                .append(&k, &v)
                .expect("append paged cache");

            print_memory_row(
                context_tokens,
                token_index + 1,
                standard.cache().capacity(),
                paged.cache().num_blocks(),
                block_size,
                bytes_per_token,
            );
        }
    }
}

fn print_memory_row(
    context: usize,
    tokens: usize,
    standard_slots: usize,
    paged_blocks: usize,
    block_size: usize,
    bytes_per_token: usize,
) {
    let standard_bytes = standard_slots * bytes_per_token;
    let paged_bytes = paged_blocks * block_size * bytes_per_token;

    println!(
        "{:>7} {:>7} {:>12} {:>12.1} {:>12} {:>12.1}",
        context,
        tokens,
        standard_slots,
        standard_bytes as f64 / 1024.0,
        paged_blocks,
        paged_bytes as f64 / 1024.0,
    );
}

fn kv_cache_benefit(c: &mut Criterion) {
    report_memory_growth();

    let mut group = c.benchmark_group("kv_cache_projection_benefit");
    group.sample_size(10);
    group.throughput(Throughput::Elements(GENERATED_TOKENS as u64));

    for context_tokens in CONTEXT_LENGTHS {
        let inputs = Inputs::new(context_tokens);
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
                BenchmarkId::new(
                    format!("paged_block_{block_size}"),
                    context_tokens,
                ),
                &context_tokens,
                |b, _| {
                    b.iter_batched(
                        || {
                            let mut attention =
                                PagedAttention::with_block_size(
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

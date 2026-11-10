// benches/attention.rs

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use felina::inference::attention::{
    AttentionBackend, AttentionInput, AttentionMask, CachedAttention, PagedAttention,
    StandardAttention,
};
use felina::tensor::Tensor;

// Number of distinct token IDs the model can represent.
const VOCAB_SIZE: usize = 256;

// Width of each token's hidden representation.
const HIDDEN_SIZE: usize = 128;

// Number of values in each attention head's key and query vectors.
const HEAD_DIM: usize = 64;
const GENERATED_TOKENS: usize = 16;
const CONTEXT_LENGTHS: [usize; 6] = [16, 64, 128, 256, 512, 1024];
const BLOCK_SIZES: [usize; 3] = [16, 32, 64];
const INITIAL_KV_CACHE_CAPACITY: usize = 512;


// Pseudo-random looking vector used for model weights
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

// Pseudo random looking vector of token ids, used for input
fn deterministic_tokens(len: usize) -> Vec<usize> {
    (0..len)
        .map(|i| {
            let x = (i as u32)
                .wrapping_mul(1_664_525)
                .wrapping_add(29)
                .wrapping_add(1_013_904_223);

            (x as usize) % VOCAB_SIZE
        })
        .collect()
}

struct Inputs {
    // The input context for the toy language model.
    context_tokens: Vec<usize>,

    // Model weights.

    // toy embedding for the context tokens
    token_embedding: Tensor,

    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    wo: Tensor,
    lm_head: Tensor,
}

impl Inputs {
    fn new(context_length: usize) -> Self {
        Self {
            context_tokens: deterministic_tokens(context_length),

            // [vocab_size, hidden_size]
            token_embedding: Tensor::new(
                deterministic_data(VOCAB_SIZE * HIDDEN_SIZE, 7),
                vec![VOCAB_SIZE, HIDDEN_SIZE],
            )
            .expect("valid token embedding"),

            // [hidden_size, head_dim]
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

            // [head_dim, hidden_size]
            wo: Tensor::new(
                deterministic_data(HEAD_DIM * HIDDEN_SIZE, 19),
                vec![HEAD_DIM, HIDDEN_SIZE],
            )
            .expect("valid Wo"),

            // [hidden_size, vocab_size]
            lm_head: Tensor::new(
                deterministic_data(HIDDEN_SIZE * VOCAB_SIZE, 23),
                vec![HIDDEN_SIZE, VOCAB_SIZE],
            )
            .expect("valid LM head"),
        }
    }

    // Looks up a single token embedding.
    // token ID -> [1, hidden_size]
    fn embed_token(&self, token_id: usize) -> Tensor {
        let start = token_id * HIDDEN_SIZE;

        Tensor::new(
            self.token_embedding.data()[start..start + HIDDEN_SIZE].to_vec(),
            vec![1, HIDDEN_SIZE],
        )
        .expect("valid token embedding")
    }

    // Embeds the prefix of the context, excluding the final token.
    // The final context token is processed as the first decode step.
    fn prefix_hidden(&self) -> Tensor {
        let prefix_len = self.context_tokens.len() - 1;
        let mut data = Vec::with_capacity(prefix_len * HIDDEN_SIZE);

        for &token_id in &self.context_tokens[..prefix_len] {
            let start = token_id * HIDDEN_SIZE;

            data.extend_from_slice(&self.token_embedding.data()[start..start + HIDDEN_SIZE]);
        }

        Tensor::new(data, vec![prefix_len, HIDDEN_SIZE]).expect("valid prefix hidden states")
    }

    // Returns the final token in the context as the first current token.
    fn initial_current(&self) -> Tensor {
        let token_id = *self
            .context_tokens
            .last()
            .expect("context must contain at least one token");

        self.embed_token(token_id)
    }

    // Projects the context prefix into K/V for cache prefill.
    fn prefix_kv(&self) -> (Tensor, Tensor) {
        let prefix = self.prefix_hidden();

        let keys = prefix.matmul(&self.wk).expect("prefix K projection");

        let values = prefix.matmul(&self.wv).expect("prefix V projection");

        (keys, values)
    }
}

fn project_q(hidden: &Tensor, wq: &Tensor) -> Tensor {
    hidden.matmul(wq).expect("Q projection")
}

// Maps an attention output back into hidden-state space.
fn project_output(attention_output: &Tensor, wo: &Tensor) -> Tensor {
    attention_output
        .matmul(wo)
        .expect("attention output projection")
}

// Converts a hidden state into vocabulary logits and greedily selects
// the next token.
fn next_token(hidden: &Tensor, lm_head: &Tensor) -> usize {
    let logits = hidden.matmul(lm_head).expect("LM head projection");

    logits
        .data()
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            left.partial_cmp(right)
                .expect("logits must not contain NaN")
        })
        .map(|(index, _)| index)
        .expect("vocabulary must not be empty")
}

// Runs autoregressive inference without a KV cache.
// K and V are recomputed for the entire growing context on every token.
fn decode_without_kv_cache(inputs: &Inputs) {
    let prefix = inputs.prefix_hidden();
    let mut history = prefix.data().to_vec();

    let mut current = inputs.initial_current();
    let mut attention = StandardAttention;

    for _ in 0..GENERATED_TOKENS {
        // Add the current token to the model context.
        history.extend_from_slice(current.data());

        let token_count = history.len() / HIDDEN_SIZE;

        let all_hidden = Tensor::new(history.clone(), vec![token_count, HIDDEN_SIZE])
            .expect("valid hidden-state history");

        // Without a KV cache, recompute K and V for the whole context.
        let keys = all_hidden.matmul(&inputs.wk).expect("K projection");

        let values = all_hidden.matmul(&inputs.wv).expect("V projection");

        let q = project_q(&current, &inputs.wq);

        let attention_output = attention
            .forward(AttentionInput {
                q: black_box(&q),
                k: black_box(&keys),
                v: black_box(&values),
                mask: AttentionMask::Causal,
            })
            .expect("standard attention");

        let hidden = project_output(&attention_output, &inputs.wo);

        // hidden state -> vocabulary logits -> next token ID.
        let token_id = next_token(&hidden, &inputs.lm_head);

        // The generated token becomes the input to the next decode step.
        current = inputs.embed_token(token_id);
    }

    black_box(current);
}

// Runs token-in, token-out autoregressive inference with a contiguous KV cache.
fn decode_with_kv_cache(inputs: &Inputs, attention: &mut CachedAttention) {
    let mut current = inputs.initial_current();

    for _ in 0..GENERATED_TOKENS {
        let q = project_q(&current, &inputs.wq);
        let k = current.matmul(&inputs.wk).expect("new K projection");

        let v = current.matmul(&inputs.wv).expect("new V projection");

        let attention_output = attention
            .forward(AttentionInput {
                q: black_box(&q),
                k: black_box(&k),
                v: black_box(&v),
                mask: AttentionMask::Causal,
            })
            .expect("cached attention");

        let hidden = project_output(&attention_output, &inputs.wo);

        let token_id = next_token(&hidden, &inputs.lm_head);

        current = inputs.embed_token(token_id);
    }

    black_box(current);
}

// Runs token-in, token-out autoregressive inference with a paged KV cache.
fn decode_with_paged_kv_cache(inputs: &Inputs, attention: &mut PagedAttention) {
    let mut current = inputs.initial_current();

    for _ in 0..GENERATED_TOKENS {
        let q = project_q(&current, &inputs.wq);
        let k = current.matmul(&inputs.wk).expect("new K projection");

        let v = current.matmul(&inputs.wv).expect("new V projection");

        let attention_output = attention
            .forward(AttentionInput {
                q: black_box(&q),
                k: black_box(&k),
                v: black_box(&v),
                mask: AttentionMask::Causal,
            })
            .expect("paged attention");

        let hidden = project_output(&attention_output, &inputs.wo);

        let token_id = next_token(&hidden, &inputs.lm_head);

        current = inputs.embed_token(token_id);
    }

    black_box(current);
}

// Reports allocated KV payload capacity after each token.
//
// This reports capacity-derived payload bytes, not process RSS or allocator
// overhead. The standard cache allocates an initial contiguous capacity and
// grows when needed. The paged cache allocates whole blocks on demand.
fn report_memory_growth() {
    let bytes_per_token = (HEAD_DIM + HEAD_DIM) * std::mem::size_of::<f32>();

    println!("\nKV cache allocated payload during decode");
    println!("Bytes per token (K + V): {bytes_per_token}");
    println!(
        "{:>7} {:>7} {:>12} {:>12} {:>12} {:>12}",
        "Context", "Tokens", "Std slots", "Std KiB", "Paged blocks", "Paged KiB"
    );

    for context_length in CONTEXT_LENGTHS {
        let inputs = Inputs::new(context_length);
        let (prefix_k, prefix_v) = inputs.prefix_kv();

        let mut standard = CachedAttention::with_capacity_defaults(
            HEAD_DIM,
            HEAD_DIM,
            INITIAL_KV_CACHE_CAPACITY,
        )
            .expect("create contiguous cache");

        standard
            .cache_mut()
            .append(prefix_k.data(), prefix_v.data())
            .expect("prefill contiguous cache");

        let block_size = 16;

        let mut paged = PagedAttention::with_block_size(HEAD_DIM, HEAD_DIM, block_size)
            .expect("create paged cache");

        paged
            .cache_mut()
            .append(prefix_k.data(), prefix_v.data())
            .expect("prefill paged cache");

        print_memory_row(
            context_length,
            context_length - 1,
            standard.cache().capacity(),
            paged.cache().num_blocks(),
            block_size,
            bytes_per_token,
        );

        // Memory growth depends on the number of generated tokens, not
        // their values, so use the same token-in/token-out decode structure.
        let mut current = inputs.initial_current();
        let mut standard_attention = standard;
        let mut paged_attention = paged;

        for step in 0..GENERATED_TOKENS {
            let q = project_q(&current, &inputs.wq);
            let k = current.matmul(&inputs.wk).expect("new K projection");

            let v = current.matmul(&inputs.wv).expect("new V projection");

            let standard_output = standard_attention
                .forward(AttentionInput {
                    q: &q,
                    k: &k,
                    v: &v,
                    mask: AttentionMask::Causal,
                })
                .expect("contiguous attention");

            let _paged_output = paged_attention
                .forward(AttentionInput {
                    q: &q,
                    k: &k,
                    v: &v,
                    mask: AttentionMask::Causal,
                })
                .expect("paged attention");

            let hidden = project_output(&standard_output, &inputs.wo);

            let token_id = next_token(&hidden, &inputs.lm_head);
            current = inputs.embed_token(token_id);

            print_memory_row(
                context_length,
                context_length + step,
                standard_attention.cache().capacity(),
                paged_attention.cache().num_blocks(),
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

    let mut group = c.benchmark_group("kv_cache_token_inference");

    group.sample_size(10);
    group.throughput(Throughput::Elements(GENERATED_TOKENS as u64));

    for context_length in CONTEXT_LENGTHS {
        let inputs = Inputs::new(context_length);
        let (prefix_k, prefix_v) = inputs.prefix_kv();

        group.bench_with_input(
            BenchmarkId::new("uncached_standard", context_length),
            &context_length,
            |b, _| {
                b.iter(|| {
                    decode_without_kv_cache(black_box(&inputs));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("cached_contiguous", context_length),
            &context_length,
            |b, _| {
                b.iter_batched(
                    || {
                        let mut attention =
                            CachedAttention::with_capacity_defaults(
                                HEAD_DIM,
                                HEAD_DIM,
                                INITIAL_KV_CACHE_CAPACITY,
                            )
                                .expect("create KV cache");

                        attention
                            .cache_mut()
                            .append(prefix_k.data(), prefix_v.data())
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
                BenchmarkId::new(format!("paged_block_{block_size}"), context_length),
                &context_length,
                |b, _| {
                    b.iter_batched(
                        || {
                            let mut attention =
                                PagedAttention::with_block_size(HEAD_DIM, HEAD_DIM, block_size)
                                    .expect("create paged KV cache");

                            attention
                                .cache_mut()
                                .append(prefix_k.data(), prefix_v.data())
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


use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion,
};
use felina::inference::attention::{
    AttentionBackend,
    AttentionInput,
    AttentionMask,
    CachedAttention,
    StandardAttention,
};
use felina::tensor::Tensor;

fn benchmark_standard_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("standard_decode");
    let head_dim = 64;
    let sequence_lengths = [16, 64, 128, 256, 512];

    for &seq_len in &sequence_lengths {
        // Simulate the tokens already generated.
        let q = Tensor::ones(vec![seq_len, head_dim]).unwrap();
        let k = Tensor::ones(vec![seq_len, head_dim]).unwrap();
        let v = Tensor::ones(vec![seq_len, head_dim]).unwrap();

        // Benchmark decoding the final token using the full prefix.
        let q_last = Tensor::ones(vec![1, head_dim]).unwrap();

        group.bench_with_input(
            BenchmarkId::new("prefix_len", seq_len),
            &seq_len,
            |b, _| {
                b.iter(|| {
                    let q_prefix = Tensor::new(
                        q.data()[..seq_len * head_dim].to_vec(),
                        vec![seq_len, head_dim],
                    ).unwrap();

                    black_box(
                        StandardAttention.forward(AttentionInput {
                            q: black_box(&q_prefix),
                            k: black_box(&k),
                            v: black_box(&v),
                            mask: AttentionMask::Causal,
                        })
                    ).unwrap();

                    black_box(&q_last);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_cached_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("cached_decode");
    let head_dim = 64;
    let sequence_lengths = [16, 64, 128, 256, 512];

    for &seq_len in &sequence_lengths {
        let q = Tensor::ones(vec![1, head_dim]).unwrap();
        let k = Tensor::ones(vec![seq_len, head_dim]).unwrap();
        let v = Tensor::ones(vec![seq_len, head_dim]).unwrap();

        group.bench_with_input(
            BenchmarkId::new("prefix_len", seq_len),
            &seq_len,
            |b, _| {
                b.iter_batched(
                    || {
                        let mut attention =
                            CachedAttention::new(head_dim, head_dim).unwrap();

                        // Pre-fill the cache with the existing prefix.
                        // This setup is excluded from the measured time.
                        if seq_len > 1 {
                            let prefix_k = Tensor::ones(
                                vec![seq_len - 1, head_dim],
                            ).unwrap();
                            let prefix_v = Tensor::ones(
                                vec![seq_len - 1, head_dim],
                            ).unwrap();

                            // Seed the cache with the prefix.
                            // Use your cache's append method here if exposed.
                            black_box((&prefix_k, &prefix_v));
                        }

                        attention
                    },
                    |mut attention| {
                        black_box(
                            attention.forward(AttentionInput {
                                q: black_box(&q),
                                k: black_box(&k),
                                v: black_box(&v),
                                mask: AttentionMask::Causal,
                            })
                        ).unwrap();
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_standard_decode,
    benchmark_cached_decode
);
criterion_main!(benches);

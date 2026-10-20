# Felina

Felina is an experimental Rust project exploring the building blocks of an ML library from scratch. It currently provides an `f32` tensor type and a small inference module focused on single-head attention, KV caching and PagedAttention. I named it after the Breaking Bad finale.

## What’s implemented

- Contiguous, row-major tensors with indexing, reshape, transpose, and borrowed views.
- Elementwise addition and multiplication, scalar scaling, reductions, and stable softmax along the last dimension.
- Tiled matrix multiplication, including vector operands, batched multiplication, and batch broadcasting.
- Standard scaled dot-product attention with optional causal masking.
- Attention with contiguous and paged key/value caches.

The current attention implementations operate on 2D, single-head tensors. Felina does not yet provide a complete model or training framework.

## Quick start

Add Felina as a dependency:

```toml
[dependencies]
felina = { path = "path/to/felina" }
```

Create and multiply tensors:

```rust
use felina::tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2])?;
let product = a.matmul(&b)?;

assert_eq!(product.shape(), &[2, 2]);
assert_eq!(product.data(), &[19.0, 22.0, 43.0, 50.0]);
Ok(())
}
```

Run standard causal attention:

```rust
use felina::inference::attention::{
    AttentionBackend, AttentionInput, AttentionMask, StandardAttention,
};
use felina::tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
let q = Tensor::new(vec![1.0, 0.0], vec![1, 2])?;
let k = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2])?;
let v = Tensor::new(vec![2.0, 4.0, 6.0, 8.0], vec![2, 2])?;

let mut attention = StandardAttention;
let output = attention.forward(AttentionInput {
    q: &q,
    k: &k,
    v: &v,
    mask: AttentionMask::Causal,
})?;
Ok(())
}
```

## Paged attention and KV caches

During autoregressive decoding, each new token adds a key and value vector to the history. Recomputing projections for the whole prefix at every step repeats work; a KV cache stores earlier keys and values so each step can add only the new rows.

`CachedAttention` keeps K and V in contiguous buffers. `PagedAttention` stores them in fixed-size blocks allocated as needed and attends to each block directly, without flattening the cache into a new contiguous tensor.

Paged attention uses online softmax to combine attention across blocks in one pass. For each query, it tracks the largest score seen so far, a running softmax denominator, and a weighted sum of value vectors. When a block introduces a larger score, it rescales the accumulated denominator and value sum before adding that block’s contribution. This keeps the result numerically stable while avoiding a full-sequence score or probability tensor; only the current block’s scores and the running accumulators are needed. Choose a block size when creating the backend:

```rust
use felina::inference::attention::{
    AttentionBackend, AttentionInput, AttentionMask, PagedAttention,
};
use felina::tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut attention = PagedAttention::with_block_size(64, 64, 16)?;
let query = Tensor::new(vec![1.0; 64], vec![1, 64])?;
let new_key = Tensor::new(vec![0.5; 64], vec![1, 64])?;
let new_value = Tensor::new(vec![0.25; 64], vec![1, 64])?;

let output = attention.forward(AttentionInput {
    q: &query,
    k: &new_key,
    v: &new_value,
    mask: AttentionMask::Causal,
})?;
assert_eq!(output.shape(), &[1, 64]);
Ok(())
}
```

The paged cache grows in whole blocks. This can reduce unused reserved capacity compared with the contiguous cache’s initial 512-token capacity, particularly for shorter contexts. The table shows estimated K+V payload capacity for block size 16 at prefill and after appending one generated token. It excludes allocator overhead and process memory outside the cache.

| Context tokens | Contiguous cache | Paged cache after prefill | Paged cache after next token |
| ---: | ---: | ---: | ---: |
| 16 | 256 KiB | 8 KiB | 16 KiB |
| 64 | 256 KiB | 32 KiB | 40 KiB |
| 128 | 256 KiB | 64 KiB | 72 KiB |
| 256 | 256 KiB | 128 KiB | 136 KiB |
| 512 | 256 KiB | 256 KiB | 264 KiB |
| 1,024 | 512 KiB | 512 KiB | 520 KiB |

These figures use the benchmark’s 64-dimensional keys and values stored as `f32` (512 payload bytes per token). Paged allocation rounds up to a whole block; for example, growing from 16 to 17 tokens allocates a second block.

## Benchmarks

The Criterion benchmark compares 16 decode steps across context lengths of 16 to 1,024 tokens. Inputs use a hidden size of 128 and attention head dimension of 64. The uncached path recomputes K/V projections for the growing prefix; cached paths project only each new token. Paged attention is measured with block sizes 16, 32, and 64. The table reports Criterion’s middle time estimate per 16-token decode run from one supplied run:

| Context | Uncached | Contiguous cache | Paged (16) | Paged (32) | Paged (64) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 3.434 ms | 285.05 µs | 281.42 µs | 239.04 µs | 239.51 µs |
| 64 | 10.147 ms | 377.71 µs | 287.81 µs | 282.80 µs | 281.39 µs |
| 128 | 18.694 ms | 523.23 µs | 348.93 µs | 473.10 µs | 382.50 µs |
| 256 | 45.319 ms | 928.75 µs | 470.00 µs | 454.92 µs | 680.11 µs |
| 512 | 162.78 ms | 1.840 ms | 743.71 µs | 1.838 ms | 1.600 ms |
| 1,024 | 163.84 ms | 3.447 ms | 1.229 ms | 1.117 ms | 1.105 ms |

Results vary with hardware and run conditions; the 128–512 token measurements in this run also show substantial variation for some block sizes. Treat them as an initial comparison, not a general performance claim.

Run the tests and benchmark with:

```sh
cargo test
cargo bench --bench attention
```

## Current scope and future goals

Felina is a learning and development project, and its APIs and performance are still evolving. Longer-term goals include multi-head attention, a fuller inference engine, neural network backpropagation, and GPU execution and optimization.

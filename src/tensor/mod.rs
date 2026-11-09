// src/tensor/mod.rs

mod basic_ops;
mod elementwise;
mod matmul;
mod types;

pub use types::{Tensor, TensorError, TensorView};

pub(crate) const TILE_SIZE: usize = 32;

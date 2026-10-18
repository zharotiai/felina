mod tensor;
mod matmul;
mod elementwise;

pub use tensor::{Tensor, TensorError};

pub(crate) const TILE_SIZE: usize = 32;
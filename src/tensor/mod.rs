mod elementwise;
mod matmul;
mod tensor;

pub use tensor::{Tensor, TensorError, TensorView};

pub(crate) const TILE_SIZE: usize = 32;

// src/ndarray_impl.rs
use crate::TensorBenchmark;
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

pub struct NdArrayBenchmark;

impl TensorBenchmark for NdArrayBenchmark {
    type Tensor = Array2<f32>;

    fn name(&self) -> &'static str {
        "ndarray"
    }

    fn create_random_tensor(&self, shape: &[usize]) -> Self::Tensor {
        match shape.len() {
            1 => {
                let arr1 = Array1::random(shape[0], Uniform::new(-1.0, 1.0));
                arr1.insert_axis(Axis(1))
            }
            2 => Array2::random((shape[0], shape[1]), Uniform::new(-1.0, 1.0)),
            _ => panic!("Unsupported shape length: {}", shape.len()),
        }
    }

    fn create_zeros(&self, shape: &[usize]) -> Self::Tensor {
        match shape.len() {
            1 => Array2::zeros((shape[0], 1)),
            2 => Array2::zeros((shape[0], shape[1])),
            _ => panic!("Unsupported shape length: {}", shape.len()),
        }
    }

    fn create_ones(&self, shape: &[usize]) -> Self::Tensor {
        match shape.len() {
            1 => Array2::ones((shape[0], 1)),
            2 => Array2::ones((shape[0], shape[1])),
            _ => panic!("Unsupported shape length: {}", shape.len()),
        }
    }

    fn add(&self, a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor {
        a + b
    }

    fn multiply(&self, a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor {
        a * b
    }

    fn matmul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor {
        a.dot(b)
    }

    fn transpose(&self, tensor: &Self::Tensor) -> Self::Tensor {
        tensor.t().to_owned()
    }

    fn sum(&self, tensor: &Self::Tensor) -> f32 {
        tensor.sum()
    }

    fn mean(&self, tensor: &Self::Tensor) -> f32 {
        tensor.mean().unwrap()
    }
}

// Helper struct for 1D operations
pub struct NdArrayBenchmark1D;

impl NdArrayBenchmark1D {
    pub fn create_random_vector(&self, size: usize) -> Array1<f32> {
        Array1::random(size, Uniform::new(-1.0, 1.0))
    }

    pub fn vector_add(&self, a: &Array1<f32>, b: &Array1<f32>) -> Array1<f32> {
        a + b
    }

    pub fn vector_multiply(&self, a: &Array1<f32>, b: &Array1<f32>) -> Array1<f32> {
        a * b
    }

    pub fn vector_dot(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        a.dot(b)
    }
}

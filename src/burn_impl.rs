use crate::TensorBenchmark;
use burn::{
    backend::{Autodiff, NdArray},
    tensor::{Distribution, Tensor},
};

pub type Backend = NdArray<f32>;
pub type AutodiffBackend = Autodiff<Backend>;

pub struct BurnBenchmark;

impl TensorBenchmark for BurnBenchmark {
    type Tensor = Tensor<Backend, 2>;

    fn name(&self) -> &'static str {
        "burn"
    }

    fn create_random_tensor(&self, shape: &[usize]) -> Self::Tensor {
        match shape.len() {
            1 => {
                let tensor: Tensor<Backend, 1> = Tensor::random(
                    [shape[0]],
                    Distribution::Uniform(-1.0, 1.0),
                    &Default::default(),
                );
                tensor.unsqueeze_dim(1)
            }
            2 => Tensor::random(
                [shape[0], shape[1]],
                Distribution::Uniform(-1.0, 1.0),
                &Default::default(),
            ),
            _ => panic!("Unsupported shape length: {}", shape.len()),
        }
    }

    fn create_zeros(&self, shape: &[usize]) -> Self::Tensor {
        match shape.len() {
            1 => {
                let tensor: Tensor<Backend, 1> = Tensor::zeros([shape[0]], &Default::default());
                tensor.unsqueeze_dim(1)
            }
            2 => Tensor::zeros([shape[0], shape[1]], &Default::default()),
            _ => panic!("Unsupported shape length: {}", shape.len()),
        }
    }

    fn create_ones(&self, shape: &[usize]) -> Self::Tensor {
        match shape.len() {
            1 => {
                let tensor: Tensor<Backend, 1> = Tensor::ones([shape[0]], &Default::default());
                tensor.unsqueeze_dim(1)
            }
            2 => Tensor::ones([shape[0], shape[1]], &Default::default()),
            _ => panic!("Unsupported shape length: {}", shape.len()),
        }
    }

    fn add(&self, a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor {
        a.clone() + b.clone()
    }

    fn multiply(&self, a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor {
        a.clone() * b.clone()
    }

    fn matmul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor {
        a.matmul(b.clone())
    }

    fn transpose(&self, tensor: &Self::Tensor) -> Self::Tensor {
        tensor.transpose()
    }

    fn sum(&self, tensor: &Self::Tensor) -> f32 {
        tensor.sum().into_scalar()
    }

    fn mean(&self, tensor: &Self::Tensor) -> f32 {
        tensor.mean().into_scalar()
    }
}

// Helper struct for 1D tensors
pub struct BurnBenchmark1D;

impl BurnBenchmark1D {
    pub fn create_random_vector(&self, size: usize) -> Tensor<Backend, 1> {
        Tensor::random(
            [size],
            Distribution::Uniform(-1.0, 1.0),
            &Default::default(),
        )
    }

    pub fn vector_add(&self, a: &Tensor<Backend, 1>, b: &Tensor<Backend, 1>) -> Tensor<Backend, 1> {
        a.clone() + b.clone()
    }

    pub fn vector_multiply(
        &self,
        a: &Tensor<Backend, 1>,
        b: &Tensor<Backend, 1>,
    ) -> Tensor<Backend, 1> {
        a.clone() * b.clone()
    }

    pub fn vector_dot(&self, a: &Tensor<Backend, 1>, b: &Tensor<Backend, 1>) -> f32 {
        (a.clone() * b.clone()).sum().into_scalar()
    }
}

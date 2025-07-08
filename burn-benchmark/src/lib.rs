use serde::{Deserialize, Serialize};
use burn::{
    backend::NdArray,
    tensor::{Distribution, Tensor},
};

pub type Backend = NdArray<f32>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    pub name: String,
    pub shape: Vec<usize>,
    pub operation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub library: String,
    pub test_case: String,
    pub mean_time_ns: f64,
    pub std_dev_ns: f64,
    pub throughput_ops_per_sec: f64,
    pub memory_usage_bytes: Option<usize>,
}

pub trait TensorBenchmark {
    type Tensor;

    fn name(&self) -> &'static str;
    fn create_random_tensor(&self, shape: &[usize]) -> Self::Tensor;
    fn create_zeros(&self, shape: &[usize]) -> Self::Tensor;
    fn create_ones(&self, shape: &[usize]) -> Self::Tensor;

    // Basic operations
    fn add(&self, a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;
    fn multiply(&self, a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;
    fn matmul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;
    fn transpose(&self, tensor: &Self::Tensor) -> Self::Tensor;
    fn sum(&self, tensor: &Self::Tensor) -> f32;
    fn mean(&self, tensor: &Self::Tensor) -> f32;
}

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
        a.clone().matmul(b.clone())
    }

    fn transpose(&self, tensor: &Self::Tensor) -> Self::Tensor {
        tensor.clone().transpose()
    }

    fn sum(&self, tensor: &Self::Tensor) -> f32 {
        tensor.clone().sum().into_scalar()
    }

    fn mean(&self, tensor: &Self::Tensor) -> f32 {
        tensor.clone().mean().into_scalar()
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

// Test case generators
pub fn generate_test_cases() -> Vec<TestCase> {
    let mut cases = Vec::new();

    // Small tensors
    for size in [10, 100, 500] {
        cases.push(TestCase {
            name: format!("vector_ops_{}", size),
            shape: vec![size],
            operation: "vector_ops".to_string(),
        });
    }

    // Medium 2D tensors
    for size in [32, 64, 128, 256, 512] {
        cases.push(TestCase {
            name: format!("matrix_ops_{}x{}", size, size),
            shape: vec![size, size],
            operation: "matrix_ops".to_string(),
        });
    }

    // Large tensors
    for size in [1024, 2048] {
        cases.push(TestCase {
            name: format!("large_matrix_{}x{}", size, size),
            shape: vec![size, size],
            operation: "matrix_ops".to_string(),
        });
    }

    // Batch operations
    for (batch, dim) in [(8, 256), (32, 128), (64, 64)] {
        cases.push(TestCase {
            name: format!("batch_ops_{}x{}", batch, dim),
            shape: vec![batch, dim],
            operation: "batch_ops".to_string(),
        });
    }

    cases
}
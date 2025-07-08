use serde::{Deserialize, Serialize};
use candle_core::{Device, Result, Tensor};

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

pub struct CandleBenchmark {
    device: Device,
}

impl CandleBenchmark {
    pub fn new() -> Result<Self> {
        Ok(Self {
            device: Device::Cpu,
        })
    }
}

impl TensorBenchmark for CandleBenchmark {
    type Tensor = Tensor;

    fn name(&self) -> &'static str {
        "candle"
    }

    fn create_random_tensor(&self, shape: &[usize]) -> Self::Tensor {
        match shape.len() {
            1 => Tensor::randn(0f32, 1f32, (shape[0],), &self.device).unwrap(),
            2 => Tensor::randn(0f32, 1f32, (shape[0], shape[1]), &self.device).unwrap(),
            _ => panic!("Unsupported shape length: {}", shape.len()),
        }
    }

    fn create_zeros(&self, shape: &[usize]) -> Self::Tensor {
        match shape.len() {
            1 => Tensor::zeros((shape[0],), candle_core::DType::F32, &self.device).unwrap(),
            2 => {
                Tensor::zeros((shape[0], shape[1]), candle_core::DType::F32, &self.device).unwrap()
            }
            _ => panic!("Unsupported shape length: {}", shape.len()),
        }
    }

    fn create_ones(&self, shape: &[usize]) -> Self::Tensor {
        match shape.len() {
            1 => Tensor::ones((shape[0],), candle_core::DType::F32, &self.device).unwrap(),
            2 => Tensor::ones((shape[0], shape[1]), candle_core::DType::F32, &self.device).unwrap(),
            _ => panic!("Unsupported shape length: {}", shape.len()),
        }
    }

    fn add(&self, a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor {
        a.add(b).unwrap()
    }

    fn multiply(&self, a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor {
        a.mul(b).unwrap()
    }

    fn matmul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor {
        a.matmul(b).unwrap()
    }

    fn transpose(&self, tensor: &Self::Tensor) -> Self::Tensor {
        // For 2D tensors, transpose the last two dimensions
        if tensor.dims().len() == 2 {
            tensor.t().unwrap()
        } else {
            tensor.transpose(0, 1).unwrap()
        }
    }

    fn sum(&self, tensor: &Self::Tensor) -> f32 {
        let sum_tensor = tensor.sum_all().unwrap();
        sum_tensor.to_scalar::<f32>().unwrap()
    }

    fn mean(&self, tensor: &Self::Tensor) -> f32 {
        let sum = self.sum(tensor);
        let numel = tensor.elem_count() as f32;
        sum / numel
    }
}

impl Default for CandleBenchmark {
    fn default() -> Self {
        Self::new().unwrap()
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
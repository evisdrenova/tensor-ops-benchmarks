# Rust Tensor Libraries Benchmark

## Overview

This project benchmarks three Rust tensor libraries - Burn, Candle, and NDArray - across various tensor operations and sizes. Each library is set up as an independent project to avoid dependency conflicts.

## Project Structure

```
tensor-benchmark/
├── candle-benchmark/     # Candle library benchmarks
├── burn-benchmark/       # Burn library benchmarks
├── ndarray-benchmark/    # NDArray library benchmarks
├── readme.md
└── .gitignore
```

## Libraries Being Tested

1. **Candle** (v0.8.4) - Minimalist ML framework for Rust
2. **Burn** (v0.17.1) - Pure Rust deep learning framework with NdArray backend
3. **NDArray** (v0.16) - Pure Rust n-dimensional arrays (baseline)

## How to Run Benchmarks

### Prerequisites
- Rust 1.70+ installed
- Cargo package manager

### Running Individual Benchmarks

Each library can be benchmarked independently:

```bash
# Candle benchmarks
cd candle-benchmark
cargo bench

# Burn benchmarks  
cd burn-benchmark
cargo bench

# NDArray benchmarks
cd ndarray-benchmark
cargo bench
```

### Running All Benchmarks

```bash
# From the root directory
./scripts/run_all_benchmarks.sh  # If script exists
# Or manually:
cd candle-benchmark && cargo bench && cd ../burn-benchmark && cargo bench && cd ../ndarray-benchmark && cargo bench
```

## What's Being Tested

### Current Benchmark Operations

1. **Tensor Creation**
   - Random tensor generation (various sizes)
   - Benchmarks throughput in elements/second

2. **Matrix Multiplication**
   - Square matrix multiplication (64x64 to 512x512)
   - Measures FLOPS (floating point operations per second)

3. **Element-wise Operations**
   - Addition and multiplication of tensors
   - Tests vectorization efficiency

4. **Reduction Operations**
   - Sum and mean calculations
   - Benchmarks aggregation performance

5. **Vector Operations** (Burn & NDArray only)
   - Dot product on 1D vectors
   - Various vector sizes (1K to 1M elements)

### Tensor Sizes Tested

- **Small**: 64x64, 128x128, 256x256
- **Medium**: 512x512, 1024x1024  
- **Large**: 2048x2048 (matrix multiplication only)
- **Vectors**: 1K, 10K, 100K, 1M elements

### Performance Metrics

- **Throughput**: Elements processed per second
- **FLOPS**: Floating point operations per second (for matrix multiplication)
- **Latency**: Mean execution time with standard deviation
- **Memory efficiency**: Implicit through throughput measurements

## Benchmark Results

Results are generated using the Criterion benchmarking framework and include:

- **HTML Reports**: Detailed performance graphs and statistics
- **Statistical Analysis**: Mean, standard deviation, and confidence intervals
- **Regression Detection**: Identifies performance regressions between runs
- **Comparison Plots**: Visual comparison between libraries

### Viewing Results

After running benchmarks, HTML reports are generated in each project's `target/criterion/` directory:

```bash
# Example: View Candle benchmark results
open candle-benchmark/target/criterion/index.html

# Or for all projects
open */target/criterion/index.html
```

## Implementation Details

### Common Tensor Operations Interface

All three libraries implement the same `TensorBenchmark` trait:

```rust
pub trait TensorBenchmark {
    type Tensor;
    
    fn create_random_tensor(&self, shape: &[usize]) -> Self::Tensor;
    fn add(&self, a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;
    fn multiply(&self, a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;
    fn matmul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;
    fn sum(&self, tensor: &Self::Tensor) -> f32;
    fn mean(&self, tensor: &Self::Tensor) -> f32;
    // ... other operations
}
```

### Hardware Configuration

- **CPU**: Multi-threaded execution (uses all available cores)
- **Data Type**: f32 (32-bit floating point)
- **Memory**: System RAM (no GPU acceleration currently)
- **Optimization**: Release mode with LTO enabled

## Future Enhancements

Planned additions for more comprehensive benchmarking:

- GPU acceleration support (CUDA, Metal, ROCm)
- Additional tensor operations (convolution, pooling, etc.)
- Mixed precision benchmarking (f16, bf16)
- Memory usage profiling
- Batch processing optimizations
- Cross-platform performance comparison

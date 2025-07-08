// benches/tensor_ops.rs
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use tensor_benchmark::*;
use tensor_benchmark::{burn_impl::*, candle_impl::*, ndarray_impl::*};

fn benchmark_tensor_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_creation");

    let sizes = vec![64, 128, 256, 512, 1024];

    for size in sizes {
        let shape = vec![size, size];
        let elements = (size * size) as u64;
        group.throughput(Throughput::Elements(elements));

        // Burn
        group.bench_with_input(BenchmarkId::new("burn_random", size), &shape, |b, shape| {
            let benchmark = BurnBenchmark;
            b.iter(|| black_box(benchmark.create_random_tensor(shape)));
        });

        // Candle
        group.bench_with_input(
            BenchmarkId::new("candle_random", size),
            &shape,
            |b, shape| {
                let benchmark = CandleBenchmark::new().unwrap();
                b.iter(|| black_box(benchmark.create_random_tensor(shape)));
            },
        );

        // NDArray
        group.bench_with_input(
            BenchmarkId::new("ndarray_random", size),
            &shape,
            |b, shape| {
                let benchmark = NdArrayBenchmark;
                b.iter(|| black_box(benchmark.create_random_tensor(shape)));
            },
        );
    }
    group.finish();
}

fn benchmark_matrix_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_multiplication");

    let sizes = vec![64, 128, 256, 512];

    for size in sizes {
        let shape = vec![size, size];
        let flops = (2 * size * size * size) as u64; // 2*n^3 for matrix multiplication
        group.throughput(Throughput::Elements(flops));

        // Burn
        group.bench_with_input(BenchmarkId::new("burn_matmul", size), &shape, |b, shape| {
            let benchmark = BurnBenchmark;
            let a = benchmark.create_random_tensor(shape);
            let tensor_b = benchmark.create_random_tensor(shape);
            b.iter(|| {
                let result = benchmark.matmul(&a, &tensor_b);
                black_box(result)
            });
        });

        // Candle
        group.bench_with_input(
            BenchmarkId::new("candle_matmul", size),
            &shape,
            |b, shape| {
                let benchmark = CandleBenchmark::new().unwrap();
                let a = benchmark.create_random_tensor(shape);
                let tensor_b = benchmark.create_random_tensor(shape);
                b.iter(|| {
                    let result = benchmark.matmul(&a, &tensor_b);
                    black_box(result)
                });
            },
        );

        // NDArray
        group.bench_with_input(
            BenchmarkId::new("ndarray_matmul", size),
            &shape,
            |b, shape| {
                let benchmark = NdArrayBenchmark;
                let a = benchmark.create_random_tensor(shape);
                let tensor_b = benchmark.create_random_tensor(shape);
                b.iter(|| {
                    let result = benchmark.matmul(&a, &tensor_b);
                    black_box(result)
                });
            },
        );
    }
    group.finish();
}

fn benchmark_element_wise_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("element_wise_ops");

    let sizes = vec![64, 128, 256, 512, 1024];

    for size in sizes {
        let shape = vec![size, size];
        let elements = (size * size) as u64;
        group.throughput(Throughput::Elements(elements));

        // Test addition
        group.bench_with_input(BenchmarkId::new("burn_add", size), &shape, |b, shape| {
            let benchmark = BurnBenchmark;
            let a = benchmark.create_random_tensor(shape);
            let tensor_b = benchmark.create_random_tensor(shape);
            b.iter(|| {
                let result = benchmark.add(&a, &tensor_b);
                black_box(result)
            });
        });

        group.bench_with_input(BenchmarkId::new("candle_add", size), &shape, |b, shape| {
            let benchmark = CandleBenchmark::new().unwrap();
            let a = benchmark.create_random_tensor(shape);
            let tensor_b = benchmark.create_random_tensor(shape);
            b.iter(|| {
                let result = benchmark.add(&a, &tensor_b);
                black_box(result)
            });
        });

        group.bench_with_input(BenchmarkId::new("ndarray_add", size), &shape, |b, shape| {
            let benchmark = NdArrayBenchmark;
            let a = benchmark.create_random_tensor(shape);
            let tensor_b = benchmark.create_random_tensor(shape);
            b.iter(|| {
                let result = benchmark.add(&a, &tensor_b);
                black_box(result)
            });
        });

        // Test multiplication
        group.bench_with_input(
            BenchmarkId::new("burn_multiply", size),
            &shape,
            |b, shape| {
                let benchmark = BurnBenchmark;
                let a = benchmark.create_random_tensor(shape);
                let tensor_b = benchmark.create_random_tensor(shape);
                b.iter(|| {
                    let result = benchmark.multiply(&a, &tensor_b);
                    black_box(result)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("candle_multiply", size),
            &shape,
            |b, shape| {
                let benchmark = CandleBenchmark::new().unwrap();
                let a = benchmark.create_random_tensor(shape);
                let tensor_b = benchmark.create_random_tensor(shape);
                b.iter(|| {
                    let result = benchmark.multiply(&a, &tensor_b);
                    black_box(result)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ndarray_multiply", size),
            &shape,
            |b, shape| {
                let benchmark = NdArrayBenchmark;
                let a = benchmark.create_random_tensor(shape);
                let tensor_b = benchmark.create_random_tensor(shape);
                b.iter(|| {
                    let result = benchmark.multiply(&a, &tensor_b);
                    black_box(result)
                });
            },
        );
    }
    group.finish();
}

fn benchmark_reduction_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduction_ops");

    let sizes = vec![64, 128, 256, 512, 1024];

    for size in sizes {
        let shape = vec![size, size];
        let elements = (size * size) as u64;
        group.throughput(Throughput::Elements(elements));

        // Test sum
        group.bench_with_input(BenchmarkId::new("burn_sum", size), &shape, |b, shape| {
            let benchmark = BurnBenchmark;
            let tensor = benchmark.create_random_tensor(shape);
            b.iter(|| black_box(benchmark.sum(&tensor)));
        });

        group.bench_with_input(BenchmarkId::new("candle_sum", size), &shape, |b, shape| {
            let benchmark = CandleBenchmark::new().unwrap();
            let tensor = benchmark.create_random_tensor(shape);
            b.iter(|| black_box(benchmark.sum(&tensor)));
        });

        group.bench_with_input(BenchmarkId::new("ndarray_sum", size), &shape, |b, shape| {
            let benchmark = NdArrayBenchmark;
            let tensor = benchmark.create_random_tensor(shape);
            b.iter(|| black_box(benchmark.sum(&tensor)));
        });

        // Test mean
        group.bench_with_input(BenchmarkId::new("burn_mean", size), &shape, |b, shape| {
            let benchmark = BurnBenchmark;
            let tensor = benchmark.creaPe_random_tensor(shape);
            b.iter(|| black_box(benchmark.mean(&tensor)));
        });

        group.bench_with_input(BenchmarkId::new("candle_mean", size), &shape, |b, shape| {
            let benchmark = CandleBenchmark::new().unwrap();
            let tensor = benchmark.create_random_tensor(shape);
            b.iter(|| black_box(benchmark.mean(&tensor)));
        });

        group.bench_with_input(
            BenchmarkId::new("ndarray_mean", size),
            &shape,
            |b, shape| {
                let benchmark = NdArrayBenchmark;
                let tensor = benchmark.create_random_tensor(shape);
                b.iter(|| black_box(benchmark.mean(&tensor)));
            },
        );
    }
    group.finish();
}

fn benchmark_vector_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_ops");

    let sizes = vec![1000, 10000, 100000, 1000000];

    for size in sizes {
        group.throughput(Throughput::Elements(size as u64));

        // Vector dot product
        group.bench_with_input(
            BenchmarkId::new("burn_vector_dot", size),
            &size,
            |b, &size| {
                let benchmark = BurnBenchmark1D;
                let a = benchmark.create_random_vector(size);
                let vec_b = benchmark.create_random_vector(size);
                b.iter(|| {
                    let result = benchmark.vector_dot(&a, &vec_b);
                    black_box(result)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ndarray_vector_dot", size),
            &size,
            |b, &size| {
                let benchmark = NdArrayBenchmark1D;
                let a = benchmark.create_random_vector(size);
                let vec_b = benchmark.create_random_vector(size);
                b.iter(|| {
                    let result = benchmark.vector_dot(&a, &vec_b);
                    black_box(result)
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_tensor_creation,
    benchmark_matrix_multiplication,
    benchmark_element_wise_operations,
    benchmark_reduction_operations,
    benchmark_vector_operations
);
criterion_main!(benches);

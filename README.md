# CUDA Programming: 20 Exercises from Easy to Very Hard

A hands-on CUDA C++ practice repository with **20 progressively harder exercises**, a starter file for every exercise, and a complete tested-by-design reference solution. The final exercises move into GPU patterns used in modern ML systems, including row-wise softmax, layer normalization, stream pipelines, concurrent GEMMs, and fused scaled dot-product attention.

> Recommended workflow: attempt `exercises/<name>/starter.cu` first, then compare with `solutions/<name>.cu`.

## What you will learn

- CUDA execution hierarchy: threads, blocks, grids, warps
- Host/device memory allocation and transfers
- Coalesced global memory access
- Shared memory, synchronization, and bank conflicts
- Reductions, atomics, scans, histograms, and stencils
- Matrix tiling and GEMM
- Warp shuffle intrinsics
- Numerically stable softmax and layer normalization
- Streams, async copies, pinned memory, and Unified Memory
- Kernel concurrency and fusion
- A compact fused attention kernel that avoids materializing the N×N score matrix in global memory

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit with `nvcc`
- CMake 3.24+
- C++17-capable host compiler

Check your environment:
```bash
nvcc --version
nvidia-smi
cmake --version
```

## Build all reference solutions

```bash
git clone [<[github-repository](https://github.com/matinfirooz/cuda-programming-exercises/tree/main)>](https://github.com/matinfirooz/cuda-programming-exercises.git)
cd cuda-programming-exercises
cmake -S . -B build
cmake --build build -j
```

Run one exercise:
```bash
./build/02_vector_add
```

Run all built solutions:
```bash
./scripts/run_all.sh build
```

## Exercise roadmap

| # | Exercise | Difficulty | Main ideas |
|---:|---|---|---|
| 1 | [Thread & Grid Indexing](exercises/01_thread_indexing/README.md) | Easy | threadIdx/blockIdx/blockDim, 1D grid-stride thinking, bounds checks |
| 2 | [Vector Addition](exercises/02_vector_add/README.md) | Easy | cudaMalloc/cudaMemcpy, kernel launch configuration, host/device validation |
| 3 | [SAXPY](exercises/03_saxpy/README.md) | Easy | memory-bound kernels, coalesced access, simple arithmetic intensity |
| 4 | [2D Matrix Addition](exercises/04_matrix_add_2d/README.md) | Easy | dim3, row-major indexing, 2D boundary checks |
| 5 | [Dot Product with Atomics](exercises/05_dot_product_atomic/README.md) | Intermediate | shared memory, synchronization, atomics |
| 6 | [Optimized Shared-Memory Reduction](exercises/06_reduction_shared/README.md) | Intermediate | tree reduction, dynamic shared memory, multi-kernel algorithms |
| 7 | [Bank-Conflict-Free Matrix Transpose](exercises/07_tiled_transpose/README.md) | Intermediate | shared-memory tiling, coalescing, bank conflicts |
| 8 | [Tiled Matrix Multiplication](exercises/08_tiled_gemm/README.md) | Intermediate | GEMM, data reuse, synchronization |
| 9 | [1D Convolution with Halo](exercises/09_conv1d_shared/README.md) | Intermediate | stencils, halo regions, constant memory |
| 10 | [2D Convolution](exercises/10_conv2d_tiled/README.md) | Intermediate | 2D stencils, shared-memory halos, image-style kernels |
| 11 | [Histogram with Shared Atomics](exercises/11_histogram_shared/README.md) | Advanced | contention, shared atomics, global merge |
| 12 | [Inclusive Prefix Scan](exercises/12_prefix_scan/README.md) | Advanced | scan, upsweep/downsweep, work-efficient parallel primitives |
| 13 | [Warp Shuffle Reduction](exercises/13_warp_reduce/README.md) | Advanced | warp intrinsics, warp-synchronous programming, reduction hierarchy |
| 14 | [Numerically Stable Row Softmax](exercises/14_row_softmax/README.md) | Advanced | stable softmax, block reduction, AI primitive |
| 15 | [Layer Normalization](exercises/15_layernorm/README.md) | Advanced | mean/variance reduction, numerical stability, AI primitive |
| 16 | [CUDA Streams & Async Copies](exercises/16_streams_overlap/README.md) | Very hard | streams, cudaMemcpyAsync, pipeline overlap |
| 17 | [Pinned-Memory Multi-Stream Pipeline](exercises/17_pinned_pipeline/README.md) | Very hard | cudaMallocHost, transfer throughput, stream pipelines |
| 18 | [Unified Memory & Prefetch](exercises/18_unified_memory/README.md) | Very hard | Unified Memory, prefetch, device placement |
| 19 | [Concurrent Batched GEMM](exercises/19_batched_stream_gemm/README.md) | Very hard | kernel concurrency, stream scheduling, batched workloads |
| 20 | [Fused Scaled Dot-Product Attention](exercises/20_fused_attention/README.md) | Very hard | fusion, shared memory, softmax |

## Repository structure

```text
cuda-programming-exercises/
├── CMakeLists.txt
├── README.md
├── LICENSE
├── include/
│   └── cuda_utils.cuh
├── exercises/
│   ├── 01_thread_indexing/
│   │   ├── README.md
│   │   └── starter.cu
│   └── ...
├── solutions/
│   ├── 01_thread_indexing.cu
│   └── ... 20 solved CUDA programs
└── scripts/
    └── run_all.sh
```

## How to practice effectively

For each problem, first write down the mapping from data elements to threads. Then decide what memory each value should live in: registers, shared memory, global memory, constant memory, pinned host memory, or managed memory. Add a CPU reference or a strong invariant before optimizing. Only after correctness should you profile and tune block size, occupancy, memory traffic, synchronization, and kernel fusion.

### Suggested 4-stage path

**Stage 1 — Fundamentals (1–4):** become comfortable with grid/block/thread indexing and memory copies.

**Stage 2 — Core GPU algorithms (5–10):** learn reductions, atomics, tiling, transposes, GEMM, and stencil-style memory access.

**Stage 3 — Parallel primitives + AI kernels (11–15):** implement histogram, scan, warp reductions, stable softmax, and layer norm.

**Stage 4 — Systems + fusion (16–20):** learn asynchronous execution, pinned memory, managed memory, concurrent workloads, and fused attention.

## Profiling

Use Nsight Systems for CPU/GPU overlap and Nsight Compute for kernel-level analysis. Typical questions to answer:

- Are global memory accesses coalesced?
- Is the kernel limited by memory bandwidth or arithmetic throughput?
- How much shared memory is used per block?
- Are there shared-memory bank conflicts?
- Is occupancy unexpectedly low?
- Are streams really overlapping copies and kernels?
- Does fusion reduce global-memory traffic enough to justify extra shared-memory/register pressure?

Example:
```bash
ncu ./build/08_tiled_gemm
nsys profile ./build/17_pinned_pipeline
```

## Notes on the hardest exercises

The implementations are intentionally educational rather than replacements for production libraries such as cuBLAS or highly tuned attention libraries. Exercise 20 limits the problem size so a single block can keep one attention row's score vector in shared memory. Its purpose is to make the dataflow and fusion strategy visible before moving to tiled/online-softmax algorithms for long sequences.

## Next improvements you can make

After finishing all 20, useful extensions are FP16/BF16 kernels, `half2` vectorization, WMMA/Tensor Cores, CUTLASS, cooperative groups, multi-GPU programming, NCCL, CUDA Graphs, and FlashAttention-style online softmax.

## License

MIT. You can use, modify, and publish the repository.

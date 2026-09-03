# Vector Addition

**Level:** Easy

## Objective
Add two large vectors on the GPU and validate against a CPU reference.

## Your task
1. Open `starter.cu`.
2. Implement the required CUDA kernel(s) and host-side setup.
3. Add correctness checking against a CPU reference or an invariant.
4. Test at least one non-round input size where relevant.
5. Compare your implementation with `../../solutions/02_vector_add.cu` only after attempting it yourself.

## Concepts
- cudaMalloc/cudaMemcpy
- kernel launch configuration
- host/device validation

## Stretch challenge
Benchmark several block sizes: 64, 128, 256, 512.

## Build only the solved reference
From the repository root:
```bash
cmake -S . -B build
cmake --build build -j
./build/02_vector_add
```

# SAXPY

**Level:** Easy

## Objective
Implement y = a*x + y for single-precision vectors.

## Your task
1. Open `starter.cu`.
2. Implement the required CUDA kernel(s) and host-side setup.
3. Add correctness checking against a CPU reference or an invariant.
4. Test at least one non-round input size where relevant.
5. Compare your implementation with `../../solutions/03_saxpy.cu` only after attempting it yourself.

## Concepts
- memory-bound kernels
- coalesced access
- simple arithmetic intensity

## Stretch challenge
Measure effective memory bandwidth.

## Build only the solved reference
From the repository root:
```bash
cmake -S . -B build
cmake --build build -j
./build/03_saxpy
```

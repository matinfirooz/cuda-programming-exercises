# Optimized Shared-Memory Reduction

**Level:** Intermediate

## Objective
Reduce a large array to a scalar using multi-stage block reductions.

## Your task
1. Open `starter.cu`.
2. Implement the required CUDA kernel(s) and host-side setup.
3. Add correctness checking against a CPU reference or an invariant.
4. Test at least one non-round input size where relevant.
5. Compare your implementation with `../../solutions/06_reduction_shared.cu` only after attempting it yourself.

## Concepts
- tree reduction
- dynamic shared memory
- multi-kernel algorithms

## Stretch challenge
Remove avoidable __syncthreads() in the final warp.

## Build only the solved reference
From the repository root:
```bash
cmake -S . -B build
cmake --build build -j
./build/06_reduction_shared
```

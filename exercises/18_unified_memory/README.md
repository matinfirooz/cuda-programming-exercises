# Unified Memory & Prefetch

**Level:** Very hard

## Objective
Use cudaMallocManaged plus cudaMemPrefetchAsync for a vector transform.

## Your task
1. Open `starter.cu`.
2. Implement the required CUDA kernel(s) and host-side setup.
3. Add correctness checking against a CPU reference or an invariant.
4. Test at least one non-round input size where relevant.
5. Compare your implementation with `../../solutions/18_unified_memory.cu` only after attempting it yourself.

## Concepts
- Unified Memory
- prefetch
- device placement

## Stretch challenge
Compare prefetched vs demand-paged execution.

## Build only the solved reference
From the repository root:
```bash
cmake -S . -B build
cmake --build build -j
./build/18_unified_memory
```

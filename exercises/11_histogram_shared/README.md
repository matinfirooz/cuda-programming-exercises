# Histogram with Shared Atomics

**Level:** Advanced

## Objective
Build a 256-bin histogram using per-block shared-memory histograms.

## Your task
1. Open `starter.cu`.
2. Implement the required CUDA kernel(s) and host-side setup.
3. Add correctness checking against a CPU reference or an invariant.
4. Test at least one non-round input size where relevant.
5. Compare your implementation with `../../solutions/11_histogram_shared.cu` only after attempting it yourself.

## Concepts
- contention
- shared atomics
- global merge

## Stretch challenge
Benchmark against direct global atomicAdd.

## Build only the solved reference
From the repository root:
```bash
cmake -S . -B build
cmake --build build -j
./build/11_histogram_shared
```

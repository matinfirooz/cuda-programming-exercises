# CUDA Streams & Async Copies

**Level:** Very hard

## Objective
Process a vector in chunks while overlapping H2D copy, kernel execution, and D2H copy.

## Your task
1. Open `starter.cu`.
2. Implement the required CUDA kernel(s) and host-side setup.
3. Add correctness checking against a CPU reference or an invariant.
4. Test at least one non-round input size where relevant.
5. Compare your implementation with `../../solutions/16_streams_overlap.cu` only after attempting it yourself.

## Concepts
- streams
- cudaMemcpyAsync
- pipeline overlap

## Stretch challenge
Use 3+ streams and experiment with chunk sizes.

## Build only the solved reference
From the repository root:
```bash
cmake -S . -B build
cmake --build build -j
./build/16_streams_overlap
```

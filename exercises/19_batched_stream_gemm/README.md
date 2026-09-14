# Concurrent Batched GEMM

**Level:** Very hard

## Objective
Launch independent tiled matrix multiplies across multiple streams.

## Your task
1. Open `starter.cu`.
2. Implement the required CUDA kernel(s) and host-side setup.
3. Add correctness checking against a CPU reference or an invariant.
4. Test at least one non-round input size where relevant.
5. Compare your implementation with `../../solutions/19_batched_stream_gemm.cu` only after attempting it yourself.

## Concepts
- kernel concurrency
- stream scheduling
- batched workloads

## Stretch challenge
Compare one stream vs many streams and explain when concurrency helps.

## Build only the solved reference
From the repository root:
```bash
cmake -S . -B build
cmake --build build -j
./build/19_batched_stream_gemm
```

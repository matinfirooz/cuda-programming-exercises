# Pinned-Memory Multi-Stream Pipeline

**Level:** Very hard

## Objective
Use page-locked host memory and multiple streams for a chunked transform pipeline.

## Your task
1. Open `starter.cu`.
2. Implement the required CUDA kernel(s) and host-side setup.
3. Add correctness checking against a CPU reference or an invariant.
4. Test at least one non-round input size where relevant.
5. Compare your implementation with `../../solutions/17_pinned_pipeline.cu` only after attempting it yourself.

## Concepts
- cudaMallocHost
- transfer throughput
- stream pipelines

## Stretch challenge
Record events per stage and build a timeline.

## Build only the solved reference
From the repository root:
```bash
cmake -S . -B build
cmake --build build -j
./build/17_pinned_pipeline
```

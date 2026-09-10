# Inclusive Prefix Scan

**Level:** Advanced

## Objective
Implement a Blelloch-style block scan for up to 1024 elements.

## Your task
1. Open `starter.cu`.
2. Implement the required CUDA kernel(s) and host-side setup.
3. Add correctness checking against a CPU reference or an invariant.
4. Test at least one non-round input size where relevant.
5. Compare your implementation with `../../solutions/12_prefix_scan.cu` only after attempting it yourself.

## Concepts
- scan
- upsweep/downsweep
- work-efficient parallel primitives

## Stretch challenge
Extend it to arbitrary-length arrays using block sums.

## Build only the solved reference
From the repository root:
```bash
cmake -S . -B build
cmake --build build -j
./build/12_prefix_scan
```

# Warp Shuffle Reduction

**Level:** Advanced

## Objective
Sum an array using __shfl_down_sync for warp-level reduction.

## Your task
1. Open `starter.cu`.
2. Implement the required CUDA kernel(s) and host-side setup.
3. Add correctness checking against a CPU reference or an invariant.
4. Test at least one non-round input size where relevant.
5. Compare your implementation with `../../solutions/13_warp_reduce.cu` only after attempting it yourself.

## Concepts
- warp intrinsics
- warp-synchronous programming
- reduction hierarchy

## Stretch challenge
Implement max and argmax using shuffle operations.

## Build only the solved reference
From the repository root:
```bash
cmake -S . -B build
cmake --build build -j
./build/13_warp_reduce
```

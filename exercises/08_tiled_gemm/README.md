# Tiled Matrix Multiplication

**Level:** Intermediate

## Objective
Implement C=A*B using shared-memory tiles.

## Your task
1. Open `starter.cu`.
2. Implement the required CUDA kernel(s) and host-side setup.
3. Add correctness checking against a CPU reference or an invariant.
4. Test at least one non-round input size where relevant.
5. Compare your implementation with `../../solutions/08_tiled_gemm.cu` only after attempting it yourself.

## Concepts
- GEMM
- data reuse
- synchronization

## Stretch challenge
Add register tiling so each thread computes multiple outputs.

## Build only the solved reference
From the repository root:
```bash
cmake -S . -B build
cmake --build build -j
./build/08_tiled_gemm
```

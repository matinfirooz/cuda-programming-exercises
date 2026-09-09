# Bank-Conflict-Free Matrix Transpose

**Level:** Intermediate

## Objective
Transpose a matrix using a padded shared-memory tile.

## Your task
1. Open `starter.cu`.
2. Implement the required CUDA kernel(s) and host-side setup.
3. Add correctness checking against a CPU reference or an invariant.
4. Test at least one non-round input size where relevant.
5. Compare your implementation with `../../solutions/07_tiled_transpose.cu` only after attempting it yourself.

## Concepts
- shared-memory tiling
- coalescing
- bank conflicts

## Stretch challenge
Compare TILE x TILE and TILE x BLOCK_ROWS layouts.

## Build only the solved reference
From the repository root:
```bash
cmake -S . -B build
cmake --build build -j
./build/07_tiled_transpose
```

# 2D Matrix Addition

**Level:** Easy

## Objective
Use a 2D grid and 2D thread blocks to add matrices.

## Your task
1. Open `starter.cu`.
2. Implement the required CUDA kernel(s) and host-side setup.
3. Add correctness checking against a CPU reference or an invariant.
4. Test at least one non-round input size where relevant.
5. Compare your implementation with `../../solutions/04_matrix_add_2d.cu` only after attempting it yourself.

## Concepts
- dim3
- row-major indexing
- 2D boundary checks

## Stretch challenge
Compare 16x16 vs 32x8 thread blocks.

## Build only the solved reference
From the repository root:
```bash
cmake -S . -B build
cmake --build build -j
./build/04_matrix_add_2d
```

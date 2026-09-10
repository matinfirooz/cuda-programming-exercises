# 2D Convolution

**Level:** Intermediate

## Objective
Apply a 3x3 image convolution using shared-memory tiles with halos.

## Your task
1. Open `starter.cu`.
2. Implement the required CUDA kernel(s) and host-side setup.
3. Add correctness checking against a CPU reference or an invariant.
4. Test at least one non-round input size where relevant.
5. Compare your implementation with `../../solutions/10_conv2d_tiled.cu` only after attempting it yourself.

## Concepts
- 2D stencils
- shared-memory halos
- image-style kernels

## Stretch challenge
Generalize from a 3x3 filter to arbitrary odd K.

## Build only the solved reference
From the repository root:
```bash
cmake -S . -B build
cmake --build build -j
./build/10_conv2d_tiled
```

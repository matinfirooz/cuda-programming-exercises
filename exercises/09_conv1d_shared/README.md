# 1D Convolution with Halo

**Level:** Intermediate

## Objective
Implement same-size 1D convolution using shared memory and halo loading.

## Your task
1. Open `starter.cu`.
2. Implement the required CUDA kernel(s) and host-side setup.
3. Add correctness checking against a CPU reference or an invariant.
4. Test at least one non-round input size where relevant.
5. Compare your implementation with `../../solutions/09_conv1d_shared.cu` only after attempting it yourself.

## Concepts
- stencils
- halo regions
- constant memory

## Stretch challenge
Move the filter coefficients to __constant__ memory.

## Build only the solved reference
From the repository root:
```bash
cmake -S . -B build
cmake --build build -j
./build/09_conv1d_shared
```

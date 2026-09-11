# Layer Normalization

**Level:** Advanced

## Objective
Implement row-wise layer normalization with block reductions.

## Your task
1. Open `starter.cu`.
2. Implement the required CUDA kernel(s) and host-side setup.
3. Add correctness checking against a CPU reference or an invariant.
4. Test at least one non-round input size where relevant.
5. Compare your implementation with `../../solutions/15_layernorm.cu` only after attempting it yourself.

## Concepts
- mean/variance reduction
- numerical stability
- AI primitive

## Stretch challenge
Fuse affine gamma/beta parameters and vectorize loads.

## Build only the solved reference
From the repository root:
```bash
cmake -S . -B build
cmake --build build -j
./build/15_layernorm
```

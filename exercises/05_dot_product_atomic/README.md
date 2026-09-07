# Dot Product with Atomics

**Level:** Intermediate

## Objective
Compute a vector dot product using per-block shared-memory reduction and a global atomicAdd.

## Your task
1. Open `starter.cu`.
2. Implement the required CUDA kernel(s) and host-side setup.
3. Add correctness checking against a CPU reference or an invariant.
4. Test at least one non-round input size where relevant.
5. Compare your implementation with `../../solutions/05_dot_product_atomic.cu` only after attempting it yourself.

## Concepts
- shared memory
- synchronization
- atomics

## Stretch challenge
Replace atomicAdd with a second-stage reduction.

## Build only the solved reference
From the repository root:
```bash
cmake -S . -B build
cmake --build build -j
./build/05_dot_product_atomic
```

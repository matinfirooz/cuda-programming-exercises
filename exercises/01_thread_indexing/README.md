# Thread & Grid Indexing

**Level:** Easy

## Objective
Map CUDA threads to a 1D array and write each element's global thread index.

## Your task
1. Open `starter.cu`.
2. Implement the required CUDA kernel(s) and host-side setup.
3. Add correctness checking against a CPU reference or an invariant.
4. Test at least one non-round input size where relevant.
5. Compare your implementation with `../../solutions/01_thread_indexing.cu` only after attempting it yourself.

## Concepts
- threadIdx/blockIdx/blockDim
- 1D grid-stride thinking
- bounds checks

## Stretch challenge
Try N values that are not multiples of the block size.

## Build only the solved reference
From the repository root:
```bash
cmake -S . -B build
cmake --build build -j
./build/01_thread_indexing
```


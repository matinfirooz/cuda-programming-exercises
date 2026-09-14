# Fused Scaled Dot-Product Attention

**Level:** Very hard

## Objective
Implement a small fused attention kernel: score QK^T, stable softmax, and weighted V without materializing the full score matrix in global memory.

## Your task
1. Open `starter.cu`.
2. Implement the required CUDA kernel(s) and host-side setup.
3. Add correctness checking against a CPU reference or an invariant.
4. Test at least one non-round input size where relevant.
5. Compare your implementation with `../../solutions/20_fused_attention.cu` only after attempting it yourself.

## Concepts
- fusion
- shared memory
- softmax
- attention
- AI kernel design

## Stretch challenge
Add masking, multiple heads, and a tiled K/V traversal for larger sequence lengths.

## Build only the solved reference
From the repository root:
```bash
cmake -S . -B build
cmake --build build -j
./build/20_fused_attention
```

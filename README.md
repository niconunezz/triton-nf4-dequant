# Triton-nf4-dequant

This repository provides an efficient and accurate implementation of NF4 dequantization using [Triton](https://github.com/openai/triton). It supports conversion to both FP16 and BF16 formats, and is designed to run on any modern GPU with minimal adjustments.

The implementation is written as a Jupyter Notebook, as it originated as a solution to the Unsloth Puzzle challenges. Despite its educational origin, the code is production-grade and intended to serve as a reference for NF4 dequantization using custom Triton kernels.

## Features

- Fast and correct NF4 to FP16/BF16 dequantization
- GPU-accelerated with Triton
- Portable across CUDA-compatible hardware
- Readable and modifiable code in notebook format

## License
This code is free to use and modify for any purpose, provided that proper credit is given to the author.

<div align="center">

<img src="figs/logo.png" alt="DART" width="220">

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

</div>



<h3 align="center">
    DART: Diffusion-Inspired Speculative Decoding for Fast LLM Inference
</h3>

---

## Overview

DART is a new speculative decoding approach for Large Language Models (LLMs) inference, which is inspired by diffusion-based Large Language Models (dLLMs). DART **surpasses EAGLE3 by 30% on average**, achieving up to 65% improvement on certain code-centric workloads.

DART's drafting requires only **a single forward pass** of **a single transformer layer** and a fast cpp-based tree search to build the draft token tree, resulting in extremely low drafting cost while preserving relatively high $\tau$ (Average Acceptance Length).

## Key Features

- **Single Forward Pass**: Produces multiple logits simultaneously with 1 forward of 1 layer.
- **N-gram Tree Search**: Uses n-gram–based tree search to build the final draft tree (C++ based).
- **Extremely Low Drafting Cost**: Results in extremely low drafting cost for efficient inference
- **Diffusion-Inspired**: Inspired by the newly emerging dllm series like LLADA in 2025

## Quick Start

### Installation

```bash
git clone https://github.com/fvliang/DART.git
cd DART
curl -LsSf https://astral.sh/uv/install.sh | sh (optimal, if you don't have a uv)
uv sync
uv pip install -e .
```

### Usage
a simple demo.
```python
uv python main.py
```

## Model Weights (HuggingFace)

<div align="center">
<table>
  <thead>
    <tr>
      <th>Base Model</th>
      <th>DART Adapted Weights</th>
      <th>N-gram Model</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-1.7B">Qwen3-1.7B</a></td>
      <td><a href="https://huggingface.co/fvliang/qwen1.7b-dart">fvliang/qwen1.7b-dart</a></td>
      <td><a href="https://huggingface.co/fvliang/dart-qwen3-ngram">fvliang/dart-qwen3-ngram</a></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-4B">Qwen3-4B</a></td>
      <td><a href="https://huggingface.co/fvliang/qwen4b-dart">fvliang/qwen4b-dart</a></td>
      <td><a href="https://huggingface.co/fvliang/dart-qwen3-ngram">fvliang/dart-qwen3-ngram</a></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-8B">Qwen3-8B</a></td>
      <td><a href="https://huggingface.co/fvliang/qwen8b-dart">fvliang/qwen8b-dart</a></td>
      <td><a href="https://huggingface.co/fvliang/dart-qwen3-ngram">fvliang/dart-qwen3-ngram</a></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-14B">Qwen3-14B</a></td>
      <td><a href="https://huggingface.co/fvliang/qwen14b-dart">fvliang/qwen14b-dart</a></td>
      <td><a href="https://huggingface.co/fvliang/dart-qwen3-ngram">fvliang/dart-qwen3-ngram</a></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-32B">Qwen3-32B</a></td>
      <td><a href="https://huggingface.co/fvliang/qwen32b-dart">fvliang/qwen32b-dart</a></td>
      <td><a href="https://huggingface.co/fvliang/dart-qwen3-ngram">fvliang/dart-qwen3-ngram</a></td>
    </tr>
  </tbody>
</table>
</div>

## Model Weights (ModelScope)

<div align="center">
<table>
  <thead>
    <tr>
      <th>Base Model</th>
      <th>DART Adapted Weights</th>
      <th>N-gram Model</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://modelscope.cn/models/Qwen/Qwen3-1.7B">Qwen3-1.7B</a></td>
      <td><a href="https://modelscope.cn/models/fvliang/Qwen3-1.7B-dart">fvliang/Qwen3-1.7B-dart</a></td>
      <td><a href="https://modelscope.cn/models/fvliang/dart-qwen3-ngram">fvliang/dart-qwen3-ngram</a></td>
    </tr>
    <tr>
      <td><a href="https://modelscope.cn/models/Qwen/Qwen3-4B">Qwen3-4B</a></td>
      <td><a href="https://modelscope.cn/models/fvliang/Qwen3-4B-dart">fvliang/Qwen3-4B-dart</a></td>
      <td><a href="https://modelscope.cn/models/fvliang/dart-qwen3-ngram">fvliang/dart-qwen3-ngram</a></td>
    </tr>
    <tr>
      <td><a href="https://modelscope.cn/models/Qwen/Qwen3-8B">Qwen3-8B</a></td>
      <td><a href="https://modelscope.cn/models/fvliang/Qwen3-8B-dart">fvliang/Qwen3-8B-dart</a></td>
      <td><a href="https://modelscope.cn/models/fvliang/dart-qwen3-ngram">fvliang/dart-qwen3-ngram</a></td>
    </tr>
    <tr>
      <td><a href="https://modelscope.cn/models/Qwen/Qwen3-14B">Qwen3-14B</a></td>
      <td><a href="https://modelscope.cn/models/fvliang/Qwen3-14B-dart">fvliang/Qwen3-14B-dart</a></td>
      <td><a href="https://modelscope.cn/models/fvliang/dart-qwen3-ngram">fvliang/dart-qwen3-ngram</a></td>
    </tr>
    <tr>
      <td><a href="https://modelscope.cn/models/Qwen/Qwen3-32B">Qwen3-32B</a></td>
      <td><a href="https://modelscope.cn/models/fvliang/Qwen3-32B-dart">fvliang/Qwen3-32B-dart</a></td>
      <td><a href="https://modelscope.cn/models/fvliang/dart-qwen3-ngram">fvliang/dart-qwen3-ngram</a></td>
    </tr>
  </tbody>
</table>
</div>

## Documentation

For detailed documentation, please refer to the [docs](docs/) directory.

<!-- ## Citation

If you find DART useful in your research, please cite:

```bibtex
@article{dart2025,
  title={DART: Diffusion-inspired Speculative Decoding for LLM},
  author={fuliang liu},
  year={2025}
}
``` -->

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<!-- ## Acknowledgments

This work is inspired by the dllm series, particularly LLADA. -->

---

<div align="center">

**If you find this project helpful, please give it a ⭐ Star!**

</div>

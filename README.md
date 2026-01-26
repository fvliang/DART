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

## Model Weights (Hugging Face)

| Base Model | DART Adapted Weights | N-gram Model |
| --- | --- | --- |
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | [fvliang/qwen1.7b-dart](https://huggingface.co/fvliang/qwen1.7b-dart) | [fvliang/dart-qwen3-ngram](https://huggingface.co/fvliang/dart-qwen3-ngram) |
| [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | [fvliang/qwen4b-dart](https://huggingface.co/fvliang/qwen4b-dart) | [fvliang/dart-qwen3-ngram](https://huggingface.co/fvliang/dart-qwen3-ngram) |
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | [fvliang/qwen8b-dart](https://huggingface.co/fvliang/qwen8b-dart) | [fvliang/dart-qwen3-ngram](https://huggingface.co/fvliang/dart-qwen3-ngram) |
| [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) | [fvliang/qwen14b-dart](https://huggingface.co/fvliang/qwen14b-dart) | [fvliang/dart-qwen3-ngram](https://huggingface.co/fvliang/dart-qwen3-ngram) |
| [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | [fvliang/qwen32b-dart](https://huggingface.co/fvliang/qwen32b-dart) | [fvliang/dart-qwen3-ngram](https://huggingface.co/fvliang/dart-qwen3-ngram) |

## Model Weights (ModelScope)

| Base Model | DART Adapted Weights | N-gram Model |
| --- | --- | --- |
| [Qwen3-1.7B](https://modelscope.cn/models/Qwen/Qwen3-1.7B) | [fvliang/Qwen3-1.7B-dart](https://modelscope.cn/models/fvliang/Qwen3-1.7B-dart) | [fvliang/dart-qwen3-ngram](https://modelscope.cn/models/fvliang/dart-qwen3-ngram) |
| [Qwen3-4B](https://modelscope.cn/models/Qwen/Qwen3-4B) | [fvliang/Qwen3-4B-dart](https://modelscope.cn/models/fvliang/Qwen3-4B-dart) | [fvliang/dart-qwen3-ngram](https://modelscope.cn/models/fvliang/dart-qwen3-ngram) |
| [Qwen3-8B](https://modelscope.cn/models/Qwen/Qwen3-8B) | [fvliang/Qwen3-8B-dart](https://modelscope.cn/models/fvliang/Qwen3-8B-dart) | [fvliang/dart-qwen3-ngram](https://modelscope.cn/models/fvliang/dart-qwen3-ngram) |
| [Qwen3-14B](https://modelscope.cn/models/Qwen/Qwen3-14B) | [fvliang/Qwen3-14B-dart](https://modelscope.cn/models/fvliang/Qwen3-14B-dart) | [fvliang/dart-qwen3-ngram](https://modelscope.cn/models/fvliang/dart-qwen3-ngram) |
| [Qwen3-32B](https://modelscope.cn/models/Qwen/Qwen3-32B) | [fvliang/Qwen3-32B-dart](https://modelscope.cn/models/fvliang/Qwen3-32B-dart) | [fvliang/dart-qwen3-ngram](https://modelscope.cn/models/fvliang/dart-qwen3-ngram) |

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

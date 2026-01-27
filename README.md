<div align="center">

<img src="figs/logo.png" alt="DART" width="220">

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.0.1-orange.svg)](#)
[![Maintained](https://img.shields.io/badge/maintained-yes-brightgreen.svg)](#)
[![Contributions](https://img.shields.io/badge/contributions-welcome-green.svg)](#)

</div>



<h3 align="center">
    Diffusion-Inspired Speculative Decoding for Fast LLM Inference
</h3>

---

## Overview

DART is a new speculative decoding approach for Large Language Models (LLMs) inference, which is inspired by diffusion-based Large Language Models (dLLMs). DART **surpasses EAGLE3 by 30% on average**, achieving up to 65% improvement on certain code-centric workloads.

DART's drafting requires only **a single forward pass** of **a single transformer layer** and a fast cpp-based tree search to build the draft token tree, resulting in extremely low drafting cost while preserving relatively high $\tau$ (Average Acceptance Length).

## Key Features

- **Fast drafting Forward**: Produces multiple logits simultaneously with 1 forward of 1 layer.
- **Fast Tree Search**: Uses n-gram–based tree search to build the final draft tree (C++ based).
- **Low Drafting Cost**: Results in extremely low drafting cost for efficient inference.
- **relatively high $\tau$**: Average Acceptance Length is competitive with EAGLE3.

## Quick Start

### Installation

```bash
git clone https://github.com/fvliang/DART.git
cd DART
curl -LsSf https://astral.sh/uv/install.sh | sh (optimal, if you don't have a uv)
uv sync
uv pip install -e .
```


## Model Weights (HuggingFace)

<div align="center">
<table>
  <thead>
    <tr>
      <th>Base Model</th>
      <th>DART Weights</th>
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
      <th>DART Weights</th>
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




### Inference

#### With UI (Gradio App)

We provide a Gradio web interface in `dart/app/app.py`. The easiest way is to run one of the prepared scripts:

```bash
bash dart/app/qwen3_1d7b_app.sh
```

You can also launch the app directly:

```bash
uv python dart/app/app.py \
  --base-model-name-or-path Qwen/Qwen3-4B \
  --dart-model-name-or-path fvliang/qwen4b-dart \
  --ngram-model-name-or-path fvliang/dart-qwen3-ngram \
  --template-name qwen \
  --device cuda \
  --max-new-tokens 2048 \
  --max-length 4096 \
  --use-small-ngram \
  --listen \
  --server-port 30000
```

After the model is fully loaded, Gradio will print a local URL in the terminal that you can open in your browser.

Tip: `--use-small-ngram` is great for fast testing. For best accuracy, omit it and load the full n-gram trie (this uses more memory and takes longer to load).

#### With Code (`main.py` / Python API)

You can use DART programmatically via `DartModel.from_pretrained(...)` and `dart_generate(...)`, similar to Hugging Face `generate`:

```python
import torch
from dart.model.dart_model import DartModel
from dart.model.template import TEMPLATE_REGISTRY

base_model_path = "Qwen/Qwen3-1.7B"
dart_model_path = "fvliang/qwen1.7b-dart"
ngram_model_path = "fvliang/dart-qwen3-ngram"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DartModel.from_pretrained(
    base_model_name_or_path=base_model_path,
    dart_model_name_or_path=dart_model_path,
    ngram_model_name_or_path=ngram_model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    is_small_ngram=True,
).to(device)
model.eval()

template = TEMPLATE_REGISTRY.get("qwen")
messages = [
    {"role": "system", "content": template.system_prompt},
    {"role": "user", "content": "Hello! Please introduce DART briefly."},
]
prompt = model.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

input_ids = model.tokenizer(
    prompt, return_tensors="pt", add_special_tokens=False
).input_ids.to(device)

output_ids = model.dart_generate(
    input_ids,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    max_new_token_num=512,
    max_length=2048,
)

output = model.tokenizer.decode(
    output_ids[0],
    skip_special_tokens=True,
    spaces_between_special_tokens=False,
    clean_up_tokenization_spaces=True,
)
print(output)
```


<!-- ## Documentation

For detailed documentation, please refer to the [docs](docs/) directory. -->

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

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

<!-- ## Acknowledgments

This work is inspired by the dllm series, particularly LLADA. -->

---

<div align="center">

**If you find this project helpful, please give it a ⭐ Star!**

</div>

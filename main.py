import argparse
import torch
from dart.model.template import TEMPLATE_REGISTRY
from dart.model.dart_model import DartModel

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args():
    parser = argparse.ArgumentParser(description="Run DART speculative decoding demo.")
    parser.add_argument(
        "--dart_model_name_or_path",
        default="fvliang/qwen1.7b-dart",
        help="Path or repo id of the DART draft model.",
    )
    parser.add_argument(
        "--base_model_name_or_path",
        default="Qwen/Qwen3-1.7B",
        help="Path or repo id of the base LLM.",
    )
    parser.add_argument(
        "--ngram_model_name_or_path",
        # default="fvliang/dart-qwen3-ngram",
        default="/data3/DART/ngram/qwen3",
        help="Path or repo id of the ngram model.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Target device, e.g. cpu / cuda / cuda:0. Default: auto-detect.",
    )
    return parser.parse_args()


def select_device(user_device: str | None) -> torch.device:
    """Select device with sanity checks."""
    if user_device:
        device = torch.device(user_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but no GPU is available.")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_prompt(tokenizer) -> str:
    test_prompt = "In at least 4000 words, write a personal reflection on a time you accomplished a goal"
    messages = [
        {"role": "system", "content": TEMPLATE_REGISTRY.get("qwen").system_prompt},
        {"role": "user", "content": test_prompt},
    ]
    conversation = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return conversation

COLORS = [
    "\033[92m",  # 绿色
    "\033[94m",  # 蓝色  
    "\033[93m",  # 黄色
    "\033[95m",  # 紫色
]
RESET = "\033[0m" 

def main():
    args = parse_args()
    device = select_device(args.device)
    is_small_ngram = True
    qwen_4b_dart = DartModel.from_pretrained(
        dart_model_name_or_path=args.dart_model_name_or_path,
        base_model_name_or_path=args.base_model_name_or_path,
        ngram_model_name_or_path=args.ngram_model_name_or_path,
        is_small_ngram=is_small_ngram,
    ).to(device)
    qwen_4b_dart.eval()

    conversation = build_prompt(qwen_4b_dart.tokenizer)
    print("Conversation:\n", conversation)

    input_ids = qwen_4b_dart.tokenizer(
        conversation, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    color_index = 0
    for text in qwen_4b_dart._dart_generate(
        input_ids,
        # temperature=1,
        # top_p=0.9,
        # top_k=64,
        max_new_token_num=1400,
        max_length=2400,
    ):
        current_color = COLORS[color_index % len(COLORS)]
        de_text = qwen_4b_dart.tokenizer.decode(
            text['id'],
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        print(f"{current_color}{de_text}{RESET}", end="", flush=True)
        if de_text.strip():
            color_index += 1
    print()


if __name__ == "__main__":
    main()
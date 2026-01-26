import functools
import numpy as np
import torch

from dart.tree_search.cpplib.cpp_ngram import load_cpp_ngram
from dart.tree_search.tree_search_config import get_config


def tree_search(
        logits, 
        preceding_ids, 
        dart_model,
        remain_total,
    ):

    device = logits.device
    draft_length = logits.shape[0]
    
    tree_search_config = get_config()
    topk_func = tree_search_config["topk_func"]
    beam_width = tree_search_config["beam_width"]
    level_weights_func = tree_search_config["level_weights_func"]
    ngram_weights = tree_search_config["ngram_weights"]
    logit_weights_func = tree_search_config["logit_weights_func"]
    
    temperature = 1.0
    d2t = dart_model.dart_layer.d2t.to(device)
    
    stop_token_id = dart_model.tokenizer.eos_token_id
    ngram_model = dart_model.ngram_model
    ngram_order = ngram_model.get_order()
    context_window = ngram_order - 1
    
    # === STEP 1: Preprocess logits with vectorization ===
    max_k = max(topk_func(i) for i in range(draft_length))
    topk_values_all, topk_indices_all = torch.topk(logits, max_k, dim=1)
    topk_indices_all = topk_indices_all + d2t[topk_indices_all]
    
    if temperature != 1.0:
        topk_values_all = torch.softmax(topk_values_all / temperature, dim=1)
    else:
        topk_values_all = torch.softmax(topk_values_all, dim=1)
    
    # Extract variable k efficiently
    topk_indices_list = []
    topk_values_list = []
    for i in range(draft_length):
        k = topk_func(i)
        topk_indices_list.append(topk_indices_all[i, :k].cpu().tolist())
        topk_values_list.append(topk_values_all[i, :k].cpu().tolist())
    
    prompt_tokens = preceding_ids.cpu().tolist()
    from dart.tree_search.cpplib.cpp_search import load_cpp_search
    cpp_search = load_cpp_search()
    searcher = cpp_search.Searcher(
        level_weights_func,
        logit_weights_func,
        ngram_weights,
        dart_model.tokenizer.eos_token_id,
        beam_width,
        remain_total,
    )

    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = searcher.search(
        draft_length,
        topk_indices_list,
        topk_values_list,
        ngram_model,
        prompt_tokens
    )

    return torch.LongTensor(draft_tokens), torch.LongTensor(retrieve_indices), torch.BoolTensor(tree_mask), torch.LongTensor(tree_position_ids)


def sequnce_cmp(seq1, seq2):
    """Comparison function for sequence sorting."""
    for i in range(min(len(seq1[0]), len(seq2[0]))):
        if seq1[0][i] != seq2[0][i]:
            return seq1[0][i] - seq2[0][i]
    return len(seq1[0]) - len(seq2[0])


def print_beam_search_tree(tree, tokenizer):
    """Print beam search tree for debugging."""
    tree = sorted(tree, key=functools.cmp_to_key(sequnce_cmp))

    def print_sub_tree(tree, level, tokenizer, prefix):
        if len(tree) == 0:
            return
        l = 0
        r = 0
        while l < len(tree):
            while r < len(tree) and tree[l][0][level] == tree[r][0][level]:
                r = r + 1
            conn = "├── " if r < len(tree) else "└── "
            print(
                prefix
                + conn
                + tokenizer.decode([tree[l][0][level]]).replace("\n", "\\n").strip()
                + f" ({tree[l][1]})"
            )
            if l + 1 < r:
                print_sub_tree(
                    tree[l + 1 : r],
                    level + 1,
                    tokenizer,
                    prefix + ("    " if r >= len(tree) else "│   "),
                )
            l = r

    print_sub_tree(tree, 0, tokenizer, "")

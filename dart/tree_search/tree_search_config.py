tree_search_config = {
        "topk_func": lambda level: 25,
        "beam_width": 20,
        "level_weights_func": lambda level: (level + 1) ** -0.7,
        "ngram_weights": 0.5,
        "logit_weights_func": lambda level: (0.9) ** (level),
    }

def get_config():
    return tree_search_config

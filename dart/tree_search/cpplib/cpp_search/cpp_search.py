# -*- coding: utf-8 -*-

import os

from torch.utils.cpp_extension import load

_abs_path = os.path.dirname(os.path.abspath(__file__))

cpp_search = load(
    name="cpp_search",
    sources=[
        f"{_abs_path}/search_binding.cpp",
        f"{_abs_path}/searcher.cpp",
        f"{_abs_path}/../cpp_ngram/trie_ngram.cpp",
        f"{_abs_path}/../cpp_ngram/aca_ngram.cpp",
        f"{_abs_path}/../cpp_utils/buffered_file_reader.cpp",
    ],
    extra_cflags=[
        "-O3", 
        "-std=c++20", 
        "-fopenmp", 
        "-DNO_NGRAM_ON_FIRST_TOKEN",
        "-DNGRAM_BATCH_LOAD",
    ],
)

def load_cpp_search():
    return cpp_search


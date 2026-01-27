# -*- coding: utf-8 -*-

import os

from torch.utils.cpp_extension import load

_abs_path = os.path.dirname(os.path.abspath(__file__))

cpp_ngram = load(
    name="ngram_cpp",
    sources=[
        f"{_abs_path}/ngram_binding.cpp",
        f"{_abs_path}/trie_ngram.cpp",
        f"{_abs_path}/../cpp_utils/buffered_file_reader.cpp",
    ],
    extra_cflags=[
        "-O3", 
        "-std=c++20", 
        "-fopenmp",
        "-DNGRAM_BATCH_LOAD",
    ],
)

def load_cpp_ngram():
    return cpp_ngram    


# -*- coding: utf-8 -*-

import os

from torch.utils.cpp_extension import load

_abs_path = os.path.dirname(os.path.abspath(__file__))

cpp_ngram = load(
    name="ngram_cpp",
    sources=[
        f"{_abs_path}/ngram_binding.cpp",
        f"{_abs_path}/trie_ngram.cpp",
        f"{_abs_path}/aca_ngram.cpp",
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

# main function
if __name__ == "__main__":

    ngram_cpp = load_cpp_ngram()
    conversation = [1, 2, 3, 2, 2, 1, 3, 2, 3, 1, 2]
    ngram = ngram_cpp.TrieNgram(3)  # order=3
    ngram.add_conversation(conversation)
    ngram.save("test_ngram.dat")
    ngram = ngram_cpp.TrieNgram.load("test_ngram.dat")

    conversation2 = [1, 2, 3]
    ngram2 = ngram_cpp.TrieNgram(3)  # order=3
    ngram2.add_conversation(conversation2)
    ngram.add_all(ngram2)

    print(ngram.get_probability([1, 2], [1, 2, 3]))
    print(ngram.get_probability([2], [1, 2, 3]))
    print(ngram.get_probability([2, 3], [1, 2, 3]))

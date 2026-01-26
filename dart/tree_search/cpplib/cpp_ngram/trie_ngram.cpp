#include "../inc/ngram.h"
#include "../inc/utils.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "omp.h"

namespace ngram {

TrieNgram::TrieNgram(size_t order_) {
  nodes.emplace_back();
  root = 0;
  order = order_;
}

void TrieNgram::add_sequence(const std::vector<token_t>& tokens) {
  auto cursor_id = root;
  for (const auto token : tokens) {
    nodes[cursor_id].freq++;

    auto iter = nodes[cursor_id].children.find(token);
    if (iter == nodes[cursor_id].children.end()) {
      auto node_id = nodes.size();
      nodes.emplace_back();
      nodes.back().parent = cursor_id;
      nodes.back().token = token;
      nodes[cursor_id].children[token] = node_id;
      cursor_id = node_id;
    } else {
      cursor_id = iter->second;
    }
  }
  nodes[cursor_id].freq++;
}

void TrieNgram::add_conversation(const std::vector<token_t>& conversation) {
  size_t conv_length = conversation.size();
  for (size_t start = 0; start < conv_length; ++start) {
    size_t end = std::min(conv_length, start + order);
    std::vector<token_t> slice(conversation.begin() + start, conversation.begin() + end);
    add_sequence(slice);
  }
}

size_t TrieNgram::get_order() const {
  return order;
}

std::pair<std::vector<prob_t>, std::vector<size_t>>
TrieNgram::get_probability(const std::vector<token_t>& context, const std::vector<token_t>& tokens) const {
  // omp_set_num_threads(NUM_THREADS);

  const size_t num_tokens = tokens.size();
  const size_t ctx_size = context.size();

  auto probabilities = std::vector<prob_t>(num_tokens, 0.0f);
  auto matched_lengths = std::vector<size_t>(num_tokens, 0);

  if (ctx_size == 0 || num_tokens == 0) {
    return make_pair(probabilities, matched_lengths);
  }

  size_t cursor_id = root;
  size_t remaining = num_tokens;

  for (size_t len = std::min(order - 1, ctx_size); len > 0; --len) {
    size_t start_pos = ctx_size - len;

    cursor_id = root;
    bool found = true;
    for (size_t i = start_pos; i < ctx_size; ++i) {
      auto iter = nodes[cursor_id].children.find(context[i]);
      if (iter == nodes[cursor_id].children.end()) {
        found = false;
        break;
      }
      cursor_id = iter->second;
    }

    if (!found) {
      continue;
    }

    freq_t context_freq = nodes[cursor_id].freq;
    // prob_t inv_freq = 1.0f / logf(static_cast<prob_t>(context_freq) + 1);
    prob_t inv_freq = 1.0f / static_cast<prob_t>(context_freq);

    for (size_t i = 0; i < num_tokens; ++i) {
      if (matched_lengths[i] > 0) {
        continue;
      }

      auto iter = nodes[cursor_id].children.find(tokens[i]);
      if (iter != nodes[cursor_id].children.end()) {
        // probabilities[i] = logf(static_cast<prob_t>(nodes[iter->second].freq) + 1) * inv_freq;
        probabilities[i] = static_cast<prob_t>(nodes[iter->second].freq) * inv_freq;
        matched_lengths[i] = len;
        --remaining;
      }
    }
  }

  return make_pair(probabilities, matched_lengths);
}

TrieNgram TrieNgram::load(const std::string& filename) {
  auto ifs = std::ifstream(filename, std::ios::binary);
  if (!ifs) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

#ifdef NGRAM_BATCH_LOAD
  auto reader = utils::BufferedFileReader(filename);
  // static const size_t buffer_size = (sizeof(token_t) + sizeof(size_t)) * (1 << 18);
  // auto buffer = std::unique_ptr<char[]>(new char[buffer_size]);
#endif

  // read header
#ifdef NGRAM_BATCH_LOAD
  size_t order = *reinterpret_cast<const size_t*>(reader.next(sizeof(size_t)));
  size_t node_count = *reinterpret_cast<const size_t*>(reader.next(sizeof(size_t)));
  TrieNgram ngram = TrieNgram(order);
  ngram.nodes.resize(node_count);
#else
  size_t order;
  ifs.read(reinterpret_cast<char*>(&order), sizeof(size_t));
  size_t node_count;
  ifs.read(reinterpret_cast<char*>(&node_count), sizeof(size_t));
  TrieNgram ngram = TrieNgram(order);
  ngram.nodes.resize(node_count);
#endif

  // read nodes
  for (size_t i = 0; i < node_count; ++i) {
    auto& node = ngram.nodes[i];

    // read node header
#ifdef NGRAM_BATCH_LOAD
    node.token = *reinterpret_cast<const token_t*>(reader.next(sizeof(token_t)));
    node.parent = *reinterpret_cast<const size_t*>(reader.next(sizeof(size_t)));
    node.freq = *reinterpret_cast<const freq_t*>(reader.next(sizeof(freq_t)));
    size_t children_size = *reinterpret_cast<const size_t*>(reader.next(sizeof(size_t)));

#else
    ifs.read(reinterpret_cast<char*>(&node.token), sizeof(token_t));
    ifs.read(reinterpret_cast<char*>(&node.parent), sizeof(size_t));
    ifs.read(reinterpret_cast<char*>(&node.freq), sizeof(freq_t));
    size_t children_size;
    ifs.read(reinterpret_cast<char*>(&children_size), sizeof(size_t));
#endif

    // read children
#ifdef NGRAM_BATCH_LOAD
    for (size_t j = 0; j < children_size; ++j) {
      token_t child_token = *reinterpret_cast<const token_t*>(reader.next(sizeof(token_t)));
      size_t child_id = *reinterpret_cast<const size_t*>(reader.next(sizeof(size_t)));
      node.children[child_token] = child_id;
    }
#else
    for (size_t j = 0; j < children_size; ++j) {
      token_t child_token;
      size_t child_id;
      ifs.read(reinterpret_cast<char*>(&child_token), sizeof(token_t));
      ifs.read(reinterpret_cast<char*>(&child_id), sizeof(size_t));
      node.children[child_token] = child_id;
    }
#endif
    if (i % 100000 == 0 || i == node_count - 1) {
      std::cout << "Loaded " << (i + 1) << " / " << node_count << " nodes.\r" << std::flush;
    }
  }
  ngram.root = 0;
  return ngram;
}

void TrieNgram::save(const std::string& filename) const {
  auto ofs = std::ofstream(filename, std::ios::binary);
  if (!ofs) {
    throw std::runtime_error("Failed to open file: " + filename);
  }
  ofs.write(reinterpret_cast<const char*>(&order), sizeof(size_t));
  size_t node_count = nodes.size();
  ofs.write(reinterpret_cast<const char*>(&node_count), sizeof(size_t));
  for (const auto& node : nodes) {
    ofs.write(reinterpret_cast<const char*>(&node.token), sizeof(token_t));
    ofs.write(reinterpret_cast<const char*>(&node.parent), sizeof(size_t));
    ofs.write(reinterpret_cast<const char*>(&node.freq), sizeof(freq_t));
    size_t children_size = node.children.size();
    ofs.write(reinterpret_cast<const char*>(&children_size), sizeof(size_t));
    for (const auto& [child_token, child_id] : node.children) {
      ofs.write(reinterpret_cast<const char*>(&child_token), sizeof(token_t));
      ofs.write(reinterpret_cast<const char*>(&child_id), sizeof(size_t));
    }
    if ((&node - &nodes[0]) % 100000 == 0 || &node - &nodes[0] == node_count - 1) {
      std::cout << "Saved " << (&node - &nodes[0] + 1) << " / " << node_count << " nodes.\r" << std::flush;
    }
  }
}


void TrieNgram::recursive_merge(size_t node_id, const TrieNgram& other, size_t other_node_id) {
  nodes[node_id].freq += other.nodes[other_node_id].freq;
  for (const auto& [token, other_child_id] : other.nodes[other_node_id].children) {
    auto iter = nodes[node_id].children.find(token);
    if (iter == nodes[node_id].children.end()) {
      // Create new child
      auto new_child_id = nodes.size();
      nodes.emplace_back();
      nodes[new_child_id].parent = node_id;
      nodes[new_child_id].token = token;
      nodes[node_id].children[token] = new_child_id;

      // Recursively add all children from other
      recursive_merge(new_child_id, other, other_child_id);
    } else {
      // Merge existing child
      size_t child_id = iter->second;
      recursive_merge(child_id, other, other_child_id);
    }
  }
}

void TrieNgram::add_all(const TrieNgram& other) {
  recursive_merge(root, other, other.root);
}

freq_t TrieNgram::recursive_reduce(size_t node_id, size_t reduced_node_id, freq_t threshold, TrieNgram& reduced_ngram) const {
  const TrieNode& current_node = nodes[node_id];
  freq_t reduced_size = 0;
#define reduced_node reduced_ngram.nodes[reduced_node_id]

  for (const auto& [token, child_id] : current_node.children) {
    const TrieNode& child_node = nodes[child_id];
    if (child_node.freq >= threshold) {
      // Add child to reduced n-gram
      size_t new_child_id = reduced_ngram.nodes.size();
      reduced_ngram.nodes.emplace_back();
      if (new_child_id % 100000 == 0) {
        std::cout << "Reduced " << new_child_id << " nodes.\r" << std::flush;
      }
      reduced_ngram.nodes[new_child_id].parent = reduced_node_id;
      reduced_ngram.nodes[new_child_id].token = token;
      reduced_node.children[token] = new_child_id;
      reduced_ngram.nodes[new_child_id].freq = child_node.freq;

      // Recursively reduce children
      freq_t child_reduced_size = recursive_reduce(child_id, new_child_id, threshold, reduced_ngram);
      reduced_size += child_reduced_size;
    } else {
      reduced_size += child_node.freq;
    }
  }
  reduced_node.freq -= reduced_size;
  return reduced_size;
#undef reduced_node
}

TrieNgram TrieNgram::reduce(freq_t threshold) const {
  TrieNgram reduced_ngram(order);
  recursive_reduce(root, reduced_ngram.root, threshold, reduced_ngram);
  std::cout << "Total reduced nodes: " << reduced_ngram.nodes.size() << std::endl;
  return reduced_ngram;
}

}  // namespace ngram

int main() {
  ngram::TrieNgram ngram(3);
  ngram.add_conversation({1, 2, 3, 2, 2, 1, 3, 2, 3, 1, 2});

  auto result = ngram.get_probability({2, 3}, {1, 2, 3});
  auto probs = result.first;
  auto matched_lengths = result.second;
  for (size_t i = 0; i < probs.size(); ++i) {
    std::cout << "Probability of token " << (i + 3) << ": " << probs[i] << std::endl;
    std::cout << "Matched length of token " << (i + 3) << ": " << matched_lengths[i] << std::endl;
  }

  return 0;

}

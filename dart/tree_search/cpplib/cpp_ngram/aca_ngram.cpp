#include "../inc/ngram.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <queue>
#include <stdexcept>

namespace ngram {

ACANgram::ACANgram(size_t order_) {
  nodes.emplace_back();
  root = 0;
  order = order_;
}

ACAState ACANgram::get_empty_state() const {
  return root;
}

size_t ACANgram::get_order() const {
  return order;
}

std::tuple<std::vector<ACAState>, std::vector<prob_t>, std::vector<size_t>>
ACANgram::get_probability(const ACAState& state, 
                          const std::vector<token_t>& new_context, 
                          const std::vector<token_t>& tokens) const {
  ACAState cursor = state;
  for (const auto& token : new_context) {
    auto iter = nodes[cursor].children.find(token);
    while (cursor != root && iter == nodes[cursor].children.end()) {
      cursor = nodes[cursor].fail;
      iter = nodes[cursor].children.find(token);
    }
    if (iter != nodes[cursor].children.end()) {
      cursor = iter->second;
    }
  }

  const size_t num_tokens = tokens.size();
  size_t remaining = num_tokens;
  auto probabilities = std::vector<prob_t>(num_tokens, 0.0f);
  auto matched_lengths = std::vector<size_t>(num_tokens, 0);

  std::vector<ACAState> new_state(num_tokens, cursor);
  if (num_tokens == 0) {
    return std::make_tuple(new_state, probabilities, matched_lengths);
  }
  
  while (cursor != root && remaining > 0) {
    size_t len = nodes[cursor].depth;
    prob_t inv_freq = 1.0f / static_cast<prob_t>(nodes[cursor].freq);
    for (size_t i = 0; i < num_tokens; ++i) {
      if (matched_lengths[i] > 0) {
        continue;
      }

      auto iter = nodes[cursor].children.find(tokens[i]);
      if (iter != nodes[cursor].children.end()) {
        probabilities[i] = static_cast<prob_t>(nodes[iter->second].freq) * inv_freq;
        matched_lengths[i] = len;
        new_state[i] = iter->second;
        --remaining;
      }
    }
    cursor = nodes[cursor].fail;
  }
  
  return std::make_tuple(new_state, probabilities, matched_lengths);
}

ACANgram ACANgram::load(const std::string& filename) {
  auto ifs = std::ifstream(filename, std::ios::binary);
  if (!ifs) {
    throw std::runtime_error("Failed to open file: " + filename);
  }
  size_t order;
  ifs.read(reinterpret_cast<char*>(&order), sizeof(size_t));
  size_t node_count;
  ifs.read(reinterpret_cast<char*>(&node_count), sizeof(size_t));
  ACANgram ngram = ACANgram(order);
  ngram.nodes.resize(node_count);
  for (size_t i = 0; i < node_count; ++i) {
    auto& node = ngram.nodes[i];
    ifs.read(reinterpret_cast<char*>(&node.token), sizeof(token_t));
    ifs.read(reinterpret_cast<char*>(&node.parent), sizeof(size_t));
    ifs.read(reinterpret_cast<char*>(&node.fail), sizeof(size_t));
    ifs.read(reinterpret_cast<char*>(&node.depth), sizeof(size_t));
    ifs.read(reinterpret_cast<char*>(&node.freq), sizeof(freq_t));
    size_t children_size;
    ifs.read(reinterpret_cast<char*>(&children_size), sizeof(size_t));
    for (size_t j = 0; j < children_size; ++j) {
      token_t child_token;
      size_t child_id;
      ifs.read(reinterpret_cast<char*>(&child_token), sizeof(token_t));
      ifs.read(reinterpret_cast<char*>(&child_id), sizeof(size_t));
      node.children[child_token] = child_id;
    }
    if (i % 100000 == 0 || i == node_count - 1) {
      std::cout << "Loaded " << (i + 1) << " / " << node_count << " nodes.\r" << std::flush;
    }
  }
  ngram.root = 0;
  return ngram;
}

void ACANgram::save(const std::string& filename) const {
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
    ofs.write(reinterpret_cast<const char*>(&node.fail), sizeof(size_t));
    ofs.write(reinterpret_cast<const char*>(&node.depth), sizeof(size_t));
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

void ACANgram::recursive_build_from_trie(const TrieNgram& trie_ngram, size_t trie_node_id, size_t aca_node_id) {
  const TrieNode& trie_node = trie_ngram.nodes[trie_node_id];
#define aca_node nodes[aca_node_id]

  for (const auto& [token, trie_child_id] : trie_node.children) {
    size_t aca_child_id = nodes.size();
    nodes.emplace_back();
    nodes[aca_child_id].token = token;
    nodes[aca_child_id].parent = aca_node_id;
    nodes[aca_child_id].fail = root;
    nodes[aca_child_id].depth = aca_node.depth + 1;
    nodes[aca_child_id].freq = trie_ngram.nodes[trie_child_id].freq;
    aca_node.children[token] = aca_child_id;

    if (aca_child_id % 100000 == 0) {
      std::cout << "Built " << aca_child_id << " ACA nodes.\r" << std::flush;
    }

    recursive_build_from_trie(trie_ngram, trie_child_id, aca_child_id);
  }
#undef aca_node
}

ACANgram ACANgram::build_from_trie(const TrieNgram& trie_ngram) {
  ACANgram aca_ngram(trie_ngram.get_order());

  aca_ngram.nodes[aca_ngram.root].token = trie_ngram.nodes[trie_ngram.root].token;
  aca_ngram.nodes[aca_ngram.root].parent = aca_ngram.root;
  aca_ngram.nodes[aca_ngram.root].fail = aca_ngram.root;
  aca_ngram.nodes[aca_ngram.root].depth = 0;
  aca_ngram.nodes[aca_ngram.root].freq = trie_ngram.nodes[trie_ngram.root].freq;
  aca_ngram.recursive_build_from_trie(trie_ngram, trie_ngram.root, aca_ngram.root);
  
  std::queue<size_t> q;
  q.push(aca_ngram.root);
  int cnt = 0;
  while (!q.empty()) {
    size_t current_id = q.front();
    q.pop();

    ++cnt;
    if (cnt % 100000 == 0) {
      std::cout << "Computed fail links for " << cnt << " ACA nodes.\r" << std::flush;
    }

    const ACANode& current_node = aca_ngram.nodes[current_id];
    if (current_node.depth > 1) {
      size_t cursor = aca_ngram.nodes[current_node.parent].fail;
      while (cursor != aca_ngram.root &&
            aca_ngram.nodes[cursor].children.find(current_node.token) == aca_ngram.nodes[cursor].children.end()) {
        cursor = aca_ngram.nodes[cursor].fail;
      }
      if (aca_ngram.nodes[cursor].children.find(current_node.token) != aca_ngram.nodes[cursor].children.end()) {
        aca_ngram.nodes[current_id].fail = aca_ngram.nodes[cursor].children.at(current_node.token);
      } else {
        aca_ngram.nodes[current_id].fail = aca_ngram.root;
      }
    }
    for (const auto& [token, child_id] : aca_ngram.nodes[current_id].children) {
      q.push(child_id);
    }
  }

  return aca_ngram;
}


}
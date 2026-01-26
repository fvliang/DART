#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "defs.h"

namespace ngram {

struct TrieNode {
  std::unordered_map<token_t, size_t> children;
  token_t token;
  size_t parent;
  freq_t freq;
  TrieNode() : token(0), parent(0), freq(0) {}
};

class TrieNgram {
  friend class ACANgram;

private:
  std::vector<TrieNode> nodes;
  size_t root;
  size_t order;

  void add_sequence(const std::vector<token_t>& tokens);
  void recursive_merge(size_t node_id, const TrieNgram& other, size_t other_node_id);
  freq_t recursive_reduce(size_t node_id, size_t reduced_node_id, freq_t threshold, TrieNgram& reduced_ngram) const;
 public:
  TrieNgram(size_t order_);

  size_t get_order() const;
  void add_conversation(const std::vector<token_t>& conversation);
  std::pair<std::vector<prob_t>, std::vector<size_t>>
  get_probability(const std::vector<token_t>& context, const std::vector<token_t>& tokens) const;
  static TrieNgram load(const std::string& filename);
  void save(const std::string& filename) const;
  void add_all(const TrieNgram& other);
  TrieNgram reduce(freq_t threshold) const;
};

struct ACANode {
  std::unordered_map<token_t, size_t> children;
  token_t token;
  size_t parent, fail, depth;
  freq_t freq;
  ACANode() : token(0), parent(0), fail(0), depth(0), freq(0) {}
};

typedef size_t ACAState;

class ACANgram {
private:
  std::vector<ACANode> nodes;
  size_t root;
  size_t order;

  void recursive_build_from_trie(const TrieNgram& trie_ngram, size_t trie_node_id, size_t aca_node_id);
 public:
  ACANgram(size_t order_);

  ACAState get_empty_state() const;
  size_t get_order() const;
  std::tuple<std::vector<ACAState>, std::vector<prob_t>, std::vector<size_t>>
  get_probability(const ACAState& state, const std::vector<token_t>& new_context, const std::vector<token_t>& tokens) const;
  static ACANgram load(const std::string& filename);
  void save(const std::string& filename) const;
  static ACANgram build_from_trie(const TrieNgram& trie_ngram);
};

}  // namespace ngram

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

}  // namespace ngram

#ifndef SEARCHER_H
#define SEARCHER_H

#include "defs.h"
#include "ngram.h"

#include <functional>
#include <tuple>
#include <vector>

namespace searcher {

class Searcher {
private:
  std::function<prob_t(size_t)> level_weight_func;
  std::function<prob_t(size_t)> logit_weight_func;
  prob_t ngram_weight;
  token_t eos_token;
  size_t beam_width;
  size_t remain_total;

  typedef std::vector<token_t> draft_tokens_t;
  typedef std::vector<std::vector<token_t>> retrieve_indices_t;
  typedef std::vector<std::vector<bool>> tree_mask_t;
  typedef std::vector<size_t> tree_positions_t;
  
  std::tuple<draft_tokens_t, retrieve_indices_t, tree_mask_t, tree_positions_t>
  encode_tree(std::vector<std::pair<std::vector<token_t>, prob_t>>& tree,
              token_t first_token);

public:
  Searcher(std::function<prob_t(size_t)> level_weight_func_, 
           std::function<prob_t(size_t)> logit_weight_func_,
           prob_t ngram_weight_, token_t eos_token_, size_t beam_width_, size_t remain_total_);

  std::tuple<draft_tokens_t, retrieve_indices_t, tree_mask_t, tree_positions_t>
  search(size_t draft_length,
         const std::vector<std::vector<token_t>>& topk_tokens,
         const std::vector<std::vector<prob_t>>& topk_logit_scores,
         const ngram::TrieNgram& ngram_model,
         const std::vector<token_t>& prompt_tokens);
  
  std::tuple<draft_tokens_t, retrieve_indices_t, tree_mask_t, tree_positions_t>
  search(size_t draft_length,
         const std::vector<std::vector<token_t>>& topk_tokens,
         const std::vector<std::vector<prob_t>>& topk_logit_scores,
         const ngram::ACANgram& ngram_model,
         const std::vector<token_t>& prompt_tokens);
};

} // namespace searcher


#endif // SEARCHER_H
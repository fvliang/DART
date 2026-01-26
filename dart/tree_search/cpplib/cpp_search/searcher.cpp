#include "../inc/searcher.h"

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <tuple>
#include <functional>

namespace searcher {

Searcher::Searcher(std::function<prob_t(size_t)> level_weight_func_, 
                   std::function<prob_t(size_t)> logit_weight_func_,
                   prob_t ngram_weight_, token_t eos_token_, size_t beam_width_, size_t remain_total_)
    : level_weight_func(level_weight_func_),
      logit_weight_func(logit_weight_func_),
      ngram_weight(ngram_weight_),
      eos_token(eos_token_),
      beam_width(beam_width_),
      remain_total(remain_total_) {}

std::tuple<Searcher::draft_tokens_t, Searcher::retrieve_indices_t, Searcher::tree_mask_t, Searcher::tree_positions_t>
Searcher::search(size_t draft_length,
                 const std::vector<std::vector<token_t>>& topk_tokens,
                 const std::vector<std::vector<prob_t>>& topk_logit_scores,
                 const ngram::TrieNgram& ngram_model,
                 const std::vector<token_t>& prompt_tokens) {

  std::vector<std::pair<std::vector<token_t>, prob_t>> all;
  std::vector<std::pair<std::vector<token_t>, prob_t>> candidates(1, {{}, 0.0f});
  std::vector<std::pair<std::vector<token_t>, prob_t>> next_candidates;

  omp_set_num_threads(NUM_THREADS);

#ifdef LOG_CTX_LEN
  static const int MAX_CTX_LEN = 5;
  static std::atomic<size_t> ctx_len_hist[MAX_CTX_LEN] = {0};
#endif

#ifdef LOG_NGRAM_TIME
    static auto period_range = std::chrono::milliseconds(200);
    static std::unique_ptr<std::chrono::high_resolution_clock::time_point> global_start_time = nullptr;
    if (global_start_time == nullptr) {
      global_start_time = std::make_unique<std::chrono::high_resolution_clock::time_point>();
      *global_start_time = std::chrono::high_resolution_clock::now();
    }
    static auto total_time = std::chrono::high_resolution_clock::duration::zero();
    static size_t ngram_calls = 0;
    static std::fstream ofs("ngram_time_log.txt", std::ios::out);
#endif

  for (size_t level = 0; level < draft_length; ++level) {
    const auto& tokens = topk_tokens[level];
    const auto& logit_scores = topk_logit_scores[level];
    
    size_t level_size = tokens.size() * candidates.size();
    next_candidates.resize(level_size);
    all.resize(all.size() + level_size);
    auto level_weight = level_weight_func(level);
    auto logit_weight = logit_weight_func(level);

#pragma omp parallel for
    for (size_t cand_idx = 0; cand_idx < candidates.size(); ++cand_idx) {
      const auto& cand_seq = candidates[cand_idx].first;
      auto cand_score = candidates[cand_idx].second;
      std::vector<token_t> context;
      context.reserve(ngram_model.get_order() - 1);
      if (cand_seq.size() >= ngram_model.get_order() - 1) {
        context.insert(context.end(), cand_seq.end() - (ngram_model.get_order() - 1), cand_seq.end());
      } else {
        context.insert(
          context.end(), 
          prompt_tokens.end() - std::min(ngram_model.get_order() - 1 - cand_seq.size(), prompt_tokens.size()), 
          prompt_tokens.end()
        );
        context.insert(context.end(), cand_seq.begin(), cand_seq.end());
      }

#ifdef NO_NGRAM_ON_FIRST_TOKEN 
      if (level == 0) {
        for (size_t token_idx = 0; token_idx < tokens.size(); ++token_idx) {
          auto logit_score = logit_scores[token_idx];

          auto combined_logit = logf(logit_score + 1e-10f) * (logit_weight);
          // auto combined_logit = logf(logit_score + 1e-10f) * (logit_weight + ngram_weight);

          prob_t total_score = cand_score + level_weight * combined_logit;

          std::vector<token_t> extended_seq;
          extended_seq.reserve(cand_seq.size() + 1);
          extended_seq.insert(extended_seq.end(), cand_seq.begin(), cand_seq.end());
          extended_seq.push_back(tokens[token_idx]);

          size_t next_idx = cand_idx * tokens.size() + token_idx;
          next_candidates[next_idx] = std::make_pair(extended_seq, total_score);
          all[all.size() - level_size + next_idx] = std::make_pair(std::move(extended_seq), total_score);
        }
      } else {
#endif
        
        std::vector<prob_t> ngram_scores;
        std::vector<size_t> ctx_lengths;
        if (ngram_weight > 0) {
#ifdef LOG_NGRAM_TIME
          auto ngram_start_time = std::chrono::high_resolution_clock::now();
#endif
          std::tie(ngram_scores, ctx_lengths) = ngram_model.get_probability(context, tokens);
#ifdef LOG_NGRAM_TIME
          auto ngram_end_time = std::chrono::high_resolution_clock::now();
#pragma omp critical
          {
            total_time += (ngram_end_time - ngram_start_time);
            ngram_calls += 1;
            auto since_start = ngram_end_time - *global_start_time;
            if (since_start >= period_range) {
              auto avg_time = (double) std::chrono::duration_cast<std::chrono::microseconds>(total_time).count() / ngram_calls;
              ofs << "Period Ending " << period_range.count()
                  << "ms: Avg N-gram Call Time = " << avg_time << " us over " << ngram_calls << " calls." << std::endl;
              ofs.flush();
              period_range += std::chrono::milliseconds(200);
              total_time = std::chrono::high_resolution_clock::duration::zero();
              ngram_calls = 0;
            }
          }
          
#endif
        } else {
          ngram_scores.resize(tokens.size(), 0.0f);
          ctx_lengths.resize(tokens.size(), 0);
        }

        for (size_t token_idx = 0; token_idx < tokens.size(); ++token_idx) {
          auto logit_score = logit_scores[token_idx];
          auto ngram_score = ngram_scores[token_idx];

#ifdef LOG_CTX_LEN
          size_t ctx_len = ctx_lengths[token_idx];
          if (ctx_len < MAX_CTX_LEN) {
            ctx_len_hist[ctx_len]++;
          }
#endif

          auto combined_logit = logf(logit_score + 1e-10f) * logit_weight;
          auto combined_ngram = logf(ngram_score + 1e-10f) * ngram_weight;

          prob_t total_score = cand_score + level_weight * (combined_logit + combined_ngram);

          std::vector<token_t> extended_seq;
          extended_seq.reserve(cand_seq.size() + 1);
          extended_seq.insert(extended_seq.end(), cand_seq.begin(), cand_seq.end());
          extended_seq.push_back(tokens[token_idx]);

          size_t next_idx = cand_idx * tokens.size() + token_idx;
          next_candidates[next_idx] = std::make_pair(extended_seq, total_score);
          all[all.size() - level_size + next_idx] = std::make_pair(std::move(extended_seq), total_score);
        }
#ifdef NO_NGRAM_ON_FIRST_TOKEN
      }
#endif
    }
    
    std::sort(next_candidates.begin(), next_candidates.end(), [&](const auto& a, const auto& b) {
      if ((a.first.back() == eos_token) != (b.first.back() == eos_token)) {
        return (a.first.back() != eos_token) > (b.first.back() != eos_token);
      } else {
        return a.second > b.second;
      }
    });

    if (next_candidates.size() > beam_width) {
      next_candidates.resize(beam_width);
    }
    std::swap(candidates, next_candidates);
  }

  std::sort(all.begin(), all.end(), [](const auto& a, const auto& b) {
    return a.second > b.second;
  });
  if (all.size() > remain_total) {
    all.resize(remain_total);
  }

#ifdef LOG_CTX_LEN
  std::cout << "Context Length Histogram:" << std::endl;
  for (size_t i = 0; i < MAX_CTX_LEN; ++i) {
    std::cout << "Length " << i << ": " << ctx_len_hist[i].load() << std::endl;
  }
#endif

  return encode_tree(all, prompt_tokens.back());
}

std::tuple<Searcher::draft_tokens_t, Searcher::retrieve_indices_t, Searcher::tree_mask_t, Searcher::tree_positions_t>
Searcher::search(size_t draft_length,
                 const std::vector<std::vector<token_t>>& topk_tokens,
                 const std::vector<std::vector<prob_t>>& topk_logit_scores,
                 const ngram::ACANgram& ngram_model,
                 const std::vector<token_t>& prompt_tokens) {

  std::vector<std::pair<std::vector<token_t>, prob_t>> all;
  std::vector<std::tuple<std::vector<token_t>, prob_t, ngram::ACAState>> candidates(1, {{}, 0.0f, ngram_model.get_empty_state()});
  std::vector<std::tuple<std::vector<token_t>, prob_t, ngram::ACAState>> next_candidates;
  
  for (size_t level = 0; level < draft_length; ++level) {
    const auto& tokens = topk_tokens[level];
    const auto& logit_scores = topk_logit_scores[level];
    
    size_t level_size = tokens.size() * candidates.size();
    next_candidates.resize(level_size);
    all.resize(all.size() + level_size);
    auto level_weight = level_weight_func(level);
    auto logit_weight = logit_weight_func(level);

#pragma omp parallel for
    for (size_t cand_idx = 0; cand_idx < candidates.size(); ++cand_idx) {
      const auto& cand_seq = std::get<0>(candidates[cand_idx]);
      auto cand_score = std::get<1>(candidates[cand_idx]);
      auto cand_state = std::get<2>(candidates[cand_idx]);
      std::vector<token_t> new_context(1);
      if (cand_seq.empty()) {
        new_context[0] = prompt_tokens.back();
      } else {
        new_context[0] = cand_seq.back();
      }
      
      // todo: handle NO_NGRAM_ON_FIRST_TOKEN case

      auto [new_states, ngram_scores, ctx_lengths] = ngram_model.get_probability(cand_state, new_context, tokens);
      for (size_t token_idx = 0; token_idx < tokens.size(); ++token_idx) {
        auto new_state = new_states[token_idx];
        auto logit_score = logit_scores[token_idx];
        auto ngram_score = ngram_scores[token_idx];

        auto combined_logit = logf(logit_score + 1e-10f) * logit_weight;
        auto combined_ngram = logf(ngram_score + 1e-10f) * ngram_weight;

        prob_t total_score = cand_score + level_weight * (combined_logit + combined_ngram);

        std::vector<token_t> extended_seq;
        extended_seq.reserve(cand_seq.size() + 1);
        extended_seq.insert(extended_seq.end(), cand_seq.begin(), cand_seq.end());
        extended_seq.push_back(tokens[token_idx]);

        size_t next_idx = cand_idx * tokens.size() + token_idx;
        next_candidates[next_idx] = std::make_tuple(extended_seq, total_score, new_state);
        all[all.size() - level_size + next_idx] = std::make_pair(std::move(extended_seq), total_score);
      }
    }
    
    std::sort(next_candidates.begin(), next_candidates.end(), [&](const auto& a, const auto& b) {
      if ((std::get<0>(a).back() == eos_token) != (std::get<0>(b).back() == eos_token)) {
        return (std::get<0>(a).back() != eos_token) > (std::get<0>(b).back() != eos_token);
      } else {
        return std::get<1>(a) > std::get<1>(b);
      }
    });

    if (next_candidates.size() > beam_width) {
      next_candidates.resize(beam_width);
    }
    std::swap(candidates, next_candidates);
  }

  std::sort(all.begin(), all.end(), [](const auto& a, const auto& b) {
    return a.second > b.second;
  });
  if (all.size() > remain_total) {
    all.resize(remain_total);
  }

  return encode_tree(all, prompt_tokens.back());
}

std::tuple<Searcher::draft_tokens_t, Searcher::retrieve_indices_t, Searcher::tree_mask_t, Searcher::tree_positions_t>
Searcher::encode_tree(std::vector<std::pair<std::vector<token_t>, prob_t>>& tree,
                      token_t first_token) {
  draft_tokens_t draft_tokens(tree.size() + 1);
  retrieve_indices_t retrieve_indices;
  tree_mask_t tree_mask;
  tree_positions_t tree_positions(tree.size() + 1);
  
  std::sort(tree.begin(), tree.end(), [](const auto& a, const auto& b) {
    return a.first.size() < b.first.size();
  });

  // build draft tokens, tree_mask, tree_positions
  draft_tokens[0] = first_token;
  tree_positions[0] = 0;
#pragma omp parallel for
  for (size_t i = 0; i < tree.size(); ++i) {
    draft_tokens[i + 1] = tree[i].first.back();
    tree_positions[i + 1] = tree[i].first.size();
  }

  // build retrieve indices
  size_t leaf_count = 0;
  size_t max_depth = 0;
  std::vector<std::pair<std::vector<token_t>, size_t>> ordered_tree;
  std::vector<size_t> leaf_ids(tree.size(), 0);

  for (size_t i = 0; i < tree.size(); ++i) {
    ordered_tree.push_back({tree[i].first, i});
    max_depth = std::max(max_depth, tree[i].first.size());
  }
  std::sort(ordered_tree.begin(), ordered_tree.end(), [](const auto& a, const auto& b) {
    for (size_t i = 0; i < std::min(a.first.size(), b.first.size()); ++i) {
      if (a.first[i] != b.first[i]) {
        return a.first[i] < b.first[i];
      }
    }
    return a.first.size() < b.first.size();
  });
  for (size_t i = 0; i < ordered_tree.size(); ++i) {
    if (i != ordered_tree.size() - 1
        && ordered_tree[i].first.size() < ordered_tree[i + 1].first.size()) {
      continue;
    }
    leaf_ids[i] = ++leaf_count;
  }

  retrieve_indices.resize(leaf_count);
  for (auto& vec : retrieve_indices) {
    vec.resize(max_depth + 1, -1);
  }
#pragma omp parallel for
  for (size_t i = 0; i < ordered_tree.size(); ++i) {
    size_t leaf_id = leaf_ids[i];
    if (leaf_id == 0) {
      continue;
    }
    size_t ptr = i;
    for (size_t len = ordered_tree[i].first.size(); len > 0; --len) {
      retrieve_indices[leaf_id - 1][len] = ordered_tree[ptr].second + 1;
      while (ptr > 0 && ordered_tree[ptr].first.size() >= len) {
        --ptr;
      }
    }
    retrieve_indices[leaf_id - 1][0] = 0;
  }

  // build tree_mask
  tree_mask.resize(tree.size() + 1);
  tree_mask[0].resize(tree.size() + 1, false);
  tree_mask[0][0] = true;
#pragma omp parallel for
  for (size_t i = 0; i < ordered_tree.size(); ++i) {
    size_t idx = ordered_tree[i].second + 1;
    tree_mask[idx].resize(tree.size() + 1, false);
    size_t ptr = i;
    for (size_t len = ordered_tree[i].first.size(); len > 0; --len) {
      tree_mask[idx][ordered_tree[ptr].second + 1] = true;
      while (ptr > 0 && ordered_tree[ptr].first.size() >= len) {
        --ptr;
      }
    }
    tree_mask[idx][0] = true;
  }
  
  return {draft_tokens, retrieve_indices, tree_mask, tree_positions};
}

} // namespace searcher
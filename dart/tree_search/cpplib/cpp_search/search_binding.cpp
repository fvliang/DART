#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../inc/searcher.h"

PYBIND11_MODULE(cpp_search, m) {
  using namespace searcher;
  namespace py = pybind11;
  m.doc() = "";

  py::class_<Searcher>(m, "Searcher")
      .def(
          py::init<std::function<prob_t(size_t)>, std::function<prob_t(size_t)>, prob_t, token_t, size_t, size_t>(),
          py::arg("level_weight_func"),
          py::arg("logit_weight_func"),
					py::arg("eos_token"),
          py::arg("ngram_weight"),
          py::arg("beam_width"),
          py::arg("remain_total"),
          "")
      .def(
          "search",
          py::overload_cast<
							size_t,
							const std::vector<std::vector<token_t>>&,
							const std::vector<std::vector<prob_t>>&,
							const ngram::TrieNgram&,
							const std::vector<token_t>&>(&Searcher::search),
          py::arg("draft_length"),
          py::arg("topk_tokens"),
          py::arg("topk_logit_scores"),
          py::arg("ngram_model"),
          py::arg("prompt_tokens"),
          "")
			.def(
					"search",
					py::overload_cast<
							size_t,
							const std::vector<std::vector<token_t>>&,
							const std::vector<std::vector<prob_t>>&,
							const ngram::ACANgram&,
							const std::vector<token_t>&>(&Searcher::search),
					py::arg("draft_length"),
					py::arg("topk_tokens"),
					py::arg("topk_logit_scores"),
					py::arg("ngram_model"),
					py::arg("prompt_tokens"),
					"");
}

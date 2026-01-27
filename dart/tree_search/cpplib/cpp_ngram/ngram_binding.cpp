#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../inc/ngram.h"

PYBIND11_MODULE(ngram_cpp, m) {
  using namespace ngram;
  namespace py = pybind11;
  m.doc() = "";

  py::class_<TrieNgram>(m, "TrieNgram")
      .def(py::init<size_t>(), py::arg("order"))
      .def("get_order", &TrieNgram::get_order, "")
      .def("add_conversation", &TrieNgram::add_conversation, "")
      .def("get_probability", &TrieNgram::get_probability, "")
      .def("save", &TrieNgram::save, "")
      .def_static("load", &TrieNgram::load, "")
      .def("add_all", &TrieNgram::add_all, "")
      .def("reduce", &TrieNgram::reduce, py::arg("threshold"), "");
}

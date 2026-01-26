#include "../inc/utils.h"

#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>

namespace utils {

BufferedFileReader::BufferedFileReader(const std::string& filename, size_t buffer_size_)
  : buffer_size(buffer_size_), data_size(0), pos(0) {
  buffer = std::make_unique<char[]>(buffer_size);
  file.open(filename, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open file: " + filename);
  }
  file.read(buffer.get(), buffer_size);
  data_size = file.gcount();
}

BufferedFileReader::~BufferedFileReader() = default;

const char* BufferedFileReader::next(size_t n) {
  if (pos + n > data_size) {
    size_t remaining = data_size - pos;
    std::memmove(buffer.get(), buffer.get() + pos, remaining);
    file.read(buffer.get() + remaining, buffer_size - remaining);
    data_size = remaining + file.gcount();
    pos = 0;
    if (pos + n > data_size) {
      throw std::runtime_error("Buffer underflow: requested " + std::to_string(n) + " bytes but only "
                               + std::to_string(data_size) + " available");
    }
  }
  size_t cur_pos = pos;
  pos += n;
  return buffer.get() + cur_pos;
}

};

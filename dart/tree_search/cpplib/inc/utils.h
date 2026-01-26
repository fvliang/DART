#ifndef UTILS_H
#define UTILS_H

#include <cstddef>
#include <fstream>
#include <memory>
#include <string>

namespace utils {

class BufferedFileReader {
private:
  std::unique_ptr<char[]> buffer;
  size_t buffer_size;
  size_t data_size;
  size_t pos;
  std::ifstream file;

public:
  BufferedFileReader(const std::string& filename, size_t buffer_size_ = 1 << 20);
  ~BufferedFileReader();

  const char* next(size_t n);
};

};

#endif // UTILS_H
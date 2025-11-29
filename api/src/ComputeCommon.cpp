#include <ComputeInterface.h>

#include <fstream>
#include <string>

namespace cut {

#define WRITE_MSG(prefix)                                                      \
  va_list args;                                                                \
  va_start(args, format);                                                      \
  /* Copy a prefix to the buffer */                                            \
  snprintf(msg, sizeof(msg), prefix);                                          \
  /* Copy args to the buffer */                                                \
  vsnprintf(&msg[std::strlen(prefix)], sizeof(msg) - std::strlen(prefix),      \
            format, args);                                                     \
  va_end(args);

void logMsg(const char *format, ...) {
  char msg[256];
  WRITE_MSG("Info: ")
  printf("\n%s\n", msg);
}

void logMsg(const char *header, const std::vector<const char *> &lines) {
  std::string msg(header);
  msg += ":";
  for (const auto line : lines) {
    msg += std::string("\n") + line;
  }
  printf("\n%s\n", msg.c_str());
}

void logMsg(const char *header, const std::vector<std::string> &lines) {
  std::string msg(header);
  msg += ":";
  for (const auto &line : lines) {
    msg += "\n" + line;
  }
  printf("\n%s\n", msg.c_str());
}

void logErr(const char *format, ...) {
  char msg[256];
  WRITE_MSG("Error: ")
  throw std::runtime_error(msg);
}

std::vector<char> readShaderFile(const std::string &filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    logErr("Failed to open shader file: %s", filename.c_str());
  }

  size_t fileSize = static_cast<size_t>(file.tellg());
  std::vector<char> buffer(fileSize);

  file.seekg(0);
  file.read(buffer.data(), fileSize);
  file.close();

  return buffer;
}

} // namespace cut

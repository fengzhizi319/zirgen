// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/compiler/zkp/util.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace zirgen {

std::vector<uint8_t> loadFile(const std::string& path) {
  std::ifstream is(path, std::ios::binary);
  is.exceptions(std::ios_base::failbit | std::ios_base::badbit);
  is.seekg(0, std::ios::end);
  size_t nbytes = is.tellg();
  is.seekg(0, std::ios::beg);
  std::vector<uint8_t> elfContents(nbytes, 0);
  is.read(reinterpret_cast<char*>(elfContents.data()), nbytes);
  is.close();
  return elfContents;
}

} // namespace zirgen

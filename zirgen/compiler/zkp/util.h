// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

/// \file
/// Small utility functions, mostly common math routines.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace zirgen {

/// Compute `ceil(a / b)` via truncated integer division.
inline size_t constexpr ceilDiv(size_t a, size_t b) {
  return (a + (b - 1)) / b;
}

/// Round `a` up to the nearest multipe of `b`.
inline size_t constexpr roundUp(size_t a, size_t b) {
  return ceilDiv(a, b) * b;
}

/// Compute the smalled power `p` of x such that `x^p >= in`
inline size_t constexpr nearestPoX(size_t in, size_t x) {
  size_t r = 1;
  while (r < in) {
    r *= x;
  }
  return r;
}

/// Compute the smalled power `2` of x such that `2^p >= in`
inline size_t constexpr nearestPo2(size_t in) {
  size_t r = 1;
  while (r < in) {
    r *= 2;
  }
  return r;
}

/// Compute `ceil(log_x(in))`, i.e. find the smallest value `out` such that `x^out >= in`.
inline size_t constexpr logXCeil(size_t in, size_t x) {
  size_t r = 0;
  size_t c = 1;
  while (c < in) {
    r++;
    c *= x;
  }
  return r;
}

/// Compute `ceil(log_2(in))`, i.e. find the smallest value `out` such that `2^out >= in`.
inline size_t constexpr log2Ceil(size_t in) {
  size_t r = 0;
  while ((size_t(1) << r) < in) {
    r++;
  }
  return r;
}

/// True if `in` is a power of 2
inline bool constexpr isPo2(size_t in) {
  return (size_t(1) << log2Ceil(in)) == in;
}

// Reads an entire file into a buffer.
std::vector<uint8_t> loadFile(const std::string& path);

} // namespace zirgen

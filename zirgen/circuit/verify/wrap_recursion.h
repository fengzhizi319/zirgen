// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "zirgen/circuit/verify/verify.h"

namespace zirgen::verify {

std::unique_ptr<CircuitInterface> getInterfaceRecursion();

} // namespace zirgen::verify

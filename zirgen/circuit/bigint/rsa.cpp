// Copyright 2024 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "zirgen/circuit/bigint/rsa.h"
#include "zirgen/Dialect/BigInt/IR/BigInt.h"

using namespace mlir;

namespace zirgen::BigInt {

void genModPow65537(mlir::OpBuilder& builder, mlir::Location loc, size_t bitwidth) {
  // Check if (S^e = M (mod N)), where e = 65537
  // 检查 (S^e = M (mod N))，其中 e = 65537
  auto S = builder.create<BigInt::LoadOp>(loc, bitwidth, 11, 0); // 加载操作数 S
  auto N = builder.create<BigInt::LoadOp>(loc, bitwidth, 12, 0); // 加载模数 N
  // We square S 16 times to get S^65536
  // 将 S 平方 16 次以得到 S^65536
  Value x = S;
  for (size_t i = 0; i < 16; i++) {
    auto xm = builder.create<BigInt::MulOp>(loc, x, x); // 乘法操作
    x = builder.create<BigInt::ReduceOp>(loc, xm, N); // 约简操作
  }
  // Multiply in one more copy of S + reduce
  // 再乘以一个 S 并约简
  auto xm = builder.create<BigInt::MulOp>(loc, x, S); // 乘法操作
  x = builder.create<BigInt::ReduceOp>(loc, xm, N); // 约简操作
  // this is our result
  // 这是我们的结果
  builder.create<BigInt::StoreOp>(loc, x, 13, 0); // 存储结果
}

// Used for testing, this RSA code uses `Def` instead of `Load`/`Store`
// 用于测试，这段 RSA 代码使用 `Def` 而不是 `Load`/`Store`
void makeRSAChecker(OpBuilder builder, Location loc, size_t bits) {
  // Check if (S^e = M (mod N)), where e = 65537
  // 检查 (S^e = M (mod N))，其中 e = 65537
  auto N = builder.create<BigInt::DefOp>(loc, bits, 0, true, bits - 1); // 定义模数 N
  auto S = builder.create<BigInt::DefOp>(loc, bits, 1, true); // 定义操作数 S
  auto M = builder.create<BigInt::DefOp>(loc, bits, 2, true); // 定义结果 M
  // We square S 16 times to get S^65536
  // 将 S 平方 16 次以得到 S^65536
  Value x = S;
  for (size_t i = 0; i < 16; i++) {
    auto xm = builder.create<BigInt::MulOp>(loc, x, x); // 乘法操作
    x = builder.create<BigInt::ReduceOp>(loc, xm, N); // 约简操作
  }
  // Multiply in one more copy of S + reduce
  // 再乘以一个 S 并约简
  auto xm = builder.create<BigInt::MulOp>(loc, x, S); // 乘法操作
  x = builder.create<BigInt::ReduceOp>(loc, xm, N); // 约简操作
  // Subtract M and see if it's zero
  // 减去 M 并检查是否为零
  auto diff = builder.create<BigInt::SubOp>(loc, x, M); // 减法操作
  builder.create<BigInt::EqualZeroOp>(loc, diff); // 检查是否为零
}

// Used for testing, to compute expected outputs.
// I verified this by comparing against:
// pow(S, 65537, N) in python
// 用于测试，计算预期输出。
// 我通过与 python 中的 pow(S, 65537, N) 进行比较来验证这一点
APInt RSA(APInt N, APInt S) {
  size_t width = S.getBitWidth();
  N = N.zext(2 * width); // 扩展 N 的位宽
  S = S.zext(2 * width); // 扩展 S 的位宽
  APInt cur = S;
  for (size_t i = 0; i < 16; i++) {
    cur = (cur * cur).urem(N); // 计算平方并取模
  }
  cur = (cur * S).urem(N); // 再乘以 S 并取模
  return cur.trunc(width); // 截断到原始位宽
}

} // namespace zirgen::BigInt

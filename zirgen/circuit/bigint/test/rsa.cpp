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
#include "zirgen/circuit/bigint/test/bibc.h"

#include <gtest/gtest.h>

using namespace zirgen;
using namespace zirgen::BigInt::test;

// 定义一个测试类，继承自BibcTest
TEST_F(BibcTest, RSA256) {
  // 创建MLIR操作构建器
  mlir::OpBuilder builder(ctx);
  // 创建一个名为"rsa_256"的函数
  auto func = makeFunc("rsa_256", builder);
  // 使用构建器和函数位置创建一个RSA检查器，位宽为256
  BigInt::makeRSAChecker(builder, func.getLoc(), 256);
  // 降低IR
  lower();

  // 定义RSA算法的输入值
  llvm::APInt N(64, 101);
  llvm::APInt S(64, 32766);
  llvm::APInt M(64, 53);
  // 断言RSA算法的输出是否与预期值相等
  EXPECT_EQ(M, BigInt::RSA(N, S));
  std::vector<llvm::APInt> inputs = {N, S, M};

  // 定义两个ZType变量
  ZType a, b;
  // 调用AB函数，传入函数和输入值，获取输出值a和b
  AB(func, inputs, a, b);
  // 断言a和b是否相等
  EXPECT_EQ(a, b);
}

// 定义另一个测试类，继承自BibcTest
TEST_F(BibcTest, RSA3072) {
  // 创建MLIR操作构建器
  mlir::OpBuilder builder(ctx);
  // 创建一个名为"rsa_3072"的函数
  auto func = makeFunc("rsa_3072", builder);
  // 使用构建器和函数位置创建一个RSA检查器，位宽为3072
  BigInt::makeRSAChecker(builder, func.getLoc(), 3072);
  // 降低IR
  lower();

  // 定义RSA算法的输入值
  llvm::APInt N(64, 22764235167642101);
  llvm::APInt S(64, 10116847215);
  llvm::APInt M(64, 14255570451702775);
  // 断言RSA算法的输出是否与预期值相等
  EXPECT_EQ(M, BigInt::RSA(N, S));
  std::vector<llvm::APInt> inputs = {N, S, M};

  // 定义两个ZType变量
  ZType a, b;
  // 调用AB函数，传入函数和输入值，获取输出值a和b
  AB(func, inputs, a, b);
  // 断言a和b是否相等
  EXPECT_EQ(a, b);
}
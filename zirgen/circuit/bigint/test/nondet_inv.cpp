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

#include "zirgen/Dialect/BigInt/IR/BigInt.h"
#include "zirgen/circuit/bigint/test/bibc.h"

#include <gtest/gtest.h>

using namespace zirgen;
using namespace zirgen::BigInt::test;

namespace {

// 定义一个函数，用于创建非确定性逆元测试
void makeNondetInvTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits) {
  // 创建输入、素数和预期值的定义操作
  auto inp = builder.create<BigInt::DefOp>(loc, bits, 0, true);
  auto prime = builder.create<BigInt::DefOp>(loc, bits, 1, true, bits - 1);
  auto expected = builder.create<BigInt::DefOp>(loc, bits, 2, true);

  // 构造常量
  mlir::Type oneType = builder.getIntegerType(1);    // 一个位宽为1的整数类型
  auto oneAttr = builder.getIntegerAttr(oneType, 1); // 值为1的整数属性
  auto one = builder.create<BigInt::ConstOp>(loc, oneAttr); // 创建值为1的常量操作

  // 创建非确定性逆元操作
  auto inv = builder.create<BigInt::NondetInvOp>(loc, inp, prime);
  // 创建乘法操作
  auto prod = builder.create<BigInt::MulOp>(loc, inp, inv);
  // 创建模约简操作
  auto reduced = builder.create<BigInt::ReduceOp>(loc, prod, prime);
  // 创建减法操作，期望结果为零
  auto expect_zero = builder.create<BigInt::SubOp>(loc, reduced, one);
  builder.create<BigInt::EqualZeroOp>(loc, expect_zero); // 创建等于零的检查操作
  // 创建结果匹配的减法操作
  auto result_match = builder.create<BigInt::SubOp>(loc, inv, expected);
  builder.create<BigInt::EqualZeroOp>(loc, result_match); // 创建等于零的检查操作
}

} // namespace

// 定义一个测试类，继承自BibcTest
TEST_F(BibcTest, NondetInv8) {
  mlir::OpBuilder builder(ctx); // 创建MLIR操作构建器
  auto func = makeFunc("nondet_inv_8", builder); // 创建一个名为"nondet_inv_8"的函数
  makeNondetInvTest(builder, func.getLoc(), 8); // 创建8位宽的非确定性逆元测试
  lower(); // 降低IR

  auto inputs = apints({"4", "3", "1"}); // 定义输入值
  ZType a, b; // 定义两个ZType变量
  AB(func, inputs, a, b); // 调用AB函数，传入函数和输入值，获取输出值a和b
  EXPECT_EQ(a, b); // 断言a和b是否相等
}

// 定义另一个测试类，继承自BibcTest
TEST_F(BibcTest, NondetInv128) {
  mlir::OpBuilder builder(ctx); // 创建MLIR操作构建器
  auto func = makeFunc("nondet_inv_128", builder); // 创建一个名为"nondet_inv_128"的函数
  makeNondetInvTest(builder, func.getLoc(), 128); // 创建128位宽的非确定性逆元测试
  lower(); // 降低IR

  auto inputs = apints({"100E", "0BB9", "03D9"}); // 定义输入值
  ZType a, b; // 定义两个ZType变量
  AB(func, inputs, a, b); // 调用AB函数，传入函数和输入值，获取输出值a和b
  EXPECT_EQ(a, b); // 断言a和b是否相等
}
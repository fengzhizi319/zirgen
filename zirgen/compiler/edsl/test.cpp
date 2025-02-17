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

#include "edsl.h"
#include <iostream>
using namespace zirgen;
using namespace std;

int main() {
  Module module;  // 创建一个模块对象
  module.addFunc<2>("test_func", {cbuf(2), mbuf(3)}, [](Buffer in, Buffer regs) {
    // clang-format off
    Val x = 1 / in[1];  // 计算 in[1] 的倒数并赋值给 x
    NONDET {  // 非确定性代码块
      regs[2] = 13;  // 将 13 赋值给 regs[2]
    }
    IF(x - 7) {  // 如果 x 不等于 7
      regs[1] = 7;  // 将 7 赋值给 regs[1]
    }
    Val a = 3;  // 定义变量 a 并赋值为 3
    Val c = -a;  // 定义变量 c 并赋值为 -a
    Val b = 4;  // 定义变量 b 并赋值为 4
    regs[0] = in[0] * in[0] + x + (a + b) * c;  // 计算表达式并赋值给 regs[0]
    // clang-format on
  });
  module.optimize();  // 优化模块


//  std::cout <<"hello!"<<"module.dump()" << std::endl;
   module.dump();  // 输出模块信息
}

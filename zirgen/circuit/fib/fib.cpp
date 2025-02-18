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

#include "zirgen/compiler/codegen/codegen.h"
#include "zirgen/compiler/codegen/protocol_info_const.h"
#include "zirgen/compiler/edsl/edsl.h"

using namespace zirgen;

int main(int argc, char* argv[]) {
  // 初始化 LLVM 环境
  llvm::InitLLVM y(argc, argv);

  // 注册 EDSL 命令行选项
  registerEdslCLOptions();

  // 注册代码生成命令行选项
  registerCodegenCLOptions();

  // 解析命令行选项
  llvm::cl::ParseCommandLineOptions(argc, argv, "fib edsl");

  // 创建模块对象
  Module module;
  /*
   cbuf：常量缓冲区（constant buffer），通常用于存储不变的数据。在这个例子中，cbuf(3, "code") 表示一个包含 3 个元素的常量缓冲区，名称为 "code"。
   gbuf：全局缓冲区（global buffer），用于存储全局可访问的数据。在这个例子中，gbuf(1, "out") 表示一个包含 1 个元素的全局缓冲区，名称为 "out"。
   buf：可变缓冲区（mutable buffer），用于存储可以在函数执行过程中修改的数据。在这个例子中，mbuf(1, "data") 表示一个包含 1 个元素的可变缓冲区，名称为 "data"。
 */
  // 添加名为 "fib" 的函数，具有 5 个缓冲区参数
  auto f = module.addFunc<5>( //
      "fib",
      {cbuf(3, "code"), gbuf(1, "out"), mbuf(1, "data"), gbuf(1, "mix"), mbuf(1, "accum")},
      [](Buffer control, Buffer out, Buffer data, Buffer mix, Buffer accum) {
        // 正常执行
        Register val = data[0];

        // 如果 control[0] 为真，则将 val 设为 1
        IF(control[0]) { val = 1; }

        // 如果 control[1] 为真，则将 val 设为前一个和前两个值之和
        IF(control[1]) { val = BACK(1, Val(val)) + BACK(2, Val(val)); }

        // 如果 control[2] 为真，则将 val 的值捕获到 out[0]
        IF(control[2]) {
          // TODO: 通过 BufAccess 修复寄存器相等性
          out[0] = CaptureVal(val);
        }

        // 设置屏障，确保在 control[2] 为假时同步
        barrier(1 - control[2]);
        barrier(1);
        barrier(1);
        barrier(1);

        // 如果 control[0]、control[1] 或 control[2] 为真，则将 accum[0] 设为 1
        IF(control[0] + control[1] + control[2]) { accum[0] = 1; }

        // 设置屏障，确保同步
        barrier(1);
      });

  // 设置函数的阶段
  module.setPhases(
      f,
      /*phases=*/{{"exec", "verify_mem", "verify_bytes", "compute_accum", "verify_accum"}});

  // 设置协议信息
  module.setProtocolInfo(FIBONACCI_CIRCUIT_INFO);

  // 生成代码
  emitCode(module.getModule());
}

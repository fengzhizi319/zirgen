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

use std::path::PathBuf;
fn main() {
    // 检查环境变量 "CARGO_CFG_TARGET_OS" 的值是否不等于 "zkvm"
    if std::env::var("CARGO_CFG_TARGET_OS").unwrap() != "zkvm" {
        // 使用 glob 模式查找 "cxx" 目录下的所有 ".cpp" 文件，并将其路径收集到 srcs 向量中
        let srcs: Vec<PathBuf> = glob::glob("cxx/*.cpp")
            .unwrap() // 处理 glob 结果中的错误
            .map(|x| x.unwrap()) // 处理每个路径结果中的错误
            .collect(); // 收集所有路径到向量中

        // 创建一个新的 cc::Build 对象，用于配置和编译 C++ 源文件
        cc::Build::new()
            .cpp(true) // 设置为编译 C++ 文件
            .files(&srcs) // 添加收集到的 C++ 源文件
            .flag_if_supported("/std:c++17") // 添加 C++17 标准的编译标志（Windows）
            .flag_if_supported("-std=c++17") // 添加 C++17 标准的编译标志（非 Windows）
            .compile("circuit"); // 编译并将生成的库命名为 "circuit"

        // 设置文件变更监控
        for src in srcs {
            // 打印 cargo 指令，以便在文件发生变更时重新运行构建脚本
            println!("cargo:rerun-if-changed={}", src.display());
        }
    }
}

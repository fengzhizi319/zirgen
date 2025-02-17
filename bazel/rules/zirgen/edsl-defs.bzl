DEFAULT_OUTS = [
    "eval_check.cu",
    "eval_check.cuh",
    "eval_check.metal",
    "eval_check.h",
    "impl.h",
    "info.rs",
    "poly_edsl.cpp",
    "poly_ext.rs",
    "rust_poly_fp.cpp",
    "rust_step_exec.cpp",
    "rust_step_verify_mem.cpp",
    "rust_step_verify_bytes.cpp",
    "rust_step_compute_accum.cpp",
    "rust_step_verify_accum.cpp",
    "step_exec.cu",
    "step_verify_mem.cu",
    "step_verify_bytes.cu",
    "step_compute_accum.cu",
    "step_verify_accum.cu",
    # "step_exec.metal",
    # "step_verify_mem.metal",
    # "step_verify_bytes.metal",
    "step_compute_accum.metal",
    "step_verify_accum.metal",
    "taps.cpp",
    "taps.rs",
    "layout.rs.inc",
    "layout.cpp.inc",
    "layout.cu.inc",
]

ZIRGEN_OUTS = [
    "defs.cpp.inc",
    "defs.cu.inc",
    "defs.rs.inc",
    "info.rs",
    "layout.cpp.inc",
    "layout.cu.inc",
    "layout.rs.inc",
    "poly_ext.rs",
    "taps.rs",
    "types.h.inc",
    "types.cuh.inc",
    "types.rs.inc",
    "validity.ir",
]

def _impl(ctx):
    outs = []  # 用于存储声明的输出文件
    out_dirs = dict()  # 用于存储输出文件的目录

    for out in ctx.attr.outs:
        # 声明输出文件
        declared = ctx.actions.declare_file(out.name)
        outs.append(declared)
        dirname = declared.dirname
        out_dirs[dirname] = dirname

    # 确保只有一个输出目录
    if len(out_dirs) != 1:
        fail("Must have exactly one output directory")

    # 运行生成代码的命令
    ctx.actions.run(
        mnemonic = "CodegenCircuits",  # 操作的助记符
        executable = ctx.executable.binary,  # 可执行的二进制文件
        arguments = [ctx.expand_location(arg, targets=ctx.attr.data) for arg in ctx.attr.extra_args] + ["--output-dir", dirname],  # 命令行参数
        inputs = ctx.files.data,  # 输入文件
        outputs = outs,  # 输出文件
        tools = [ctx.executable.binary],  # 工具
    )

    # 设置运行文件
    runfiles = ctx.runfiles(files=outs)
    return [DefaultInfo(files=depset(outs), runfiles=runfiles)]  # 返回默认信息

# 定义构建规则
_build_circuit_rule = rule(
    implementation=_impl,  # 实现函数
    attrs={
        "binary": attr.label(
            mandatory=True,  # 必须的属性
            executable=True,  # 可执行文件
            cfg="exec",  # 配置
            doc="The cpp program to run to generate results",  # 文档说明
        ),
        "data": attr.label_list(
            allow_files=True,  # 允许文件
        ),
        "outs": attr.output_list(mandatory=True),  # 输出文件列表
        "extra_args": attr.string_list(),  # 额外的命令行参数
    },
)

def build_circuit(name, srcs = [], bin = None, deps = [], outs = None, data = [], extra_args = []):
        # 如果没有指定 outs，则使用默认的输出文件列表
        if outs == None:
            outs = DEFAULT_OUTS

        # 如果没有指定 bin，则生成一个默认的二进制目标名称
        if not bin:
            bin = name + "_gen"
            # 定义一个 cc_binary 目标，用于编译 C++ 源文件
            native.cc_binary(
                name = bin,  # 二进制目标的名称
                srcs = srcs,  # 源文件列表
                deps = deps + [  # 依赖项列表
                    "@zirgen//zirgen/compiler/edsl",
                    "@zirgen//zirgen/compiler/codegen",
                ],
            )
        #打印bin
        print(bin)
        print(name)

        # 调用 _build_circuit_rule 规则来生成电路
        _build_circuit_rule(
            name = name,  # 规则的名称
            binary = bin,  # 可执行的二进制文件
            data = ["@zirgen//zirgen/compiler/codegen:data"] + data,  # 数据文件列表
            outs = outs,  # 输出文件列表
            extra_args = extra_args,  # 额外的命令行参数
        )

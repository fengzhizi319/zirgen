use risc0_zkp::layout::Reg;
use risc0_zirgen_dsl::codegen::_support::{define_buffer_list, define_global_buffer, define_tap_buffer, isz, set_field};
use risc0_zirgen_dsl::codegen::taps::Tap;

// 设置字段类型为 BabyBear
set_field!(BabyBear);

// 定义缓冲区列表
/*
这些缓冲区在电路生成和验证过程中用于存储和访问数据。具体来说：
行缓冲区：用于存储每一行的数据。
Tap 缓冲区：用于存储 Tap 操作的数据。
全局缓冲区：用于存储全局范围内的数据
 */
define_buffer_list! {
    all: [accum, code, data, global, mix,], // 所有缓冲区
    rows: [accum, code, data,], // 行缓冲区
    taps: [accum, code, data,], // Tap 缓冲区
    globals: [global, mix,], // 全局缓冲区
}

// 定义 Tap 缓冲区
define_tap_buffer! {accum, /*count=*/1, /*groupId=*/0}
define_tap_buffer! {code, /*count=*/1, /*groupId=*/1}
define_tap_buffer! {data, /*count=*/8, /*groupId=*/2}

// 定义全局缓冲区
define_global_buffer! {global, /*count=*/5}
define_global_buffer! {mix, /*count=*/4}

// 定义 NondetRegLayout 结构体
pub struct NondetRegLayout {
    pub _super: &'static Reg, // 父级寄存器
}

// 为 NondetRegLayout 实现 Component 接口
impl risc0_zkp::layout::Component for NondetRegLayout {
    fn ty_name(&self) -> &'static str {
        "NondetRegLayout" // 返回类型名称
    }

    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?; // 访问父级组件
        Ok(())
    }
}

// 定义 IsZeroLayout 结构体
pub struct IsZeroLayout {
    pub _super: &'static NondetRegLayout, // 父级 NondetRegLayout
    pub inv: &'static NondetRegLayout, // 逆元寄存器
}

// 为 IsZeroLayout 实现 Component 接口
impl risc0_zkp::layout::Component for IsZeroLayout {
    fn ty_name(&self) -> &'static str {
        "IsZeroLayout" // 返回类型名称
    }

    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?; // 访问父级组件
        v.visit_component("inv", self.inv)?; // 访问逆元组件
        Ok(())
    }
}

// 定义 CycleCounterLayout 结构体
pub struct CycleCounterLayout {
    pub _super: &'static NondetRegLayout, // 父级 NondetRegLayout
    pub is_first_cycle: &'static IsZeroLayout, // 是否为第一个周期
}

// 为 CycleCounterLayout 实现 Component 接口
impl risc0_zkp::layout::Component for CycleCounterLayout {
    fn ty_name(&self) -> &'static str {
        "CycleCounterLayout" // 返回类型名称
    }

    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?; // 访问父级组件
        v.visit_component("is_first_cycle", self.is_first_cycle)?; // 访问第一个周期组件
        Ok(())
    }
}

// 定义 Tap13Array 类型
/*
Tap13Array 的主要功能是定义一个包含 13 个 Tap 元素的数组类型。
在电路生成和验证过程中，这个数组用于存储和管理多个 Tap 数据。
Tap 结构体包含了组、位置和回溯等信息，通过 Tap13Array 可以方便
地访问和操作这些 Tap 数据。
Tap 数据在电路生成和验证过程中起着关键作用。具体来说，Tap 数据用于存储和管理电路中不同位置的状态信息。
每个 Tap 结构体包含了组、位置和回溯等信息，通过这些信息可以方便地访问和操作电路中的特定数据点。
这样可以确保在电路执行过程中，能够准确地跟踪和验证每个步骤的状态和结果。
 */
pub type Tap13Array = [Tap; 13];

// 定义 TopLayout 结构体
pub struct TopLayout {
    pub cycle: &'static CycleCounterLayout, // 周期计数器布局
    pub d2: &'static NondetRegLayout, // d2 寄存器布局
    pub d3: &'static NondetRegLayout, // d3 寄存器布局
    pub d1: &'static NondetRegLayout, // d1 寄存器布局
    pub terminate: &'static IsZeroLayout, // 终止条件布局
}

// 为 TopLayout 实现 Component 接口
impl risc0_zkp::layout::Component for TopLayout {
    fn ty_name(&self) -> &'static str {
        "TopLayout" // 返回类型名称
    }

    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("cycle", self.cycle)?; // 访问周期计数器组件
        v.visit_component("d2", self.d2)?; // 访问 d2 组件
        v.visit_component("d3", self.d3)?; // 访问 d3 组件
        v.visit_component("d1", self.d1)?; // 访问 d1 组件
        v.visit_component("terminate", self.terminate)?; // 访问终止条件组件
        Ok(())
    }
}

// 定义 _globalLayout 结构体
pub struct _globalLayout {
    pub f0: &'static NondetRegLayout, // f0 寄存器布局
    pub f1: &'static NondetRegLayout, // f1 寄存器布局
    pub f_last: &'static NondetRegLayout, // f_last 寄存器布局
    pub steps: &'static NondetRegLayout, // 步骤寄存器布局
    pub total_cycles: &'static NondetRegLayout, // 总周期寄存器布局
}

// 为 _globalLayout 实现 Component 接口
impl risc0_zkp::layout::Component for _globalLayout {
    fn ty_name(&self) -> &'static str {
        "_globalLayout" // 返回类型名称
    }

    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("f0", self.f0)?; // 访问 f0 组件
        v.visit_component("f1", self.f1)?; // 访问 f1 组件
        v.visit_component("f_last", self.f_last)?; // 访问 f_last 组件
        v.visit_component("steps", self.steps)?; // 访问步骤组件
        v.visit_component("total_cycles", self.total_cycles)?; // 访问总周期组件
        Ok(())
    }
}

// 定义 NondetRegStruct 结构体
#[derive(Copy, Clone, Debug)]
pub struct NondetRegStruct<'a> {
    pub _super: Val, // 父级值
    pub _layout: BoundLayout<'a, NondetRegLayout, Val>, // 布局绑定
}

// 定义 ComponentStruct 结构体
#[derive(Copy, Clone, Debug)]
pub struct ComponentStruct {}

// 定义 RegStruct 结构体
#[derive(Copy, Clone, Debug)]
pub struct RegStruct<'a> {
    pub _super: NondetRegStruct<'a>, // 父级 NondetRegStruct
    pub reg: NondetRegStruct<'a>, // 寄存器结构
    pub _layout: BoundLayout<'a, NondetRegLayout, Val>, // 布局绑定
}

// 定义 GetCycleStruct 结构体
#[derive(Copy, Clone, Debug)]
pub struct GetCycleStruct {
    pub _super: Val, // 父级值
}

// 定义 IsZeroStruct 结构体
#[derive(Copy, Clone, Debug)]
pub struct IsZeroStruct<'a> {
    pub _super: NondetRegStruct<'a>, // 父级 NondetRegStruct
    pub is_zero: NondetRegStruct<'a>, // 零检测结构
    pub inv: NondetRegStruct<'a>, // 逆元结构
    pub _layout: BoundLayout<'a, IsZeroLayout, Val>, // 布局绑定
}

// 定义 CycleCounterStruct 结构体
#[derive(Copy, Clone, Debug)]
pub struct CycleCounterStruct<'a> {
    pub _super: NondetRegStruct<'a>, // 父级 NondetRegStruct
    pub cycle: NondetRegStruct<'a>, // 周期结构
    pub is_first_cycle: IsZeroStruct<'a>, // 第一个周期结构
    pub _layout: BoundLayout<'a, CycleCounterLayout, Val>, // 布局绑定
}

// 定义 TopStruct 结构体
#[derive(Copy, Clone, Debug)]
pub struct TopStruct<'a> {
    pub _super: ComponentStruct, // 父级组件结构
    pub cycle: CycleCounterStruct<'a>, // 周期计数器结构
    pub first: IsZeroStruct<'a>, // 第一个周期结构
    pub d1: RegStruct<'a>, // d1 寄存器结构
    pub d2: RegStruct<'a>, // d2 寄存器结构
    pub d3: RegStruct<'a>, // d3 寄存器结构
    pub terminate: IsZeroStruct<'a>, // 终止条件结构
    pub _layout: BoundLayout<'a, TopLayout, Val>, // 布局绑定
}

// 定义常量 LAYOUT__0
pub const LAYOUT__0: &CycleCounterLayout = &CycleCounterLayout {
    _super: &NondetRegLayout {
        _super: &Reg { offset: 0 },
    },
    is_first_cycle: &IsZeroLayout {
        _super: &NondetRegLayout {
            _super: &Reg { offset: 1 },
        },
        inv: &NondetRegLayout {
            _super: &Reg { offset: 2 },
        },
    },
};

// 定义 Tap 列表常量
pub const TAP_LIST: Tap13Array = [
    make_tap!(0, 0, 0),
    make_tap!(1, 0, 0),
    make_tap!(2, 0, 0),
    make_tap!(2, 0, 1),
    make_tap!(2, 1, 0),
    make_tap!(2, 2, 0),
    make_tap!(2, 3, 0),
    make_tap!(2, 3, 1),
    make_tap!(2, 4, 0),
    make_tap!(2, 4, 1),
    make_tap!(2, 5, 0),
    make_tap!(2, 6, 0),
    make_tap!(2, 7, 0),
];

// 定义常量 LAYOUT_TOP
pub const LAYOUT_TOP: &TopLayout = &TopLayout {
    cycle: LAYOUT__0,
    d2: &NondetRegLayout {
        _super: &Reg { offset: 3 },
    },
    d3: &NondetRegLayout {
        _super: &Reg { offset: 4 },
    },
    d1: &NondetRegLayout {
        _super: &Reg { offset: 5 },
    },
    terminate: &IsZeroLayout {
        _super: &NondetRegLayout {
            _super: &Reg { offset: 6 },
        },
        inv: &NondetRegLayout {
            _super: &Reg { offset: 7 },
        },
    },
};

// 定义常量 LAYOUT_GLOBAL
pub const LAYOUT_GLOBAL: &_globalLayout = &_globalLayout {
    f0: &NondetRegLayout {
        _super: &Reg { offset: 0 },
    },
    f1: &NondetRegLayout {
        _super: &Reg { offset: 1 },
    },
    f_last: &NondetRegLayout {
        _super: &Reg { offset: 2 },
    },
    steps: &NondetRegLayout {
        _super: &Reg { offset: 3 },
    },
    total_cycles: &NondetRegLayout {
        _super: &Reg { offset: 4 },
    },
};

// 执行逆元操作
pub fn exec_inv<'a>(ctx: &'a ExecContext, arg0: Val) -> Result<Val> {
    return Ok(inv_0(arg0)?);
}

// 执行零检测操作
pub fn exec_isz<'a>(ctx: &'a ExecContext, arg0: Val) -> Result<Val> {
    return Ok(isz(arg0)?);
}

// 执行加法操作
pub fn exec_add<'a>(ctx: &'a ExecContext, arg0: Val, arg1: Val) -> Result<Val> {
    return Ok((arg0 + arg1));
}

// 执行减法操作
pub fn exec_sub<'a>(ctx: &'a ExecContext, arg0: Val, arg1: Val) -> Result<Val> {
    return Ok((arg0 - arg1));
}

// 执行乘法操作
pub fn exec_mul<'a>(ctx: &'a ExecContext, arg0: Val, arg1: Val) -> Result<Val> {
    return Ok((arg0 * arg1));
}

// 回溯 NondetReg 寄存器
pub fn back_nondet_reg<'a>(
    ctx: &'a ExecContext,
    distance0: Index,
    layout1: BoundLayout<'a, NondetRegLayout, Val>,
) -> Result<NondetRegStruct<'a>> {
    let x2: NondetRegStruct = NondetRegStruct {
        _super: (layout1.map(|c| c._super)).load(ctx, distance0),
        _layout: layout1,
    };
    return Ok(x2);
}

// 执行 NondetReg 寄存器操作
pub fn exec_nondet_reg<'a>(
    ctx: &'a ExecContext,
    arg0: Val,
    layout1: BoundLayout<'a, NondetRegLayout, Val>,
) -> Result<NondetRegStruct<'a>> {
    (layout1.map(|c| c._super)).store(ctx, arg0);
    let x2: NondetRegStruct = NondetRegStruct {
        _super: (layout1.map(|c| c._super)).load(ctx, 0),
        _layout: layout1,
    };
    return Ok(x2);
}

// 执行组件操作
pub fn exec_component<'a>(ctx: &'a ExecContext) -> Result<ComponentStruct> {
    return Ok(ComponentStruct {});
}

// 回溯寄存器
pub fn back_reg<'a>(
    ctx: &'a ExecContext,
    distance0: Index,
    layout1: BoundLayout<'a, NondetRegLayout, Val>,
) -> Result<RegStruct<'a>> {
    // 回溯 NondetReg 寄存器
    let x2: NondetRegStruct = back_nondet_reg(ctx, distance0, layout1)?;
    return Ok(RegStruct {
        _super: x2.clone(),
        reg: x2,
        _layout: layout1,
    });
}

// 执行寄存器操作
pub fn exec_reg<'a>(
    ctx: &'a ExecContext,
    arg0: Val,
    layout1: BoundLayout<'a, NondetRegLayout, Val>,
) -> Result<RegStruct<'a>> {
    let x2: NondetRegStruct = exec_nondet_reg(ctx, arg0, layout1)?;
    // 检查寄存器值
    eqz!((arg0 - x2._super), "Reg(<preamble>:5)");
    return Ok(RegStruct {
        _super: x2.clone(),
        reg: x2,
        _layout: layout1,
    });
}

// 执行日志操作
pub fn exec_log<'a>(ctx: &'a ExecContext, arg0: &str, arg1: &[Val]) -> Result<ComponentStruct> {
    // 调用外部日志函数
    invoke_extern!(ctx, log, arg0, arg1);
    return Ok(ComponentStruct {});
}

// 获取周期
pub fn exec_get_cycle<'a>(ctx: &'a ExecContext) -> Result<GetCycleStruct> {
    // 调用外部获取周期函数
    let x0: Val = invoke_extern!(ctx, get_cycle);
    return Ok(GetCycleStruct { _super: x0 });
}

// 执行零检测操作
pub fn exec_is_zero<'a>(
    ctx: &'a ExecContext,
    arg0: Val,
    layout1: BoundLayout<'a, IsZeroLayout, Val>,
) -> Result<IsZeroStruct<'a>> {
    // 执行零检测
    let x2: Val = exec_isz(ctx, arg0)?;
    let x3: NondetRegStruct = exec_nondet_reg(ctx, x2, (layout1.map(|c| c._super)))?;
    // 执行逆元操作
    let x4: Val = exec_inv(ctx, arg0)?;
    let x5: NondetRegStruct = exec_nondet_reg(ctx, x4, (layout1.map(|c| c.inv)))?;
    // 检查零检测结果
    let x6: Val = exec_sub(ctx, Val::new(1), x3._super)?;
    let x7: Val = exec_mul(ctx, x3._super, x6)?;
    eqz!(x7, "IsZero(zirgen/dsl/examples/fibonacci.zir:12)");
    let x8: Val = exec_mul(ctx, arg0, x5._super)?;
    let x9: Val = exec_sub(ctx, Val::new(1), x3._super)?;
    eqz!((x8 - x9), "IsZero(zirgen/dsl/examples/fibonacci.zir:14)");
    let x10: Val = exec_mul(ctx, x3._super, arg0)?;
    eqz!(x10, "IsZero(zirgen/dsl/examples/fibonacci.zir:16)");
    let x11: Val = exec_mul(ctx, x3._super, x5._super)?;
    eqz!(x11, "IsZero(zirgen/dsl/examples/fibonacci.zir:18)");
    return Ok(IsZeroStruct {
        _super: x3,
        is_zero: x3,
        inv: x5,
        _layout: layout1,
    });
}
pub fn exec_cycle_counter<'a>(
    ctx: &'a ExecContext,
    layout0: BoundLayout<'a, CycleCounterLayout, Val>,
    global1: BufferRow<Val>,
) -> Result<CycleCounterStruct<'a>> {
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:28)
    let x2: BoundLayout<_globalLayout, _> = bind_layout!(LAYOUT_GLOBAL, global1);
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:29)
    let x3: NondetRegStruct = exec_nondet_reg(ctx, Val::new(6), (x2.map(|c| c.total_cycles)))?;
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:31)
    let x4: GetCycleStruct = exec_get_cycle(ctx)?;
    let x5: NondetRegStruct = exec_nondet_reg(ctx, x4._super, (layout0.map(|c| c._super)))?;
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:32)
    let x6: IsZeroStruct = exec_is_zero(ctx, x5._super, (layout0.map(|c| c.is_first_cycle)))?;
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:34)
    let x7: Val = exec_sub(ctx, Val::new(1), x6._super._super)?;
    let x8: ComponentStruct;
    if is_true(x6._super._super) {
        let x9: ComponentStruct = exec_component(ctx)?;
        x8 = x9;
    } else if is_true(x7) {
        // CycleCounter(zirgen/dsl/examples/fibonacci.zir:39)
        let x10: NondetRegStruct = back_nondet_reg(ctx, 1, (layout0.map(|c| c._super)))?;
        let x11: Val = exec_add(ctx, x10._super, Val::new(1))?;
        eqz!(
            (x5._super - x11),
            "CycleCounter(zirgen/dsl/examples/fibonacci.zir:39)"
        );
        // CycleCounter(zirgen/dsl/examples/fibonacci.zir:37)
        let x12: ComponentStruct = exec_component(ctx)?;
        x8 = x12;
    } else {
        bail!("Reached unreachable mux arm")
    }
    return Ok(CycleCounterStruct {
        _super: x5,
        cycle: x5,
        is_first_cycle: x6,
        _layout: layout0,
    });
}
pub fn exec_top<'a>(
    ctx: &'a ExecContext,
    layout0: BoundLayout<'a, TopLayout, Val>,
    global1: BufferRow<Val>,
) -> Result<TopStruct<'a>> {
    // Top(zirgen/dsl/examples/fibonacci.zir:44)
    let x2: BoundLayout<_globalLayout, _> = bind_layout!(LAYOUT_GLOBAL, global1);
    // Top(zirgen/dsl/examples/fibonacci.zir:49)
    let x3: CycleCounterStruct = exec_cycle_counter(ctx, (layout0.map(|c| c.cycle)), global1)?;
    // Top(zirgen/dsl/examples/fibonacci.zir:55)
    let x4: Val = exec_sub(ctx, Val::new(1), x3.is_first_cycle._super._super)?;
    let x5: RegStruct;
    if is_true(x3.is_first_cycle._super._super) {
        // Top(zirgen/dsl/examples/fibonacci.zir:45)
        let x6: RegStruct = back_reg(ctx, 0, (x2.map(|c| c.f0)))?;
        x5 = x6;
    } else if is_true(x4) {
        // Top(zirgen/dsl/examples/fibonacci.zir:55)
        let x7: RegStruct = back_reg(ctx, 1, (layout0.map(|c| c.d2)))?;
        x5 = x7;
    } else {
        bail!("Reached unreachable mux arm")
    }
    let x8: RegStruct = exec_reg(ctx, x5._super._super, (layout0.map(|c| c.d1)))?;
    // Top(zirgen/dsl/examples/fibonacci.zir:56)
    let x9: Val = exec_sub(ctx, Val::new(1), x3.is_first_cycle._super._super)?;
    let x10: RegStruct;
    if is_true(x3.is_first_cycle._super._super) {
        // Top(zirgen/dsl/examples/fibonacci.zir:46)
        let x11: RegStruct = back_reg(ctx, 0, (x2.map(|c| c.f1)))?;
        x10 = x11;
    } else if is_true(x9) {
        // Top(zirgen/dsl/examples/fibonacci.zir:56)
        let x12: RegStruct = back_reg(ctx, 1, (layout0.map(|c| c.d3)))?;
        x10 = x12;
    } else {
        bail!("Reached unreachable mux arm")
    }
    let x13: RegStruct = exec_reg(ctx, x10._super._super, (layout0.map(|c| c.d2)))?;
    // Top(zirgen/dsl/examples/fibonacci.zir:59)
    let x14: Val = exec_add(ctx, x8._super._super, x13._super._super)?;
    let x15: RegStruct = exec_reg(ctx, x14, (layout0.map(|c| c.d3)))?;
    // Top(zirgen/dsl/examples/fibonacci.zir:47)
    let x16: RegStruct = back_reg(ctx, 0, (x2.map(|c| c.steps)))?;
    // Top(zirgen/dsl/examples/fibonacci.zir:62)
    let x17: Val = exec_sub(ctx, x3._super._super, x16._super._super)?;
    let x18: Val = exec_add(ctx, x17, Val::new(1))?;
    let x19: IsZeroStruct = exec_is_zero(ctx, x18, (layout0.map(|c| c.terminate)))?;
    // Top(zirgen/dsl/examples/fibonacci.zir:63)
    let x20: Val = exec_sub(ctx, Val::new(1), x19._super._super)?;
    let x21: ComponentStruct;
    if is_true(x19._super._super) {
        // Top(zirgen/dsl/examples/fibonacci.zir:64)
        let x22: RegStruct = exec_reg(ctx, x15._super._super, (x2.map(|c| c.f_last)))?;
        let x23: RegStruct = back_reg(ctx, 0, (x2.map(|c| c.f_last)))?;
        // Top(zirgen/dsl/examples/fibonacci.zir:65)
        let x24: [Val] = [x23._super._super];
        let x25: ComponentStruct = exec_log(ctx, "f_last = %u", &x24)?;
        // Top(zirgen/dsl/examples/fibonacci.zir:63)
        let x26: ComponentStruct = exec_component(ctx)?;
        x21 = x26;
    } else if is_true(x20) {
        // Top(zirgen/dsl/examples/fibonacci.zir:66)
        let x27: ComponentStruct = exec_component(ctx)?;
        x21 = x27;
    } else {
        bail!("Reached unreachable mux arm")
    } // Top(zirgen/dsl/examples/fibonacci.zir:44)
    let x28: ComponentStruct = exec_component(ctx)?;
    return Ok(TopStruct {
        _super: x28,
        cycle: x3,
        first: x3.is_first_cycle,
        d1: x8,
        d2: x13,
        d3: x15,
        terminate: x19,
        _layout: layout0,
    });
}
pub fn step_top<'a>(
    ctx: &'a ExecContext,
    data0: BufferRow<Val>,
    global1: BufferRow<Val>,
) -> Result<()> {
    let x2: BoundLayout<TopLayout, _> = bind_layout!(LAYOUT_TOP, data0);
    let x3: TopStruct = exec_top(ctx, x2, global1)?;
    return Ok(());
}
pub fn validity_taps<'a>(
    ctx: &'a ValidityTapsContext,
    taps0: BufferRow<ExtVal>,
    poly_mix1: PolyMix,
    global2: BufferRow<Val>,
) -> Result<MixState> {
    let x3: ExtVal = get(ctx, taps0, 2, 0)?;
    let x4: ExtVal = get(ctx, taps0, 4, 0)?;
    let x5: ExtVal = get(ctx, taps0, 5, 0)?;
    let x6: ExtVal = get(ctx, taps0, 6, 0)?;
    let x7: ExtVal = get(ctx, taps0, 8, 0)?;
    let x8: ExtVal = get(ctx, taps0, 10, 0)?;
    let x9: ExtVal = get(ctx, taps0, 11, 0)?;
    let x10: ExtVal = get(ctx, taps0, 12, 0)?;
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:34)
    // Top(zirgen/dsl/examples/fibonacci.zir:49)
    let x11: MixState = trivial_constraint()?;
    let x12: BoundLayout<_globalLayout, _> = bind_layout!(LAYOUT_GLOBAL, global2);
    // Top(zirgen/dsl/examples/fibonacci.zir:62)
    let x13: ExtVal = ((x3 - ((x12.map(|c| c.steps)).map(|c| c._super)).load(ctx, 0))
        + ExtVal::new(Val::new(1), Val::new(0), Val::new(0), Val::new(0)));
    // IsZero(zirgen/dsl/examples/fibonacci.zir:12)
    let x14: ExtVal = (ExtVal::new(Val::new(1), Val::new(0), Val::new(0), Val::new(0)) - x9);
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:32)
    // Top(zirgen/dsl/examples/fibonacci.zir:49)
    let x15: ExtVal = (ExtVal::new(Val::new(1), Val::new(0), Val::new(0), Val::new(0)) - x4);
    // IsZero(zirgen/dsl/examples/fibonacci.zir:14)
    let x16: MixState = and_eqz_ext(ctx, and_eqz_ext(ctx, x11, (x4 * x15))?, ((x3 * x5) - x15))?;
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:34)
    let x17: MixState = and_cond_ext(
        and_eqz_ext(ctx, and_eqz_ext(ctx, x16, (x4 * x3))?, (x4 * x5))?,
        x15,
        and_eqz_ext(
            ctx,
            x11,
            (x3 - (get(ctx, taps0, 3, 0)?
                + ExtVal::new(Val::new(1), Val::new(0), Val::new(0), Val::new(0)))),
        )?,
    )?;
    // Top(zirgen/dsl/examples/fibonacci.zir:55)
    let x18: ExtVal = (((x12.map(|c| c.f0)).map(|c| c._super)).load_unchecked(ctx, 0) * x4);
    // Top(zirgen/dsl/examples/fibonacci.zir:56)
    let x19: ExtVal = (((x12.map(|c| c.f1)).map(|c| c._super)).load_unchecked(ctx, 0) * x4);
    // Reg(<preamble>:5)
    let x20: MixState = and_eqz_ext(
        ctx,
        and_eqz_ext(ctx, x17, ((x18 + (get(ctx, taps0, 7, 0)? * x15)) - x8))?,
        ((x19 + (get(ctx, taps0, 9, 0)? * x15)) - x6),
    )?;
    // IsZero(zirgen/dsl/examples/fibonacci.zir:14)
    // Top(zirgen/dsl/examples/fibonacci.zir:62)
    let x21: MixState = and_eqz_ext(
        ctx,
        and_eqz_ext(ctx, and_eqz_ext(ctx, x20, ((x8 + x6) - x7))?, (x9 * x14))?,
        ((x13 * x10) - x14),
    )?;
    // Top(zirgen/dsl/examples/fibonacci.zir:63)
    let x22: MixState = and_cond_ext(
        and_eqz_ext(ctx, and_eqz_ext(ctx, x21, (x9 * x13))?, (x9 * x10))?,
        x9,
        and_eqz_ext(
            ctx,
            x11,
            (x7 - ((x12.map(|c| c.f_last)).map(|c| c._super)).load(ctx, 0)),
        )?,
    )?;
    return Ok(x22);
}
pub fn validity_regs<'a>(
    ctx: &'a ValidityRegsContext,
    poly_mix0: PolyMix,
    data1: BufferRow<Val>,
    global2: BufferRow<Val>,
) -> Result<MixState> {
    let x3: BoundLayout<TopLayout, _> = bind_layout!(LAYOUT_TOP, data1);
    let x4: BoundLayout<_globalLayout, _> = bind_layout!(LAYOUT_GLOBAL, global2);
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:39)
    // Top(zirgen/dsl/examples/fibonacci.zir:49)
    let x5: Val =
        ((((x3.map(|c| c.cycle)).map(|c| c._super)).map(|c| c._super)).load(ctx, 1) + Val::new(1));
    let x6: Val = ((((x3.map(|c| c.cycle)).map(|c| c._super)).map(|c| c._super)).load(ctx, 0) - x5);
    // Top(zirgen/dsl/examples/fibonacci.zir:59)
    let x7: Val = (((x3.map(|c| c.d1)).map(|c| c._super)).load(ctx, 0)
        + ((x3.map(|c| c.d2)).map(|c| c._super)).load(ctx, 0));
    // Top(zirgen/dsl/examples/fibonacci.zir:62)
    let x8: Val = ((((x3.map(|c| c.cycle)).map(|c| c._super)).map(|c| c._super)).load(ctx, 0)
        - ((x4.map(|c| c.steps)).map(|c| c._super)).load(ctx, 0));
    let x9: Val = (x8 + Val::new(1));
    // IsZero(zirgen/dsl/examples/fibonacci.zir:12)
    let x10: Val = (Val::new(1)
        - (((x3.map(|c| c.terminate)).map(|c| c._super)).map(|c| c._super)).load(ctx, 0));
    let x11: Val =
        ((((x3.map(|c| c.terminate)).map(|c| c._super)).map(|c| c._super)).load(ctx, 0) * x10);
    // IsZero(zirgen/dsl/examples/fibonacci.zir:14)
    let x12: Val = (Val::new(1)
        - (((x3.map(|c| c.terminate)).map(|c| c._super)).map(|c| c._super)).load(ctx, 0));
    // IsZero(zirgen/dsl/examples/fibonacci.zir:16)
    let x13: Val =
        ((((x3.map(|c| c.terminate)).map(|c| c._super)).map(|c| c._super)).load(ctx, 0) * x9);
    // IsZero(zirgen/dsl/examples/fibonacci.zir:18)
    let x14: Val = ((((x3.map(|c| c.terminate)).map(|c| c._super)).map(|c| c._super)).load(ctx, 0)
        * (((x3.map(|c| c.terminate)).map(|c| c.inv)).map(|c| c._super)).load(ctx, 0));
    // Reg(<preamble>:5)
    // Top(zirgen/dsl/examples/fibonacci.zir:64)
    let x15: Val = (((x3.map(|c| c.d3)).map(|c| c._super)).load(ctx, 0)
        - ((x4.map(|c| c.f_last)).map(|c| c._super)).load(ctx, 0));
    // IsZero(zirgen/dsl/examples/fibonacci.zir:12)
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:32)
    // Top(zirgen/dsl/examples/fibonacci.zir:49)
    let x16: Val = (Val::new(1)
        - ((((x3.map(|c| c.cycle)).map(|c| c.is_first_cycle)).map(|c| c._super))
            .map(|c| c._super))
        .load(ctx, 0));
    let x17: Val = (((((x3.map(|c| c.cycle)).map(|c| c.is_first_cycle)).map(|c| c._super))
        .map(|c| c._super))
    .load(ctx, 0)
        * x16);
    // IsZero(zirgen/dsl/examples/fibonacci.zir:14)
    let x18: Val = ((((x3.map(|c| c.cycle)).map(|c| c._super)).map(|c| c._super)).load(ctx, 0)
        * ((((x3.map(|c| c.cycle)).map(|c| c.is_first_cycle)).map(|c| c.inv)).map(|c| c._super))
            .load(ctx, 0));
    let x19: Val = (Val::new(1)
        - ((((x3.map(|c| c.cycle)).map(|c| c.is_first_cycle)).map(|c| c._super))
            .map(|c| c._super))
        .load(ctx, 0));
    let x20: MixState = and_eqz(ctx, and_eqz(ctx, trivial_constraint()?, x17)?, (x18 - x19))?;
    // IsZero(zirgen/dsl/examples/fibonacci.zir:16)
    let x21: Val = (((((x3.map(|c| c.cycle)).map(|c| c.is_first_cycle)).map(|c| c._super))
        .map(|c| c._super))
    .load(ctx, 0)
        * (((x3.map(|c| c.cycle)).map(|c| c._super)).map(|c| c._super)).load(ctx, 0));
    // IsZero(zirgen/dsl/examples/fibonacci.zir:18)
    let x22: Val = (((((x3.map(|c| c.cycle)).map(|c| c.is_first_cycle)).map(|c| c._super))
        .map(|c| c._super))
    .load(ctx, 0)
        * ((((x3.map(|c| c.cycle)).map(|c| c.is_first_cycle)).map(|c| c.inv)).map(|c| c._super))
            .load(ctx, 0));
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:34)
    let x23: Val = (Val::new(1)
        - ((((x3.map(|c| c.cycle)).map(|c| c.is_first_cycle)).map(|c| c._super))
            .map(|c| c._super))
        .load(ctx, 0));
    let x24: MixState = and_cond(
        and_eqz(ctx, and_eqz(ctx, x20, x21)?, x22)?,
        x23,
        and_eqz(ctx, trivial_constraint()?, x6)?,
    )?;
    // Top(zirgen/dsl/examples/fibonacci.zir:55)
    let x25: Val = (Val::new(1)
        - ((((x3.map(|c| c.cycle)).map(|c| c.is_first_cycle)).map(|c| c._super))
            .map(|c| c._super))
        .load(ctx, 0));
    let x26: Val = (((x4.map(|c| c.f0)).map(|c| c._super)).load_unchecked(ctx, 0)
        * ((((x3.map(|c| c.cycle)).map(|c| c.is_first_cycle)).map(|c| c._super))
            .map(|c| c._super))
        .load(ctx, 0));
    let x27: Val = (((x3.map(|c| c.d2)).map(|c| c._super)).load_unchecked(ctx, 1) * x25);
    // Reg(<preamble>:5)
    let x28: Val = ((x26 + x27) - ((x3.map(|c| c.d1)).map(|c| c._super)).load(ctx, 0));
    // Top(zirgen/dsl/examples/fibonacci.zir:56)
    let x29: Val = (Val::new(1)
        - ((((x3.map(|c| c.cycle)).map(|c| c.is_first_cycle)).map(|c| c._super))
            .map(|c| c._super))
        .load(ctx, 0));
    let x30: Val = (((x4.map(|c| c.f1)).map(|c| c._super)).load_unchecked(ctx, 0)
        * ((((x3.map(|c| c.cycle)).map(|c| c.is_first_cycle)).map(|c| c._super))
            .map(|c| c._super))
        .load(ctx, 0));
    let x31: Val = (((x3.map(|c| c.d3)).map(|c| c._super)).load_unchecked(ctx, 1) * x29);
    // Reg(<preamble>:5)
    let x32: Val = ((x30 + x31) - ((x3.map(|c| c.d2)).map(|c| c._super)).load(ctx, 0));
    // Top(zirgen/dsl/examples/fibonacci.zir:59)
    let x33: MixState = and_eqz(
        ctx,
        and_eqz(ctx, and_eqz(ctx, x24, x28)?, x32)?,
        (x7 - ((x3.map(|c| c.d3)).map(|c| c._super)).load(ctx, 0)),
    )?;
    // IsZero(zirgen/dsl/examples/fibonacci.zir:14)
    // Top(zirgen/dsl/examples/fibonacci.zir:62)
    let x34: MixState = and_eqz(
        ctx,
        and_eqz(ctx, x33, x11)?,
        ((x9 * (((x3.map(|c| c.terminate)).map(|c| c.inv)).map(|c| c._super)).load(ctx, 0)) - x12),
    )?;
    // Top(zirgen/dsl/examples/fibonacci.zir:63)
    let x35: MixState = and_cond(
        and_eqz(ctx, and_eqz(ctx, x34, x13)?, x14)?,
        (((x3.map(|c| c.terminate)).map(|c| c._super)).map(|c| c._super)).load(ctx, 0),
        and_eqz(ctx, trivial_constraint()?, x15)?,
    )?;
    return Ok(x35);
}

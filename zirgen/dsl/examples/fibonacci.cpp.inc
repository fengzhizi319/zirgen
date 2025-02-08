SET_FIELD(BabyBear);
constexpr size_t kRegCountAccum = 1;
constexpr size_t kRegCountCode = 1;
constexpr size_t kRegCountData = 8;
constexpr size_t kRegCountGlobal = 5;
constexpr size_t kRegCountMix = 4;
struct NondetRegLayout {
  Reg _super;
};
struct IsZeroLayout {
  NondetRegLayout _super;
  NondetRegLayout inv;
};
struct CycleCounterLayout {
  NondetRegLayout _super;
  IsZeroLayout isFirstCycle;
};
using Tap13Array = std::array<Tap, 13>;
struct TopLayout {
  CycleCounterLayout cycle;
  NondetRegLayout d2;
  NondetRegLayout d3;
  NondetRegLayout d1;
  IsZeroLayout terminate;
};
struct _globalLayout {
  NondetRegLayout f0;
  NondetRegLayout f1;
  NondetRegLayout fLast;
  NondetRegLayout steps;
  NondetRegLayout totalCycles;
};
struct NondetRegStruct {
  Val _super;
  BoundLayout<NondetRegLayout> _layout;
};
struct ComponentStruct {
};
struct RegStruct {
  NondetRegStruct _super;
  NondetRegStruct reg;
  BoundLayout<NondetRegLayout> _layout;
};
struct GetCycleStruct {
  Val _super;
};
struct IsZeroStruct {
  NondetRegStruct _super;
  NondetRegStruct isZero;
  NondetRegStruct inv;
  BoundLayout<IsZeroLayout> _layout;
};
struct CycleCounterStruct {
  NondetRegStruct _super;
  NondetRegStruct cycle;
  IsZeroStruct isFirstCycle;
  BoundLayout<CycleCounterLayout> _layout;
};
struct TopStruct {
  ComponentStruct _super;
  CycleCounterStruct cycle;
  IsZeroStruct first;
  RegStruct d1;
  RegStruct d2;
  RegStruct d3;
  IsZeroStruct terminate;
  BoundLayout<TopLayout> _layout;
};
constexpr CycleCounterLayout kLayout__0 = CycleCounterLayout{
  ._super = NondetRegLayout{
  ._super = /*offset=*/0},   .isFirstCycle = IsZeroLayout{
  ._super = NondetRegLayout{
  ._super = /*offset=*/1},   .inv = NondetRegLayout{
  ._super = /*offset=*/2}}};
constexpr Tap13Array kTapList = Tap13Array{MAKE_TAP(0, 0, 0), MAKE_TAP(1, 0, 0), MAKE_TAP(2, 0, 0), MAKE_TAP(2, 0, 1), MAKE_TAP(2, 1, 0), MAKE_TAP(2, 2, 0), MAKE_TAP(2, 3, 0), MAKE_TAP(2, 3, 1), MAKE_TAP(2, 4, 0), MAKE_TAP(2, 4, 1), MAKE_TAP(2, 5, 0), MAKE_TAP(2, 6, 0), MAKE_TAP(2, 7, 0)};
constexpr TopLayout kLayout_Top = TopLayout{
  .cycle = kLayout__0,   .d2 = NondetRegLayout{
  ._super = /*offset=*/3},   .d3 = NondetRegLayout{
  ._super = /*offset=*/4},   .d1 = NondetRegLayout{
  ._super = /*offset=*/5},   .terminate = IsZeroLayout{
  ._super = NondetRegLayout{
  ._super = /*offset=*/6},   .inv = NondetRegLayout{
  ._super = /*offset=*/7}}};
constexpr _globalLayout kLayoutGlobal = _globalLayout{
  .f0 = NondetRegLayout{
  ._super = /*offset=*/0},   .f1 = NondetRegLayout{
  ._super = /*offset=*/1},   .fLast = NondetRegLayout{
  ._super = /*offset=*/2},   .steps = NondetRegLayout{
  ._super = /*offset=*/3},   .totalCycles = NondetRegLayout{
  ._super = /*offset=*/4}};
Val exec_Inv(ExecContext& ctx,Val arg0)   {
return inv_0(arg0);
}
Val exec_Isz(ExecContext& ctx,Val arg0)   {
return isz(arg0);
}
Val exec_Add(ExecContext& ctx,Val arg0, Val arg1)   {
return (arg0 + arg1);
}
Val exec_Sub(ExecContext& ctx,Val arg0, Val arg1)   {
return (arg0 - arg1);
}
Val exec_Mul(ExecContext& ctx,Val arg0, Val arg1)   {
return (arg0 * arg1);
}
NondetRegStruct back_NondetReg(ExecContext& ctx,Index distance0, BoundLayout<NondetRegLayout> layout1)   {
NondetRegStruct x2 = NondetRegStruct{
  ._super = LOAD(LAYOUT_LOOKUP(layout1, _super), distance0),   ._layout = layout1};
return x2;
}
NondetRegStruct exec_NondetReg(ExecContext& ctx,Val arg0, BoundLayout<NondetRegLayout> layout1)   {
STORE(LAYOUT_LOOKUP(layout1, _super), arg0);
NondetRegStruct x2 = NondetRegStruct{
  ._super = LOAD(LAYOUT_LOOKUP(layout1, _super), 0),   ._layout = layout1};
return x2;
}
ComponentStruct exec_Component(ExecContext& ctx)   {
return ComponentStruct{
};
}
RegStruct back_Reg(ExecContext& ctx,Index distance0, BoundLayout<NondetRegLayout> layout1)   {
// Reg(<preamble>:4)
NondetRegStruct x2 = back_NondetReg(ctx,distance0, layout1);
return RegStruct{
  ._super = x2,   .reg = x2,   ._layout = layout1};
}
RegStruct exec_Reg(ExecContext& ctx,Val arg0, BoundLayout<NondetRegLayout> layout1)   {
NondetRegStruct x2 = exec_NondetReg(ctx,arg0, layout1);
// Reg(<preamble>:5)
EQZ((arg0 - x2._super), "Reg(<preamble>:5)");
return RegStruct{
  ._super = x2,   .reg = x2,   ._layout = layout1};
}
ComponentStruct exec_Log(ExecContext& ctx,std::string arg0, std::initializer_list<Val> arg1)   {
// Log(<preamble>:22)
INVOKE_EXTERN(ctx,log, arg0, arg1);
return ComponentStruct{
};
}
GetCycleStruct exec_GetCycle(ExecContext& ctx)   {
// GetCycle(zirgen/dsl/examples/fibonacci.zir:2)
Val x0 = INVOKE_EXTERN(ctx,getCycle);
return GetCycleStruct{
  ._super = x0};
}
IsZeroStruct exec_IsZero(ExecContext& ctx,Val arg0, BoundLayout<IsZeroLayout> layout1)   {
// IsZero(zirgen/dsl/examples/fibonacci.zir:6)
Val x2 = exec_Isz(ctx,arg0);
NondetRegStruct x3 = exec_NondetReg(ctx,x2, LAYOUT_LOOKUP(layout1, _super));
// IsZero(zirgen/dsl/examples/fibonacci.zir:9)
Val x4 = exec_Inv(ctx,arg0);
NondetRegStruct x5 = exec_NondetReg(ctx,x4, LAYOUT_LOOKUP(layout1, inv));
// IsZero(zirgen/dsl/examples/fibonacci.zir:12)
Val x6 = exec_Sub(ctx,Val(1), x3._super);
Val x7 = exec_Mul(ctx,x3._super, x6);
EQZ(x7, "IsZero(zirgen/dsl/examples/fibonacci.zir:12)");
// IsZero(zirgen/dsl/examples/fibonacci.zir:14)
Val x8 = exec_Mul(ctx,arg0, x5._super);
Val x9 = exec_Sub(ctx,Val(1), x3._super);
EQZ((x8 - x9), "IsZero(zirgen/dsl/examples/fibonacci.zir:14)");
// IsZero(zirgen/dsl/examples/fibonacci.zir:16)
Val x10 = exec_Mul(ctx,x3._super, arg0);
EQZ(x10, "IsZero(zirgen/dsl/examples/fibonacci.zir:16)");
// IsZero(zirgen/dsl/examples/fibonacci.zir:18)
Val x11 = exec_Mul(ctx,x3._super, x5._super);
EQZ(x11, "IsZero(zirgen/dsl/examples/fibonacci.zir:18)");
return IsZeroStruct{
  ._super = x3,   .isZero = x3,   .inv = x5,   ._layout = layout1};
}
CycleCounterStruct exec_CycleCounter(ExecContext& ctx,BoundLayout<CycleCounterLayout> layout0, GlobalBuf global1)   {
// CycleCounter(zirgen/dsl/examples/fibonacci.zir:28)
BoundLayout<_globalLayout> x2 = BIND_LAYOUT(kLayoutGlobal, global1);
// CycleCounter(zirgen/dsl/examples/fibonacci.zir:29)
NondetRegStruct x3 = exec_NondetReg(ctx,Val(6), LAYOUT_LOOKUP(x2, totalCycles));
// CycleCounter(zirgen/dsl/examples/fibonacci.zir:31)
GetCycleStruct x4 = exec_GetCycle(ctx);
NondetRegStruct x5 = exec_NondetReg(ctx,x4._super, LAYOUT_LOOKUP(layout0, _super));
// CycleCounter(zirgen/dsl/examples/fibonacci.zir:32)
IsZeroStruct x6 = exec_IsZero(ctx,x5._super, LAYOUT_LOOKUP(layout0, isFirstCycle));
// CycleCounter(zirgen/dsl/examples/fibonacci.zir:34)
Val x7 = exec_Sub(ctx,Val(1), x6._super._super);
ComponentStruct x8;
if (to_size_t(x6._super._super)) {
ComponentStruct x9 = exec_Component(ctx);
x8 = x9;
} else if (to_size_t(x7)) {
// CycleCounter(zirgen/dsl/examples/fibonacci.zir:39)
NondetRegStruct x10 = back_NondetReg(ctx,1, LAYOUT_LOOKUP(layout0, _super));
Val x11 = exec_Add(ctx,x10._super, Val(1));
EQZ((x5._super - x11), "CycleCounter(zirgen/dsl/examples/fibonacci.zir:39)");
// CycleCounter(zirgen/dsl/examples/fibonacci.zir:37)
ComponentStruct x12 = exec_Component(ctx);
x8 = x12;
} else {
   assert(0 && "Reached unreachable mux arm");
}
return CycleCounterStruct{
  ._super = x5,   .cycle = x5,   .isFirstCycle = x6,   ._layout = layout0};
}
TopStruct exec_Top(ExecContext& ctx,BoundLayout<TopLayout> layout0, GlobalBuf global1)   {
// Top(zirgen/dsl/examples/fibonacci.zir:44)
BoundLayout<_globalLayout> x2 = BIND_LAYOUT(kLayoutGlobal, global1);
// Top(zirgen/dsl/examples/fibonacci.zir:49)
CycleCounterStruct x3 = exec_CycleCounter(ctx,LAYOUT_LOOKUP(layout0, cycle), global1);
// Top(zirgen/dsl/examples/fibonacci.zir:55)
Val x4 = exec_Sub(ctx,Val(1), x3.isFirstCycle._super._super);
RegStruct x5;
if (to_size_t(x3.isFirstCycle._super._super)) {
// Top(zirgen/dsl/examples/fibonacci.zir:45)
RegStruct x6 = back_Reg(ctx,0, LAYOUT_LOOKUP(x2, f0));
x5 = x6;
} else if (to_size_t(x4)) {
// Top(zirgen/dsl/examples/fibonacci.zir:55)
RegStruct x7 = back_Reg(ctx,1, LAYOUT_LOOKUP(layout0, d2));
x5 = x7;
} else {
   assert(0 && "Reached unreachable mux arm");
}
RegStruct x8 = exec_Reg(ctx,x5._super._super, LAYOUT_LOOKUP(layout0, d1));
// Top(zirgen/dsl/examples/fibonacci.zir:56)
Val x9 = exec_Sub(ctx,Val(1), x3.isFirstCycle._super._super);
RegStruct x10;
if (to_size_t(x3.isFirstCycle._super._super)) {
// Top(zirgen/dsl/examples/fibonacci.zir:46)
RegStruct x11 = back_Reg(ctx,0, LAYOUT_LOOKUP(x2, f1));
x10 = x11;
} else if (to_size_t(x9)) {
// Top(zirgen/dsl/examples/fibonacci.zir:56)
RegStruct x12 = back_Reg(ctx,1, LAYOUT_LOOKUP(layout0, d3));
x10 = x12;
} else {
   assert(0 && "Reached unreachable mux arm");
}
RegStruct x13 = exec_Reg(ctx,x10._super._super, LAYOUT_LOOKUP(layout0, d2));
// Top(zirgen/dsl/examples/fibonacci.zir:59)
Val x14 = exec_Add(ctx,x8._super._super, x13._super._super);
RegStruct x15 = exec_Reg(ctx,x14, LAYOUT_LOOKUP(layout0, d3));
// Top(zirgen/dsl/examples/fibonacci.zir:47)
RegStruct x16 = back_Reg(ctx,0, LAYOUT_LOOKUP(x2, steps));
// Top(zirgen/dsl/examples/fibonacci.zir:62)
Val x17 = exec_Sub(ctx,x3._super._super, x16._super._super);
Val x18 = exec_Add(ctx,x17, Val(1));
IsZeroStruct x19 = exec_IsZero(ctx,x18, LAYOUT_LOOKUP(layout0, terminate));
// Top(zirgen/dsl/examples/fibonacci.zir:63)
Val x20 = exec_Sub(ctx,Val(1), x19._super._super);
ComponentStruct x21;
if (to_size_t(x19._super._super)) {
// Top(zirgen/dsl/examples/fibonacci.zir:64)
RegStruct x22 = exec_Reg(ctx,x15._super._super, LAYOUT_LOOKUP(x2, fLast));
RegStruct x23 = back_Reg(ctx,0, LAYOUT_LOOKUP(x2, fLast));
// Top(zirgen/dsl/examples/fibonacci.zir:65)
std::initializer_list<Val> x24 = std::initializer_list<Val>{x23._super._super};
ComponentStruct x25 = exec_Log(ctx,"f_last = %u", x24);
// Top(zirgen/dsl/examples/fibonacci.zir:63)
ComponentStruct x26 = exec_Component(ctx);
x21 = x26;
} else if (to_size_t(x20)) {
// Top(zirgen/dsl/examples/fibonacci.zir:66)
ComponentStruct x27 = exec_Component(ctx);
x21 = x27;
} else {
   assert(0 && "Reached unreachable mux arm");
}
// Top(zirgen/dsl/examples/fibonacci.zir:44)
ComponentStruct x28 = exec_Component(ctx);
return TopStruct{
  ._super = x28,   .cycle = x3,   .first = x3.isFirstCycle,   .d1 = x8,   .d2 = x13,   .d3 = x15,   .terminate = x19,   ._layout = layout0};
}
void step_Top(ExecContext& ctx,MutableBuf data0, GlobalBuf global1)   {
BoundLayout<TopLayout> x2 = BIND_LAYOUT(kLayout_Top, data0);
TopStruct x3 = exec_Top(ctx,x2, global1);
return ;
}
MixState validityTaps(ValidityTapsContext& ctx,ConstantExtBuf taps0, PolyMix polyMix1, GlobalBuf global2)   {
ExtVal x3 = get(ctx,taps0, 2, 0);
ExtVal x4 = get(ctx,taps0, 4, 0);
ExtVal x5 = get(ctx,taps0, 5, 0);
ExtVal x6 = get(ctx,taps0, 6, 0);
ExtVal x7 = get(ctx,taps0, 8, 0);
ExtVal x8 = get(ctx,taps0, 10, 0);
ExtVal x9 = get(ctx,taps0, 11, 0);
ExtVal x10 = get(ctx,taps0, 12, 0);
// CycleCounter(zirgen/dsl/examples/fibonacci.zir:34)
// Top(zirgen/dsl/examples/fibonacci.zir:49)
MixState x11 = trivialConstraint();
BoundLayout<_globalLayout> x12 = BIND_LAYOUT(kLayoutGlobal, global2);
// Top(zirgen/dsl/examples/fibonacci.zir:62)
ExtVal x13 = ((x3 - LOAD(LAYOUT_LOOKUP(x12, steps._super), 0)) + ExtVal(1, 0, 0, 0));
// IsZero(zirgen/dsl/examples/fibonacci.zir:12)
ExtVal x14 = (ExtVal(1, 0, 0, 0) - x9);
// CycleCounter(zirgen/dsl/examples/fibonacci.zir:32)
// Top(zirgen/dsl/examples/fibonacci.zir:49)
ExtVal x15 = (ExtVal(1, 0, 0, 0) - x4);
// IsZero(zirgen/dsl/examples/fibonacci.zir:14)
MixState x16 = andEqzExt(ctx,andEqzExt(ctx,x11, (x4 * x15)), ((x3 * x5) - x15));
// CycleCounter(zirgen/dsl/examples/fibonacci.zir:34)
MixState x17 = andCondExt(andEqzExt(ctx,andEqzExt(ctx,x16, (x4 * x3)), (x4 * x5)), x15, andEqzExt(ctx,x11, (x3 - (get(ctx,taps0, 3, 0) + ExtVal(1, 0, 0, 0)))));
// Top(zirgen/dsl/examples/fibonacci.zir:55)
ExtVal x18 = (LOAD(LAYOUT_LOOKUP(x12, f0._super), 0) * x4);
// Top(zirgen/dsl/examples/fibonacci.zir:56)
ExtVal x19 = (LOAD(LAYOUT_LOOKUP(x12, f1._super), 0) * x4);
// Reg(<preamble>:5)
MixState x20 = andEqzExt(ctx,andEqzExt(ctx,x17, ((x18 + (get(ctx,taps0, 7, 0) * x15)) - x8)), ((x19 + (get(ctx,taps0, 9, 0) * x15)) - x6));
// IsZero(zirgen/dsl/examples/fibonacci.zir:14)
// Top(zirgen/dsl/examples/fibonacci.zir:62)
MixState x21 = andEqzExt(ctx,andEqzExt(ctx,andEqzExt(ctx,x20, ((x8 + x6) - x7)), (x9 * x14)), ((x13 * x10) - x14));
// Top(zirgen/dsl/examples/fibonacci.zir:63)
MixState x22 = andCondExt(andEqzExt(ctx,andEqzExt(ctx,x21, (x9 * x13)), (x9 * x10)), x9, andEqzExt(ctx,x11, (x7 - LOAD(LAYOUT_LOOKUP(x12, fLast._super), 0))));
return x22;
}
MixState validityRegs(ValidityRegsContext& ctx,PolyMix polyMix0, MutableBuf data1, GlobalBuf global2)   {
BoundLayout<TopLayout> x3 = BIND_LAYOUT(kLayout_Top, data1);
BoundLayout<_globalLayout> x4 = BIND_LAYOUT(kLayoutGlobal, global2);
// CycleCounter(zirgen/dsl/examples/fibonacci.zir:39)
// Top(zirgen/dsl/examples/fibonacci.zir:49)
Val x5 = (LOAD(LAYOUT_LOOKUP(x3, cycle._super._super), 1) + Val(1));
Val x6 = (LOAD(LAYOUT_LOOKUP(x3, cycle._super._super), 0) - x5);
// Top(zirgen/dsl/examples/fibonacci.zir:59)
Val x7 = (LOAD(LAYOUT_LOOKUP(x3, d1._super), 0) + LOAD(LAYOUT_LOOKUP(x3, d2._super), 0));
// Top(zirgen/dsl/examples/fibonacci.zir:62)
Val x8 = (LOAD(LAYOUT_LOOKUP(x3, cycle._super._super), 0) - LOAD(LAYOUT_LOOKUP(x4, steps._super), 0));
Val x9 = (x8 + Val(1));
// IsZero(zirgen/dsl/examples/fibonacci.zir:12)
Val x10 = (Val(1) - LOAD(LAYOUT_LOOKUP(x3, terminate._super._super), 0));
Val x11 = (LOAD(LAYOUT_LOOKUP(x3, terminate._super._super), 0) * x10);
// IsZero(zirgen/dsl/examples/fibonacci.zir:14)
Val x12 = (Val(1) - LOAD(LAYOUT_LOOKUP(x3, terminate._super._super), 0));
// IsZero(zirgen/dsl/examples/fibonacci.zir:16)
Val x13 = (LOAD(LAYOUT_LOOKUP(x3, terminate._super._super), 0) * x9);
// IsZero(zirgen/dsl/examples/fibonacci.zir:18)
Val x14 = (LOAD(LAYOUT_LOOKUP(x3, terminate._super._super), 0) * LOAD(LAYOUT_LOOKUP(x3, terminate.inv._super), 0));
// Reg(<preamble>:5)
// Top(zirgen/dsl/examples/fibonacci.zir:64)
Val x15 = (LOAD(LAYOUT_LOOKUP(x3, d3._super), 0) - LOAD(LAYOUT_LOOKUP(x4, fLast._super), 0));
// IsZero(zirgen/dsl/examples/fibonacci.zir:12)
// CycleCounter(zirgen/dsl/examples/fibonacci.zir:32)
// Top(zirgen/dsl/examples/fibonacci.zir:49)
Val x16 = (Val(1) - LOAD(LAYOUT_LOOKUP(x3, cycle.isFirstCycle._super._super), 0));
Val x17 = (LOAD(LAYOUT_LOOKUP(x3, cycle.isFirstCycle._super._super), 0) * x16);
// IsZero(zirgen/dsl/examples/fibonacci.zir:14)
Val x18 = (LOAD(LAYOUT_LOOKUP(x3, cycle._super._super), 0) * LOAD(LAYOUT_LOOKUP(x3, cycle.isFirstCycle.inv._super), 0));
Val x19 = (Val(1) - LOAD(LAYOUT_LOOKUP(x3, cycle.isFirstCycle._super._super), 0));
MixState x20 = andEqz(ctx,andEqz(ctx,trivialConstraint(), x17), (x18 - x19));
// IsZero(zirgen/dsl/examples/fibonacci.zir:16)
Val x21 = (LOAD(LAYOUT_LOOKUP(x3, cycle.isFirstCycle._super._super), 0) * LOAD(LAYOUT_LOOKUP(x3, cycle._super._super), 0));
// IsZero(zirgen/dsl/examples/fibonacci.zir:18)
Val x22 = (LOAD(LAYOUT_LOOKUP(x3, cycle.isFirstCycle._super._super), 0) * LOAD(LAYOUT_LOOKUP(x3, cycle.isFirstCycle.inv._super), 0));
// CycleCounter(zirgen/dsl/examples/fibonacci.zir:34)
Val x23 = (Val(1) - LOAD(LAYOUT_LOOKUP(x3, cycle.isFirstCycle._super._super), 0));
MixState x24 = andCond(andEqz(ctx,andEqz(ctx,x20, x21), x22), x23, andEqz(ctx,trivialConstraint(), x6));
// Top(zirgen/dsl/examples/fibonacci.zir:55)
Val x25 = (Val(1) - LOAD(LAYOUT_LOOKUP(x3, cycle.isFirstCycle._super._super), 0));
Val x26 = (LOAD(LAYOUT_LOOKUP(x4, f0._super), 0) * LOAD(LAYOUT_LOOKUP(x3, cycle.isFirstCycle._super._super), 0));
Val x27 = (LOAD(LAYOUT_LOOKUP(x3, d2._super), 1) * x25);
// Reg(<preamble>:5)
Val x28 = ((x26 + x27) - LOAD(LAYOUT_LOOKUP(x3, d1._super), 0));
// Top(zirgen/dsl/examples/fibonacci.zir:56)
Val x29 = (Val(1) - LOAD(LAYOUT_LOOKUP(x3, cycle.isFirstCycle._super._super), 0));
Val x30 = (LOAD(LAYOUT_LOOKUP(x4, f1._super), 0) * LOAD(LAYOUT_LOOKUP(x3, cycle.isFirstCycle._super._super), 0));
Val x31 = (LOAD(LAYOUT_LOOKUP(x3, d3._super), 1) * x29);
// Reg(<preamble>:5)
Val x32 = ((x30 + x31) - LOAD(LAYOUT_LOOKUP(x3, d2._super), 0));
// Top(zirgen/dsl/examples/fibonacci.zir:59)
MixState x33 = andEqz(ctx,andEqz(ctx,andEqz(ctx,x24, x28), x32), (x7 - LOAD(LAYOUT_LOOKUP(x3, d3._super), 0)));
// IsZero(zirgen/dsl/examples/fibonacci.zir:14)
// Top(zirgen/dsl/examples/fibonacci.zir:62)
MixState x34 = andEqz(ctx,andEqz(ctx,x33, x11), ((x9 * LOAD(LAYOUT_LOOKUP(x3, terminate.inv._super), 0)) - x12));
// Top(zirgen/dsl/examples/fibonacci.zir:63)
MixState x35 = andCond(andEqz(ctx,andEqz(ctx,x34, x13), x14), LOAD(LAYOUT_LOOKUP(x3, terminate._super._super), 0), andEqz(ctx,trivialConstraint(), x15));
return x35;
}

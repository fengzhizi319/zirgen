set_field!(BabyBear);
define_buffer_list!{
all: [accum,code,data,global,mix,],
rows: [accum,code,data,],
taps: [accum,code,data,],
globals: [global,mix,],}
define_tap_buffer!{accum, /*count=*/1, /*groupId=*/0}
define_tap_buffer!{code, /*count=*/1, /*groupId=*/1}
define_tap_buffer!{data, /*count=*/1, /*groupId=*/2}
define_global_buffer!{global, /*count=*/4}
define_global_buffer!{mix, /*count=*/4}
pub type Tap3Array = [Tap; 3];
pub const TAP_LIST: Tap3Array = [make_tap!(0, 0, 0), make_tap!(1, 0, 0), make_tap!(2, 0, 0)];
inputFilename: zirgen/dsl/examples/hello_world.zir

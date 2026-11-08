from std.builtin.variadics import TypeList
from std.compile import compile_info
from std.sys.info import CompilationTarget, _current_target, _TargetType
from std._plugin import StdPlugin, PluginHooks, TargetDeviceKind
from std.collections.string.string_span import _get_kgen_string


struct PhantomHooks[target: CompilationTarget](PluginHooks):
    comptime device_kind = TargetDeviceKind.other("phantom", True)


struct PhantomPlugin(StdPlugin):
    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["phantom"]()
    comptime HookFor[target: CompilationTarget] = PhantomHooks[target]


comptime PhantomTarget: _TargetType = __mlir_attr[
    `#kgen.target<triple = "aarch64-unknown-linux-gnu", `,
    `arch = "generic", `,
    `features = "+neon", `,
    `data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-`,
    `i64:64-i128:128-n32:64-S128-Fn32", `,
    `index_bit_width = 64, `,
    `simd_bit_width = 128, `,
    `opaque_plugin = `,
    TypeList.of[PhantomPlugin]().values,
    `> : !kgen.target`,
]


def main():
    comptime T = CompilationTarget[_mlir_value=PhantomTarget]
    print("is_accelerator:", T().is_accelerator(), "is_gpu:", T().is_gpu())
    comptime match T.stdlib_plugin.device_kind:
        case .other(o):
            print("device_kind: other(", o[0], ",", o[1], ")")
        case .cpu:
            print("device_kind: cpu")
        case .gpu:
            print("device_kind: gpu")

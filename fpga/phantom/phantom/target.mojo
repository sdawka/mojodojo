"""The phantom NPU as a Mojo compilation target.

`PhantomTarget` is a `#kgen.target` that carries a stdlib plugin
(`opaque_plugin`). Code that asks the target what it is gets the answer
`TargetDeviceKind.other("phantom", True)` (an accelerator that is not a GPU).

The triple is the host's aarch64 triple: the shipped compiler has no codegen
backend for a new triple (that needs a C++ TargetLowering/TargetBackend built
from the same source commit; see ../README.md). Device code therefore never
gets compiled *for* the phantom NPU; the tile is fixed-function and is driven
through `phantom.context`, whose launch runs the cycle-accurate simulator.
"""

from std.builtin.variadics import TypeList
from std.sys.info import CompilationTarget, _TargetType, _current_target
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

comptime Phantom = CompilationTarget[_mlir_value=PhantomTarget]
"""`CompilationTarget` for the phantom NPU."""


def device_kind_string[target: _TargetType = _current_target()]() -> String:
    """Renders a target's device kind, e.g. `other(phantom, accelerator=True)`.
    """
    comptime T = CompilationTarget[_mlir_value=target]
    comptime match T.stdlib_plugin.device_kind:
        case .other(o):
            return String(t"other({o[0]}, accelerator={o[1]})")
        case .cpu:
            return "cpu"
        case .gpu:
            return "gpu"

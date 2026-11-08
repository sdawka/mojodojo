// Shim for the tablegen-generated KGENEnums.h: TargetTraits.h only needs
// `EmitAs`. Values copied from Mojo/include/Mojo/KGENDialect/KGENEnums.td
// (modular@2841b316, Apache-2.0 WITH LLVM-exception). I32EnumAttr lowers to an
// `enum class : uint32_t`.
#pragma once
#include <cstdint>
namespace M::KGEN {
enum class EmitAs : uint32_t {
  ASM = 0, LLVM = 1, LLVM_OPT = 2, OBJECT = 3, LLVM_BITCODE = 4,
  LLVM_OPT_BITCODE = 5,
};
} // namespace M::KGEN

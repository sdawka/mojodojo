//===- phantom_traits.cpp - "phantom" NPU target traits compiler plugin ---===//
//
// Out-of-tree Mojo compiler plugin. The compiler dlopen()s every library in
// MODULAR_COMPILER_PLUGINS (or modular.cfg `mojo_plugin_paths`) and calls
// `M_KGEN_getTargetTraits` (Mojo/lib/Target/max_license/Plugin/
// PluginTraits.cpp). We return a `TargetTraits` that owns triples whose arch is
// "phantom", so the KGEN target registry stops rejecting them.
//
// ABI surface: only the TargetTraits vtable (declared in the pinned
// Mojo/lib/Target/TargetTraits.h) plus llvm::StringRef / ArrayRef /
// SmallVectorImpl / std::string / llvm::Triple layouts, all header-inline.
// No LLVM or KGEN symbol is linked.
//===----------------------------------------------------------------------===//

#include "Target/TargetTraits.h"

#include "llvm/TargetParser/Triple.h"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace M::KGEN;

namespace {

bool logEnabled() {
  static const bool on = std::getenv("PHANTOM_PLUGIN_LOG") != nullptr;
  return on;
}

#define PLOG(...)                                                             \
  do {                                                                         \
    if (logEnabled()) {                                                        \
      std::fprintf(stderr, "[phantom-plugin] " __VA_ARGS__);                   \
      std::fputc('\n', stderr);                                                \
    }                                                                          \
  } while (0)

class PhantomTraits final : public TargetTraits {
public:
  llvm::StringRef name() const override { return "phantom"; }

  bool matches(const llvm::Triple &triple) const override {
    // Triple::str() is inline (returns the stored `Data` string).
    llvm::StringRef t(triple.str());
    bool hit = t.starts_with("phantom-") || t == "phantom";
    PLOG("matches(\"%s\") -> %s", triple.str().c_str(), hit ? "true" : "false");
    return hit;
  }

  bool isGPU() const override { return false; }

  std::string codegenTriple(llvm::StringRef triple) const override {
    const char *ct = std::getenv("PHANTOM_CODEGEN_TRIPLE");
    std::string out = ct ? ct : triple.str();
    PLOG("codegenTriple(\"%s\") -> \"%s\"", triple.str().c_str(), out.c_str());
    return out;
  }

  llvm::StringRef getAsmExtension() const override { return ".phantom.s"; }
  llvm::StringRef getLLVMExtension() const override { return ".phantom.ll"; }
  llvm::StringRef getObjectExtension() const override { return ".phantom.o"; }
  llvm::StringRef getBitcodeExtension() const override { return ".phantom.bc"; }

  llvm::ArrayRef<EmissionKind> supportedEmissionKinds() const override {
    // TargetTraits::commonEmissionKinds() lives in the (non-exported) mojo
    // binary, so the table is restated here.
    static const EmissionKind kinds[] = {
        {"llvm", EmitAs::LLVM, "unoptimized LLVM IR (phantom NPU)"},
        {"asm", EmitAs::ASM, "assembly (phantom NPU)"},
        {"object", EmitAs::OBJECT, "object file (phantom NPU)"},
    };
    return kinds;
  }

  llvm::StringRef acceleratorSectionTitle() const override {
    return "Phantom NPU (simulated int8 systolic tile)";
  }
  llvm::ArrayRef<AcceleratorArch> supportedAcceleratorArchs() const override {
    static const AcceleratorArch archs[] = {
        {"phantom-n4", "4x4 int8 output-stationary tile (fpga/DESIGN.md)"},
        {"phantom-n8", "8x8 int8 output-stationary tile"},
        {"phantom-n16", "16x16 int8 output-stationary tile"},
    };
    return archs;
  }

protected:
  // true: do not gate behind a MAX install check.
  bool isBaseTarget() const override { return true; }
};

} // namespace

extern "C" __attribute__((visibility("default"))) const TargetTraits *
M_KGEN_getTargetTraits() {
  static PhantomTraits traits;
  PLOG("M_KGEN_getTargetTraits() called -> %p", (const void *)&traits);
  return &traits;
}

__attribute__((constructor)) static void phantomPluginLoaded() {
  PLOG("loaded (dlopen) into pid %d", (int)getpid());
}

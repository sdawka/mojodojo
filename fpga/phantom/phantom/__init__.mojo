"""Phantom NPU: the invented int8 systolic tile, presented to Mojo as a device.

- `phantom.target`: the compilation target + stdlib plugin that make
  `CompilationTarget.device_kind()` report `other("phantom", True)`.
- `phantom.device`: the device model (memory + int8 matmul) on the simulator.
- `phantom.context`: a `DeviceContext`-style API over the C-ABI driver
  library `libphantom.dylib` (built from `capi.mojo`).
"""

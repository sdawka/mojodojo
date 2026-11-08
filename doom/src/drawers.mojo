# SPDX-License-Identifier: GPL-2.0-or-later (see ../COPYING)
# Translation of doomgeneric r_draw.c drawers: Copyright(C) 1993-1996 Id Software, Inc.;
# Copyright(C) 2005-2014 Simon Howard. Mojo port for this repo.
# drawers.mojo -- X-DOOM-2: DOOM's column/span drawers (r_draw.c) in Mojo.
#
# Each kernel is a C-ABI export called by a marshalling stub in
# shim/r_draw_mojo.c, which passes the engine globals (dc_*, ds_*, ylookup,
# columnofs, ...) as arguments. Semantics follow upstream r_draw.c exactly:
# 32-bit two's-complement fixed point (Int32 arithmetic wraps like the C code
# compiles), arithmetic right shifts on fixed_t, the same RANGECHECK conditions
# (returned as 1 so the stub raises the identical I_Error), and the same side
# effects on dc_yl/dc_yh/fuzzpos/ds_x1/ds_x2 via pointers.
#
# Build with `-D DOOM_SIMD` for the vectorized span variant (still bit-exact).

from std.sys import is_defined

comptime SCREENWIDTH = 320
comptime SCREENHEIGHT = 200
comptime FRACBITS = 16
comptime FUZZTABLE = 50

comptime BytePtr = MutPointer[UInt8, MutUntrackedOrigin]
comptime IntPtr = MutPointer[Int32, MutUntrackedOrigin]
comptime YLookup = MutPointer[BytePtr, MutUntrackedOrigin]

comptime USE_SIMD = is_defined["DOOM_SIMD"]()


@inline(.always)
def _dest(ylookup: YLookup, columnofs: IntPtr, x: Int32, y: Int32) -> BytePtr:
    # ylookup[y] + columnofs[x]
    return ylookup[unsafe_offset=Int(y)].unsafe_offset(
        Int(columnofs[unsafe_offset=Int(x)])
    )


@inline(.always)
def _col_out_of_range(x: Int32, yl: Int32, yh: Int32) -> Bool:
    # (unsigned)dc_x >= SCREENWIDTH || dc_yl < 0 || dc_yh >= SCREENHEIGHT
    return x.cast[.uint32]() >= SCREENWIDTH or yl < 0 or yh >= SCREENHEIGHT


@inline(.always)
def _column[
    translated: Bool, low: Bool
](
    dest_in: BytePtr,
    dest2_in: BytePtr,
    count_in: Int32,
    frac_in: Int32,
    fracstep: Int32,
    source: BytePtr,
    colormap: BytePtr,
    translation: BytePtr,
):
    var dest = dest_in
    var dest2 = dest2_in
    var count = count_in
    var frac = frac_in
    # do { ... } while (count--);  -> count+1 pixels
    while True:
        var texel: Int
        comptime if translated:
            texel = Int(frac >> FRACBITS)  # upstream: no &127 when translated
        else:
            texel = Int((frac >> FRACBITS) & 127)
        var c = source[unsafe_offset=texel]
        comptime if translated:
            c = translation[unsafe_offset=Int(c)]
        var px = colormap[unsafe_offset=Int(c)]
        dest[] = px
        dest = dest.unsafe_offset(SCREENWIDTH)
        comptime if low:
            dest2[] = px
            dest2 = dest2.unsafe_offset(SCREENWIDTH)
        frac += fracstep
        if count == 0:
            break
        count -= 1


@export
def mojo_draw_column(
    ylookup: YLookup,
    columnofs: IntPtr,
    x: Int32,
    yl: Int32,
    yh: Int32,
    centery: Int32,
    iscale: Int32,
    texturemid: Int32,
    source: BytePtr,
    colormap: BytePtr,
    translation: BytePtr,
) abi("C") -> Int32:
    var count = yh - yl
    if count < 0:
        return 0
    if _col_out_of_range(x, yl, yh):
        return 1
    var frac = texturemid + (yl - centery) * iscale
    comptime if USE_SIMD:
        _column_unrolled(
            _dest(ylookup, columnofs, x, yl), count, frac, iscale, source, colormap
        )
    else:
        _column[False, False](
            _dest(ylookup, columnofs, x, yl),
            _dest(ylookup, columnofs, x, yl),
            count,
            frac,
            iscale,
            source,
            colormap,
            translation,
        )
    return 0


@inline(.always)
def _column_unrolled(
    dest_in: BytePtr,
    count_in: Int32,
    frac_in: Int32,
    fracstep: Int32,
    source: BytePtr,
    colormap: BytePtr,
):
    # Same pixels as _column[False, False]; 4 texel indices computed per step
    # as a SIMD vector (frac + i*fracstep wraps identically lane-wise).
    var dest = dest_in
    var n = Int(count_in) + 1
    var frac = frac_in
    comptime W = 4
    var lanes = SIMD[DType.int32, W](0, 1, 2, 3) * fracstep
    var step_w = fracstep * W
    while n >= W:
        var idx = ((SIMD[DType.int32, W](frac) + lanes) >> FRACBITS) & 127
        comptime for i in range(W):
            dest.unsafe_offset(i * SCREENWIDTH)[] = colormap[
                unsafe_offset=Int(source[unsafe_offset=Int(idx[i])])
            ]
        dest = dest.unsafe_offset(W * SCREENWIDTH)
        frac += step_w
        n -= W
    while n > 0:
        dest[] = colormap[
            unsafe_offset=Int(source[unsafe_offset=Int((frac >> FRACBITS) & 127)])
        ]
        dest = dest.unsafe_offset(SCREENWIDTH)
        frac += fracstep
        n -= 1


@export
def mojo_draw_column_low(
    ylookup: YLookup,
    columnofs: IntPtr,
    x: Int32,
    yl: Int32,
    yh: Int32,
    centery: Int32,
    iscale: Int32,
    texturemid: Int32,
    source: BytePtr,
    colormap: BytePtr,
    translation: BytePtr,
) abi("C") -> Int32:
    var count = yh - yl
    if count < 0:
        return 0
    if _col_out_of_range(x, yl, yh):  # upstream checks dc_x, not x<<1
        return 1
    var x2 = x << 1
    var frac = texturemid + (yl - centery) * iscale
    _column[False, True](
        _dest(ylookup, columnofs, x2, yl),
        _dest(ylookup, columnofs, x2 + 1, yl),
        count,
        frac,
        iscale,
        source,
        colormap,
        translation,
    )
    return 0


@export
def mojo_draw_translated_column(
    ylookup: YLookup,
    columnofs: IntPtr,
    x: Int32,
    yl: Int32,
    yh: Int32,
    centery: Int32,
    iscale: Int32,
    texturemid: Int32,
    source: BytePtr,
    colormap: BytePtr,
    translation: BytePtr,
) abi("C") -> Int32:
    var count = yh - yl
    if count < 0:
        return 0
    if _col_out_of_range(x, yl, yh):
        return 1
    var frac = texturemid + (yl - centery) * iscale
    _column[True, False](
        _dest(ylookup, columnofs, x, yl),
        _dest(ylookup, columnofs, x, yl),
        count,
        frac,
        iscale,
        source,
        colormap,
        translation,
    )
    return 0


@export
def mojo_draw_translated_column_low(
    ylookup: YLookup,
    columnofs: IntPtr,
    x: Int32,
    yl: Int32,
    yh: Int32,
    centery: Int32,
    iscale: Int32,
    texturemid: Int32,
    source: BytePtr,
    colormap: BytePtr,
    translation: BytePtr,
) abi("C") -> Int32:
    var count = yh - yl
    if count < 0:
        return 0
    var x2 = x << 1
    if _col_out_of_range(x2, yl, yh):  # upstream checks x<<1 here
        return 1
    var frac = texturemid + (yl - centery) * iscale
    _column[True, True](
        _dest(ylookup, columnofs, x2, yl),
        _dest(ylookup, columnofs, x2 + 1, yl),
        count,
        frac,
        iscale,
        source,
        colormap,
        translation,
    )
    return 0


@inline(.always)
def _fuzz[
    low: Bool
](
    ylookup: YLookup,
    columnofs: IntPtr,
    x: Int32,
    yl_p: IntPtr,
    yh_p: IntPtr,
    viewheight: Int32,
    colormaps: BytePtr,
    fuzzoffset: IntPtr,
    fuzzpos_p: IntPtr,
) -> Int32:
    if yl_p[] == 0:
        yl_p[] = 1
    if yh_p[] == viewheight - 1:
        yh_p[] = viewheight - 2
    var yl = yl_p[]
    var yh = yh_p[]
    var count = yh - yl
    if count < 0:
        return 0
    var xx: Int32
    comptime if low:
        xx = x << 1
    else:
        xx = x
    if _col_out_of_range(xx, yl, yh):
        return 1
    # Fuzz reads pixels one row above/below, including ones this loop just
    # wrote, so it is inherently sequential. frac is computed but unused
    # upstream; it has no observable effect, so it is omitted.
    var dest = _dest(ylookup, columnofs, xx, yl)
    var dest2 = _dest(ylookup, columnofs, xx + 1, yl)
    var fuzzpos = fuzzpos_p[]
    var shade = colormaps.unsafe_offset(6 * 256)
    while True:
        var off = Int(fuzzoffset[unsafe_offset=Int(fuzzpos)])
        dest[] = shade[unsafe_offset=Int(dest[unsafe_offset=off])]
        comptime if low:
            dest2[] = shade[unsafe_offset=Int(dest2[unsafe_offset=off])]
        fuzzpos += 1
        if fuzzpos == FUZZTABLE:
            fuzzpos = 0
        dest = dest.unsafe_offset(SCREENWIDTH)
        comptime if low:
            dest2 = dest2.unsafe_offset(SCREENWIDTH)
        if count == 0:
            break
        count -= 1
    fuzzpos_p[] = fuzzpos
    return 0


@export
def mojo_draw_fuzz_column(
    ylookup: YLookup,
    columnofs: IntPtr,
    x: Int32,
    yl_p: IntPtr,
    yh_p: IntPtr,
    viewheight: Int32,
    colormaps: BytePtr,
    fuzzoffset: IntPtr,
    fuzzpos_p: IntPtr,
) abi("C") -> Int32:
    return _fuzz[False](
        ylookup, columnofs, x, yl_p, yh_p, viewheight, colormaps, fuzzoffset, fuzzpos_p
    )


@export
def mojo_draw_fuzz_column_low(
    ylookup: YLookup,
    columnofs: IntPtr,
    x: Int32,
    yl_p: IntPtr,
    yh_p: IntPtr,
    viewheight: Int32,
    colormaps: BytePtr,
    fuzzoffset: IntPtr,
    fuzzpos_p: IntPtr,
) abi("C") -> Int32:
    return _fuzz[True](
        ylookup, columnofs, x, yl_p, yh_p, viewheight, colormaps, fuzzoffset, fuzzpos_p
    )


@inline(.always)
def _span_pack(xf: Int32, yf: Int32) -> UInt32:
    # ((xf << 10) & 0xffff0000) | ((yf >> 6) & 0x0000ffff), yf>>6 arithmetic
    return ((xf << 10).cast[.uint32]() & 0xFFFF0000) | (
        (yf >> 6).cast[.uint32]() & 0x0000FFFF
    )


@inline(.always)
def _span_out_of_range(y: Int32, x1: Int32, x2: Int32) -> Bool:
    return (
        x2 < x1
        or x1 < 0
        or x2 >= SCREENWIDTH
        or y.cast[.uint32]() > SCREENHEIGHT
    )


@export
def mojo_draw_span(
    ylookup: YLookup,
    columnofs: IntPtr,
    y: Int32,
    x1_p: IntPtr,
    x2_p: IntPtr,
    xfrac: Int32,
    yfrac: Int32,
    xstep: Int32,
    ystep: Int32,
    source: BytePtr,
    colormap: BytePtr,
) abi("C") -> Int32:
    var x1 = x1_p[]
    var x2 = x2_p[]
    if _span_out_of_range(y, x1, x2):
        return 1
    var position = _span_pack(xfrac, yfrac)
    var step = _span_pack(xstep, ystep)
    var dest = _dest(ylookup, columnofs, x1, y)
    var n = Int(x2 - x1) + 1
    comptime if USE_SIMD:
        comptime W = 8
        var lanes = SIMD[DType.uint32, W](0, 1, 2, 3, 4, 5, 6, 7) * step
        while n >= W:
            var pos = SIMD[DType.uint32, W](position) + lanes
            var spot = ((pos >> 4) & 0x0FC0) | (pos >> 26)
            var out = SIMD[DType.uint8, W]()
            comptime for i in range(W):
                out[i] = colormap[
                    unsafe_offset=Int(source[unsafe_offset=Int(spot[i])])
                ]
            dest.unsafe_store(out)
            dest = dest.unsafe_offset(W)
            position += step * W
            n -= W
    while n > 0:
        var spot = ((position >> 4) & 0x0FC0) | (position >> 26)
        dest[] = colormap[unsafe_offset=Int(source[unsafe_offset=Int(spot)])]
        dest = dest.unsafe_offset(1)
        position += step
        n -= 1
    return 0


@export
def mojo_draw_span_low(
    ylookup: YLookup,
    columnofs: IntPtr,
    y: Int32,
    x1_p: IntPtr,
    x2_p: IntPtr,
    xfrac: Int32,
    yfrac: Int32,
    xstep: Int32,
    ystep: Int32,
    source: BytePtr,
    colormap: BytePtr,
) abi("C") -> Int32:
    var x1 = x1_p[]
    var x2 = x2_p[]
    if _span_out_of_range(y, x1, x2):
        return 1
    var position = _span_pack(xfrac, yfrac)
    var step = _span_pack(xstep, ystep)
    var count = x2 - x1
    x1 <<= 1
    x2 <<= 1
    x1_p[] = x1  # upstream mutates ds_x1/ds_x2
    x2_p[] = x2
    var dest = _dest(ylookup, columnofs, x1, y)
    while True:
        var spot = ((position >> 4) & 0x0FC0) | (position >> 26)
        var px = colormap[unsafe_offset=Int(source[unsafe_offset=Int(spot)])]
        dest[] = px
        dest.unsafe_offset(1)[] = px
        dest = dest.unsafe_offset(2)
        position += step
        if count == 0:
            break
        count -= 1
    return 0


def keep_exports():
    # Called (never at runtime) from host.mojo's main so the compiler emits the
    # @export kernels above: exports in an imported module are otherwise dropped.
    var z = BytePtr(unsafe_from_address=16)  # dummy; never dereferenced
    var zi = IntPtr(unsafe_from_address=16)  # dummy; never dereferenced
    var zy = YLookup(unsafe_from_address=16)  # dummy; never dereferenced
    _ = mojo_draw_column(zy, zi, 0, 0, 0, 0, 0, 0, z, z, z)
    _ = mojo_draw_column_low(zy, zi, 0, 0, 0, 0, 0, 0, z, z, z)
    _ = mojo_draw_translated_column(zy, zi, 0, 0, 0, 0, 0, 0, z, z, z)
    _ = mojo_draw_translated_column_low(zy, zi, 0, 0, 0, 0, 0, 0, z, z, z)
    _ = mojo_draw_fuzz_column(zy, zi, 0, zi, zi, 0, z, zi, zi)
    _ = mojo_draw_fuzz_column_low(zy, zi, 0, zi, zi, 0, z, zi, zi)
    _ = mojo_draw_span(zy, zi, 0, zi, zi, 0, 0, 0, 0, z, z)
    _ = mojo_draw_span_low(zy, zi, 0, zi, zi, 0, 0, 0, 0, z, z)

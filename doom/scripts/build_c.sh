#!/usr/bin/env bash
# Compile the doomgeneric engine (no platform backend) + our C glue into static libs.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DG="$HERE/third_party/doomgeneric/doomgeneric"
B="$HERE/build"; O="$B/obj"
mkdir -p "$O"; touch "$B/.metadata_never_index"
CFLAGS="-O2 -g0 -w -DNORMALUNIX -DLINUX -D_DEFAULT_SOURCE -DDOOMGENERIC_RESX=320 -DDOOMGENERIC_RESY=200 -I$DG"
# Upstream Makefile object list minus the platform backend (doomgeneric_xlib.o) and r_draw.o.
CORE="dummy am_map doomdef doomstat dstrings d_event d_items d_iwad d_loop d_main d_mode d_net f_finale f_wipe g_game hu_lib hu_stuff info i_cdmus i_endoom i_joystick i_scale i_sound i_system i_timer memio m_argv m_bbox m_cheat m_config m_controls m_fixed m_menu m_misc m_random p_ceilng p_doors p_enemy p_floor p_inter p_lights p_map p_maputl p_mobj p_plats p_pspr p_saveg p_setup p_sight p_spec p_switch p_telept p_tick p_user r_bsp r_data r_main r_plane r_segs r_sky r_things sha1 sounds statdump st_lib st_stuff s_sound tables v_video wi_stuff w_checksum w_file w_main w_wad z_zone w_file_stdc i_input i_video doomgeneric"
objs=()
for f in $CORE; do
  if [ ! -f "$O/$f.o" ] || [ "$DG/$f.c" -nt "$O/$f.o" ]; then clang $CFLAGS -c "$DG/$f.c" -o "$O/$f.o"; fi
  objs+=("$O/$f.o")
done
rm -f "$B/libdoomcore.a"; ar rcs "$B/libdoomcore.a" "${objs[@]}"
clang $CFLAGS -c "$DG/r_draw.c" -o "$B/r_draw_c.o"                 # stock C drawers
clang $CFLAGS -c "$HERE/shim/r_draw_mojo.c" -o "$B/r_draw_mojo.o"  # stubs -> Mojo drawers
clang $CFLAGS -DDRAW_VERIFY -c "$HERE/shim/r_draw_mojo.c" -o "$B/r_draw_verify.o"  # C vs Mojo per-call check
clang $CFLAGS -c "$HERE/shim/dg_glue.c" -o "$B/dg_glue.o"
# Tier 3: recorders (+ per-frame GPU-vs-C verify variant) and the R_RenderPlayerView bracket.
clang $CFLAGS -c "$HERE/shim/r_draw_gpu.c" -o "$B/r_draw_gpu.o"
clang $CFLAGS -DDRAW_GPU_VERIFY -c "$HERE/shim/r_draw_gpu.c" -o "$B/r_draw_gpu_verify.o"
clang $CFLAGS -c "$HERE/shim/r_main_gpu.c" -o "$B/r_main_gpu.o"
clang $CFLAGS -c "$HERE/shim/drawbench.c" -o "$B/drawbench_c.o"                     # --microbench stub
clang $CFLAGS -DHAVE_MOJO_DRAW -c "$HERE/shim/drawbench.c" -o "$B/drawbench_mojo.o"  # C vs Mojo drawers
# Reference: the stock upstream SDL host, pure C (no sound), for tier-1 fps_reference.
if command -v sdl2-config >/dev/null; then
  clang $CFLAGS $(sdl2-config --cflags) -c "$DG/doomgeneric_sdl.c" -o "$O/doomgeneric_sdl.o"
  clang "$O/doomgeneric_sdl.o" "$B/r_draw_c.o" "$B/libdoomcore.a" $(sdl2-config --libs) -o "$B/doom-stock-sdl"
fi
echo "C build ok"

#!/bin/bash
set -eu

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

export NANO_MODULE_PATH="$repo_root/modules"

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/nanolang-graphical-demos.XXXXXX")"
trap 'rm -rf "$tmp_dir"' EXIT

bin/nanoc examples/gpu/ocean.nano -o "$tmp_dir/ocean"
bin/nanoc examples/advanced/ui_code_display_demo.nano -o "$tmp_dir/ui_code_display_demo"

grep -q 'if (not gpu_frame_ok)' examples/gpu/ocean.nano
grep -q 'SDL_RENDERER_SOFTWARE' examples/gpu/ocean.nano
grep -q 'nl_open_font_portable "DejaVuSansMono"' examples/advanced/ui_code_display_demo.nano
grep -q 'SDL_RENDERER_SOFTWARE' examples/advanced/ui_code_display_demo.nano

echo "Graphical demo initialization checks passed"

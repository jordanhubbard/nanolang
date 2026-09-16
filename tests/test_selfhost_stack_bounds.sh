#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

compiler=${NANOC_STACK_COMPILER:-./bin/nanoc}
cc_bin=${CC:-cc}
work=$(mktemp -d "${TMPDIR:-/tmp}/nanolang-stack-bounds.XXXXXX")
trap 'rm -rf "$work"' EXIT

rm -f obj/nano_modules/parser.o obj/nano_modules/parser.o.c \
    obj/nano_modules/transpiler.o obj/nano_modules/transpiler.o.c
NANO_VERBOSE_BUILD=1 "$compiler" src_nano/nanoc_v06.nano \
    -o "$work/nanoc" --keep-c >/dev/null

includes=(
    -I./src
    -I./src_nano/generated
    -I./src_nano
    -I./src_nano/compiler
    -I./modules/std
)

"$cc_bin" -std=c99 -O0 -fstack-usage "${includes[@]}" \
    -c obj/nano_modules/parser.o.c -o "$work/parser.o"
"$cc_bin" -std=c99 -O0 -fstack-usage "${includes[@]}" \
    -c obj/nano_modules/transpiler.o.c -o "$work/transpiler.o"

frame_size() {
    local report=$1
    local symbol=$2
    local size
    size=$(awk -F '\t' -v symbol="$symbol" '$1 ~ (symbol "$") { print $2; exit }' "$report")
    if [[ ! $size =~ ^[0-9]+$ ]]; then
        echo "I could not measure the O0 frame for $symbol" >&2
        exit 1
    fi
    printf '%s' "$size"
}

parse_frame=$(frame_size "$work/parser.su" parser__parse_primary)
generate_frame=$(frame_size "$work/transpiler.su" transpiler__generate_expression)
max_frame=$((64 * 1024))

if (( parse_frame > max_frame || generate_frame > max_frame )); then
    echo "I found an oversized self-host frame: parse_primary=$parse_frame, generate_expression=$generate_frame, limit=$max_frame" >&2
    exit 1
fi

deep_source="$work/deep_expression.nano"
{
    echo 'fn main() -> int {'
    printf '    let result: int = '
    for ((i = 0; i < 256; i++)); do printf '(+ 1 '; done
    printf '0'
    for ((i = 0; i < 256; i++)); do printf ')'; done
    printf '\n    (println (int_to_string result))\n    return 0\n}\nshadow main { assert true }\n'
} >"$deep_source"

# Exercise recursive parsing and generation with less stack than the former
# multi-megabyte record frames required.
(
    ulimit -s 3072
    "$work/nanoc" "$deep_source" -o "$work/deep_expression" >/dev/null
)
[[ $("$work/deep_expression") == 256 ]]

echo "I keep self-host recursion bounded: parse_primary=${parse_frame}B, generate_expression=${generate_frame}B."

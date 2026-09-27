#!/bin/sh
set -eu

version=43.0.0
archive=wasmtime-v${version}-x86_64-linux.tar.xz
sha256=e75a4933253fbc7b027c670b699490f163e3c86784f1db66581ae80fc0eb652c
url=https://github.com/bytecodealliance/wasmtime/releases/download/v${version}/${archive}
root=${RUNNER_TEMP:-/tmp}/nanolang-wasmtime-${version}
download=$root/$archive
install_dir=$root/bin

if [ "$(uname -s)" != Linux ] || [ "$(uname -m)" != x86_64 ]; then
    echo "I require my pinned x86-64 Linux Wasmtime CI host." >&2
    exit 1
fi

mkdir -p "$root" "$install_dir"
curl --fail --location --silent --show-error "$url" --output "$download"
printf '%s  %s\n' "$sha256" "$download" | sha256sum --check --status
tar -xJf "$download" -C "$root"
install -m 755 "$root/wasmtime-v${version}-x86_64-linux/wasmtime" "$install_dir/wasmtime"
"$install_dir/wasmtime" --version | grep -F "wasmtime ${version} " >/dev/null
if [ -n "${GITHUB_PATH:-}" ]; then
    printf '%s\n' "$install_dir" >> "$GITHUB_PATH"
fi

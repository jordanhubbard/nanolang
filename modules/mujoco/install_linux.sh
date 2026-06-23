#!/bin/sh
set -eu

script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)
repo_root=$(CDPATH= cd "$script_dir/../.." && pwd)
cd "$repo_root"

dest="${NANOLANG_NATIVE_DEPS:-.nanolang/native}/mujoco"

has_mujoco() {
    test -f "$dest/include/mujoco/mujoco.h" &&
        { test -f "$dest/lib/libmujoco.so" || test -f "$dest/bin/libmujoco.so"; }
}

if has_mujoco; then
    echo "I already have MuJoCo in $dest."
    exit 0
fi

case "$(uname -m)" in
    x86_64|amd64)
        asset_arch="linux-x86_64"
        ;;
    aarch64|arm64)
        asset_arch="linux-aarch64"
        ;;
    *)
        echo "I do not know which MuJoCo Linux asset matches $(uname -m)." >&2
        exit 1
        ;;
esac

command -v python3 >/dev/null 2>&1 || {
    echo "I need python3 to download MuJoCo from GitHub." >&2
    exit 1
}

url=$(python3 - "$asset_arch" <<'PY'
import json
import sys
import urllib.request

asset_arch = sys.argv[1]
api = "https://api.github.com/repos/google-deepmind/mujoco/releases/latest"

with urllib.request.urlopen(api, timeout=30) as response:
    release = json.load(response)

for asset in release.get("assets", []):
    name = asset.get("name", "")
    if asset_arch in name and (name.endswith(".tar.gz") or name.endswith(".tgz")):
        print(asset["browser_download_url"])
        break
else:
    raise SystemExit(f"I could not find a MuJoCo release asset for {asset_arch}.")
PY
)

tmp=$(mktemp -d "${TMPDIR:-/tmp}/nanolang-mujoco.XXXXXX")
trap 'rm -rf "$tmp" "$dest.tmp"' EXIT HUP INT TERM

archive="$tmp/mujoco.tar.gz"
echo "I am downloading MuJoCo for $asset_arch."
python3 - "$url" "$archive" <<'PY'
import sys
import urllib.request

url, archive = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(url, timeout=120) as response:
    with open(archive, "wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
PY

rm -rf "$dest.tmp"
mkdir -p "$(dirname "$dest")" "$dest.tmp"
tar -xzf "$archive" -C "$dest.tmp" --strip-components=1

test -f "$dest.tmp/include/mujoco/mujoco.h" || {
    echo "The MuJoCo archive did not contain include/mujoco/mujoco.h." >&2
    exit 1
}

if ! test -f "$dest.tmp/lib/libmujoco.so" && ! test -f "$dest.tmp/bin/libmujoco.so"; then
    echo "The MuJoCo archive did not contain libmujoco.so." >&2
    exit 1
fi

rm -rf "$dest"
mv "$dest.tmp" "$dest"
echo "I installed MuJoCo in $dest."

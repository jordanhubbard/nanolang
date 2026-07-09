#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

NANOC_BIN="./bin/nanoc"
[[ -x "$NANOC_BIN" ]] || { echo "error: $NANOC_BIN not found (run make build first)" >&2; exit 1; }

TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

mkdir -p "$TMP_ROOT/scripts"
cat > "$TMP_ROOT/scripts/nanoc-install.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf "%s\n" "$@" > "${NANO_CAPTURE_ARGS:?}"
EOF
chmod +x "$TMP_ROOT/scripts/nanoc-install.sh"

CAPTURE_FILE="$TMP_ROOT/args.txt"
NANO_CAPTURE_ARGS="$CAPTURE_FILE" NANOLANG_ROOT="$TMP_ROOT" "$NANOC_BIN" add mylib@^1.2.0 tools@latest

first_arg="$(sed -n '1p' "$CAPTURE_FILE")"
second_arg="$(sed -n '2p' "$CAPTURE_FILE")"
third_arg="$(sed -n '3p' "$CAPTURE_FILE")"

[[ "$first_arg" == "--save" ]]
[[ "$second_arg" == "mylib@^1.2.0" ]]
[[ "$third_arg" == "tools@latest" ]]

"$NANOC_BIN" --help | grep -q "add \[pkg@range ...\]"

echo "PASS: nanoc add forwards --save and is documented in --help"

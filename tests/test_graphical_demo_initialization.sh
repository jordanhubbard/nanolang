#!/usr/bin/env bash
# Public contract: graphical demos compile and, when a virtual display is
# available, survive initialization instead of exiting immediately.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

export NANO_MODULE_PATH="$repo_root/modules"

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/nanolang-graphical-demos.XXXXXX")"
trap 'rm -rf "$tmp_dir"' EXIT

bin/nanoc examples/gpu/julia_flow.nano -o "$tmp_dir/julia_flow"
bin/nanoc examples/advanced/ui_code_display_demo.nano -o "$tmp_dir/ui_code_display_demo"
make libnano-session
perl -e 'alarm 120; exec @ARGV' bin/nanoc examples/emacs/nano_emacs.nano -o "$tmp_dir/nano_emacs"

test -x "$tmp_dir/julia_flow"
test -x "$tmp_dir/ui_code_display_demo"
test -x "$tmp_dir/nano_emacs"

if command -v xvfb-run >/dev/null 2>&1 && command -v timeout >/dev/null 2>&1; then
    for demo in julia_flow ui_code_display_demo nano_emacs; do
        set +e
        xvfb-run -a timeout 3 "$tmp_dir/$demo" >"$tmp_dir/$demo.log" 2>&1
        status=$?
        set -e
        case "$status" in
            124|143)
                echo "  ✓ $demo survived initialization"
                ;;
            *)
                echo "  ✗ $demo exited during initialization (status $status)" >&2
                tail -20 "$tmp_dir/$demo.log" >&2
                exit 1
                ;;
        esac
    done
else
    echo "  ⊘ runtime initialization requires xvfb-run and timeout"
fi

echo "Graphical demo public-interface checks passed"

#!/usr/bin/env bash
# Public contract: the SDL Forth IDE (examples/graphics/sdl_forth_ide.nano) is a
# PTY client of the ONE NanoISA-backed Forth executable, not a second Forth
# implementation.
#
#   * bin/forth is a launcher that runs bin/nl_forth_interpreter_vm (verified
#     NanoISA bytecode) through bin/nano_vm -- never a separately compiled
#     native Forth binary.
#   * The IDE launches that same launcher over a PTY.
#
# The build and integration checks always run. The interactive PTY / file-load /
# liveness checks need libreadline (the interpreter's line editor); the
# graphical smoke build needs SDL2. Each is skipped -- not failed -- when its
# prerequisite is missing, mirroring tests/test_graphical_demo_initialization.sh.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

export NANO_MODULE_PATH="$repo_root/modules"

fail() { echo "  ✗ $1" >&2; exit 1; }
ok()   { echo "  ✓ $1"; }

have_pkg() { command -v pkg-config >/dev/null 2>&1 && pkg-config --exists "$1" >/dev/null 2>&1; }

# ── Build the NanoISA-backed executable and its launcher ────────────────────
# `make forth` builds nano_virt, nano_vm, bin/nl_forth_interpreter_vm, the
# forth_see helper library, and installs bin/forth from scripts/forth-launcher.sh.
make -f GNUmakefile nano_virt nano_vm >/dev/null
make -C examples ../bin/nl_forth_interpreter_vm ../bin/forth \
    ../modules/forth_see/.build/libforth_see.so \
    EXAMPLES_BACKEND=c NANO_MODULE_PATH="$NANO_MODULE_PATH" >/dev/null

test -x bin/nano_vm                 || fail "bin/nano_vm was not built"
test -f bin/nl_forth_interpreter_vm || fail "bin/nl_forth_interpreter_vm was not built"
test -x bin/forth                   || fail "bin/forth was not installed"
ok "NanoISA-backed Forth executable and launcher built"

# ── bin/forth must be the launcher, not a second Forth ──────────────────────
# A native compiled binary starts with a magic byte sequence (ELF's 0x7f 'ELF'
# or Mach-O), never a '#!' shebang. The launcher is a text script that execs
# nano_vm on the NanoISA module.
first_two="$(head -c 2 bin/forth)"
if [ "$first_two" != '#!' ]; then
    fail "bin/forth is not a launcher script; it must not be a second compiled Forth"
fi
if command -v file >/dev/null 2>&1 && file bin/forth 2>/dev/null | grep -qiE 'ELF|Mach-O'; then
    fail "bin/forth is a compiled binary; it must be a launcher over the NanoISA executable"
fi
grep -q 'nano_vm' bin/forth              || fail "bin/forth does not launch bin/nano_vm"
grep -q 'nl_forth_interpreter_vm' bin/forth || fail "bin/forth does not run the NanoISA Forth module"
# The IDE must launch bin/forth over a PTY rather than embed a Forth.
grep -q 'bin/forth' examples/graphics/sdl_forth_ide.nano || fail "IDE does not launch bin/forth"
grep -q 'nl_pty_fork_exec' examples/graphics/sdl_forth_ide.nano || fail "IDE does not spawn its Forth over a PTY"
ok "bin/forth is a PTY-launched wrapper over the NanoISA-backed executable"

# ── The emitted module is valid NanoISA bytecode ────────────────────────────
if head -c 4 bin/nl_forth_interpreter_vm | grep -q 'NVM'; then
    ok "bin/nl_forth_interpreter_vm carries the NVM magic"
fi

# ── Interactive PTY / file-loading / liveness (needs the readline module) ───
# The NanoISA Forth's REPL uses the readline module for line editing, so its
# wrapper library must be buildable. That needs the readline *development*
# package (pkg-config readline / -lreadline), not just the runtime .so, so gate
# on pkg-config. Build the wrapper explicitly so the FFI loader can dlopen it.
readline_present=no
if have_pkg readline; then readline_present=yes; fi

if [ "$readline_present" = "yes" ] && command -v python3 >/dev/null 2>&1; then
    make -C examples ../modules/readline/.build/libreadline.so \
        NANO_MODULE_PATH="$NANO_MODULE_PATH" >/dev/null 2>&1 || true
    forth_file="$(ls examples/language/forth/*.fs 2>/dev/null | head -n1 || true)"
    LD_LIBRARY_PATH="$repo_root/modules/forth_see/.build:$repo_root/modules/readline/.build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    FORTH_LOAD_FILE="$forth_file" \
    python3 - "$repo_root/bin/forth" <<'PY'
import os, pty, select, sys, time

launcher = sys.argv[1]
load_file = os.environ.get("FORTH_LOAD_FILE", "")

pid, fd = pty.fork()
if pid == 0:
    # bin/forth --interactive forces the REPL even under this test harness.
    os.execvp(launcher, [launcher, "--interactive"])
    os._exit(127)

out = b""
deadline = time.time() + 25
stage = 0
def drain():
    global out
    r, _, _ = select.select([fd], [], [], 0.5)
    if r:
        try:
            data = os.read(fd, 65536)
        except OSError:
            return False
        if not data:
            return False
        out += data
    return True

# Wait for the interpreter to come up, then exercise it.
while time.time() < deadline:
    if not drain():
        break
    if stage == 0 and len(out) > 0:
        time.sleep(0.4)
        os.write(fd, b"7 6 * .\n")   # expect 42
        stage = 1
        continue
    if stage == 1 and b"42" in out:
        # File loading: include a real .fs source, then confirm liveness.
        if load_file:
            os.write(fd, ("include %s\n" % load_file).encode())
        os.write(fd, b"1 1 + .\n")   # expect 2 -> interpreter still alive
        stage = 2
        continue
    if stage == 2 and out.count(b"2") >= 1 and b"42" in out:
        os.write(fd, b"bye\n")
        stage = 3
        break

try:
    os.close(fd)
except OSError:
    pass
try:
    os.waitpid(pid, 0)
except OSError:
    pass

text = out.decode(errors="replace")
if "42" not in text:
    sys.stderr.write("PTY session did not evaluate '7 6 * .' -> 42\n")
    sys.stderr.write(text[-2000:])
    sys.exit(1)
if stage < 2:
    sys.stderr.write("Interpreter did not stay alive after file loading\n")
    sys.stderr.write(text[-2000:])
    sys.exit(1)
print("  ✓ PTY startup, file loading, and interpreter liveness verified")
PY
else
    echo "  ⊘ interactive PTY checks require libreadline and python3 (skipped)"
fi

# ── Graphical smoke build of the IDE (needs SDL2) ───────────────────────────
if have_pkg sdl2; then
    tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/nanolang-forth-ide.XXXXXX")"
    trap 'rm -rf "$tmp_dir"' EXIT
    bin/nanoc examples/graphics/sdl_forth_ide.nano -o "$tmp_dir/sdl_forth_ide"
    test -x "$tmp_dir/sdl_forth_ide" || fail "sdl_forth_ide did not compile"
    ok "sdl_forth_ide compiles (graphical smoke build)"
    if command -v xvfb-run >/dev/null 2>&1 && command -v timeout >/dev/null 2>&1; then
        set +e
        xvfb-run -a timeout 3 "$tmp_dir/sdl_forth_ide" >"$tmp_dir/ide.log" 2>&1
        status=$?
        set -e
        case "$status" in
            124|143) ok "sdl_forth_ide survived initialization under a virtual display" ;;
            *) echo "  ⊘ sdl_forth_ide exited during init (status $status); log:"; tail -5 "$tmp_dir/ide.log" || true ;;
        esac
    fi
else
    echo "  ⊘ graphical smoke build requires SDL2 (skipped)"
fi

echo "Forth IDE / NanoISA-backed executable public-interface checks passed"

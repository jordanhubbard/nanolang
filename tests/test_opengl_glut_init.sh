#!/usr/bin/env bash
# ============================================================================
# GLUT initialization boundary — runtime coverage
# ============================================================================
#
# The OpenGL examples let GLFW own the window but draw GLUT primitives
# (glutSolidSphere, glutSolidTeapot, ...). GLUT reads its own global state
# there, so freeglut aborts the process with
#
#     freeglut ERROR: glutSolidSphere called without first calling glutInit.
#
# unless the program initializes GLUT first. modules/glut owns that shared
# boundary (glut_ensure_initialized); this test covers it at three levels:
#
#   1. boundary unit test — drives modules/glut/glut_init.c against a stub
#      GLUT, so it runs anywhere (only a C compiler is required)
#   2. call-site guard — every source under examples/ and modules/ that draws
#      GLUT primitives goes through the shared boundary and none re-implements
#      glutInit by hand
#   3. launch smoke — when the OpenGL toolchain and a display are present,
#      build both GLUT examples and check they survive launch
#   4. macOS framework guard — compiling the teapot must use the SDK's GLUT
#      framework without invoking Homebrew to install freeglut
#
# Usage:
#   bash tests/test_opengl_glut_init.sh            # all sections
#   bash tests/test_opengl_glut_init.sh --quick    # skip the launch smoke
#
# Sections 1 and 2 need no OpenGL toolchain and no display, so they are the
# regression net on headless machines and build agents. Section 3 skips
# (without failing) when its prerequisites are missing, listing every unmet
# prerequisite so the environment can be provisioned in one pass.
# ============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

QUICK=false
for arg in "$@"; do
    case "$arg" in
        --quick) QUICK=true ;;
        --help|-h) sed -n '2,26p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

FAILURES=0
LAUNCH_SECONDS="${GLUT_LAUNCH_SECONDS:-5}"
GLUT_ERROR_SIGNATURE='without first calling glutInit'

pass() { echo -e "${GREEN}✅${NC} $1"; }
skip() { echo -e "${YELLOW}⏭${NC}  SKIP: $1"; }
fail() { echo -e "${RED}❌${NC} $1"; FAILURES=$((FAILURES + 1)); }

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/nanolang-glut-init.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT

echo "=========================================="
echo "GLUT initialization boundary tests"
echo "=========================================="
echo ""

# ---------------------------------------------------------------------------
# 1. Boundary unit test (stub GLUT — no OpenGL toolchain needed)
# ---------------------------------------------------------------------------
echo "-- Boundary unit test --"

CC_BIN="${NANO_CC:-${CC:-cc}}"
if ! command -v "$CC_BIN" >/dev/null 2>&1; then
    skip "boundary unit test (C compiler '$CC_BIN' not found)"
else
    if "$CC_BIN" -D_POSIX_C_SOURCE=200809L \
        -Itests/stubs/glut -Imodules/glut \
        -o "$WORK_DIR/test_glut_init_boundary" \
        tests/test_glut_init_boundary.c modules/glut/glut_init.c \
        >"$WORK_DIR/build.log" 2>&1; then
        if "$WORK_DIR/test_glut_init_boundary"; then
            pass "glut init boundary unit test"
        else
            fail "glut init boundary unit test"
        fi
    else
        cat "$WORK_DIR/build.log"
        fail "compiling the glut init boundary unit test"
    fi
fi
echo ""

# ---------------------------------------------------------------------------
# 2. Call-site guard — primitives are drawn only behind the shared boundary
# ---------------------------------------------------------------------------
#
# This is the section that runs everywhere: the launch smoke below needs the
# OpenGL toolchain and a display, so on a headless machine or a build agent
# without freeglut/GLFW this guard is the only thing standing between a new
# call site and the "called without first calling glutInit" abort. It therefore
# scans every NanoLang source that draws primitives — examples/ and modules/
# alike — instead of only examples/opengl/, so a call site added elsewhere
# cannot slip past it.
echo "-- Call-site guard --"

GLUT_PRIMITIVE_SOURCES=0
while IFS= read -r source; do
    # modules/glut/glut.nano only declares the extern bindings; it is the
    # boundary itself, not a call site.
    [ "$source" = "modules/glut/glut.nano" ] && continue
    GLUT_PRIMITIVE_SOURCES=$((GLUT_PRIMITIVE_SOURCES + 1))

    if grep -q 'glut_ensure_initialized' "$source"; then
        pass "$source initializes GLUT through the shared boundary"
    else
        fail "$source draws GLUT primitives without calling glut_ensure_initialized"
    fi

    if grep -qE '\(glutInit ' "$source"; then
        fail "$source calls glutInit directly instead of glut_ensure_initialized"
    else
        pass "$source does not hand-roll glutInit"
    fi
done < <(grep -rlE '\(glut(Solid|Wire)[A-Za-z]+ ' \
    --include='*.nano' examples modules 2>/dev/null | sort)

if [ "$GLUT_PRIMITIVE_SOURCES" -eq 0 ]; then
    fail "no source draws GLUT primitives — the guard would never catch a regression"
fi
echo ""

# ---------------------------------------------------------------------------
# 3. macOS framework guard — a native framework must not trigger Homebrew
# ---------------------------------------------------------------------------
echo "-- macOS framework dependency guard --"

if [ "$(uname -s)" != "Darwin" ]; then
    skip "native GLUT framework check (not macOS)"
elif [ ! -x ./bin/nanoc_c ]; then
    skip "native GLUT framework check (run 'make build')"
else
    STUB_BIN="$WORK_DIR/stub-bin"
    mkdir -p "$STUB_BIN"
    REAL_BREW="$(command -v brew)"
    cat >"$STUB_BIN/brew" <<'STUB'
#!/usr/bin/env bash
case " $* " in
    *" freeglut "*) echo "$*" >>"$BREW_CALLS"; exit 99 ;;
    *) exec "$REAL_BREW" "$@" ;;
esac
STUB
    chmod +x "$STUB_BIN/brew"
    : >"$WORK_DIR/brew-calls"

    # The stub makes any package-manager invocation fail immediately. A normal
    # compile exercises module resolution and proves it reaches the linker.
    if PATH="$STUB_BIN:$PATH" BREW_CALLS="$WORK_DIR/brew-calls" REAL_BREW="$REAL_BREW" \
        NANO_MODULE_PATH="$PROJECT_ROOT/modules" \
        ./bin/nanoc_c examples/opengl/opengl_teapot.nano -o "$WORK_DIR/teapot" \
        >"$WORK_DIR/framework-build.log" 2>&1; then
        if [ -s "$WORK_DIR/brew-calls" ]; then
            cat "$WORK_DIR/brew-calls"
            fail "teapot dependency resolution invoked Homebrew for native GLUT"
        else
            pass "teapot uses the native GLUT framework without Homebrew"
        fi
    else
        cat "$WORK_DIR/framework-build.log"
        fail "teapot failed while resolving the native GLUT framework"
    fi
fi
echo ""

# ---------------------------------------------------------------------------
# 4. Launch smoke — both examples must survive launch on a real display
# ---------------------------------------------------------------------------
echo "-- Launch smoke --"

# Report *every* unmet prerequisite, not just the first one: whoever is
# provisioning the machine can then install the whole set in one pass instead of
# rediscovering the next missing piece on each run.
launch_prerequisite_missing() {
    if [ "$QUICK" = true ]; then
        echo "--quick requested"
        return 0
    fi

    # A plain string rather than an array: `set -u` makes an empty array an
    # unbound variable on the bash 3.2 that ships with macOS.
    local missing=""
    note_missing() { missing="${missing:+$missing; }$1"; }

    if [ ! -x ./bin/nanoc_c ]; then
        note_missing "./bin/nanoc_c not built (run 'make build')"
    fi
    if ! command -v perl >/dev/null 2>&1; then
        note_missing "perl not available to bound the run"
    fi
    if ! command -v pkg-config >/dev/null 2>&1; then
        note_missing "pkg-config not available"
    else
        local packages="glfw3 glew"
        if [ "$(uname -s)" != "Darwin" ]; then
            packages="glut $packages"
        fi
        for pkg in $packages; do
            if ! pkg-config --exists "$pkg" 2>/dev/null; then
                note_missing "$pkg development package not installed"
            fi
        done
    fi
    if [ "$(uname -s)" != "Darwin" ] && [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
        note_missing "no display available (headless)"
    fi

    if [ -z "$missing" ]; then
        return 1
    fi

    echo "$missing"
    return 0
}

launch_example() {
    local source="$1"
    local name
    name="$(basename "$source" .nano)"
    local binary="$WORK_DIR/$name"
    local log="$WORK_DIR/$name.log"

    if ! NANO_MODULE_PATH="$PROJECT_ROOT/modules" ./bin/nanoc_c "$source" -o "$binary" \
        >"$log" 2>&1; then
        cat "$log"
        fail "$name failed to build"
        return
    fi

    if [ "$(uname -s)" = "Darwin" ] && otool -L "$binary" | grep -q '/opt/homebrew/.*/libglut'; then
        fail "$name links Homebrew FreeGLUT alongside Apple GLUT.framework"
        return
    fi

    # Bound the run: these examples loop until the window closes, so surviving
    # until the alarm fires is the success case.
    perl -e "alarm $LAUNCH_SECONDS; exec @ARGV" "$binary" >"$log" 2>&1
    local status=$?

    if grep -q "$GLUT_ERROR_SIGNATURE" "$log"; then
        cat "$log"
        fail "$name aborted: GLUT primitives used before initialization"
        return
    fi

    case $status in
        # 0: clean exit, 142/124: still running when the alarm/timeout fired.
        0|124|142)
            pass "$name survived launch (${LAUNCH_SECONDS}s)"
            ;;
        *)
            if grep -qE 'Failed to initialize GLFW|Failed to create window' "$log"; then
                skip "$name (no usable GLFW window in this environment)"
            else
                cat "$log"
                fail "$name exited with status $status during launch"
            fi
            ;;
    esac
}

if reason="$(launch_prerequisite_missing)"; then
    skip "launch smoke ($reason)"
else
    for example in examples/opengl/opengl_solar_system.nano examples/opengl/opengl_teapot.nano; do
        launch_example "$example"
    done
fi
echo ""

# ---------------------------------------------------------------------------
echo "=========================================="
if [ $FAILURES -eq 0 ]; then
    echo -e "${GREEN}✅ GLUT initialization boundary tests passed${NC}"
    exit 0
fi
echo -e "${RED}❌ $FAILURES GLUT initialization check(s) failed${NC}"
exit 1

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
#   2. example guard — every example drawing GLUT primitives goes through the
#      shared boundary and none re-implements glutInit by hand
#   3. launch smoke — when the OpenGL toolchain and a display are present,
#      build both GLUT examples and check they survive launch
#
# Usage:
#   bash tests/test_opengl_glut_init.sh            # all sections
#   bash tests/test_opengl_glut_init.sh --quick    # skip the launch smoke
#
# Sections skip (without failing) when their prerequisites are missing.
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
# 2. Example guard — primitives are drawn only behind the shared boundary
# ---------------------------------------------------------------------------
echo "-- Example guard --"

GLUT_PRIMITIVE_EXAMPLES=()
while IFS= read -r example; do
    GLUT_PRIMITIVE_EXAMPLES+=("$example")
done < <(grep -lE '\(glut(Solid|Wire)[A-Za-z]+ ' examples/opengl/*.nano 2>/dev/null | sort)

if [ ${#GLUT_PRIMITIVE_EXAMPLES[@]} -eq 0 ]; then
    fail "no example draws GLUT primitives — the guard would never catch a regression"
else
    for example in "${GLUT_PRIMITIVE_EXAMPLES[@]}"; do
        if grep -q 'glut_ensure_initialized' "$example"; then
            pass "$example initializes GLUT through the shared boundary"
        else
            fail "$example draws GLUT primitives without calling glut_ensure_initialized"
        fi

        if grep -qE '\(glutInit ' "$example"; then
            fail "$example calls glutInit directly instead of glut_ensure_initialized"
        else
            pass "$example does not hand-roll glutInit"
        fi
    done
fi
echo ""

# ---------------------------------------------------------------------------
# 3. Launch smoke — both examples must survive launch on a real display
# ---------------------------------------------------------------------------
echo "-- Launch smoke --"

launch_prerequisite_missing() {
    if [ "$QUICK" = true ]; then
        echo "--quick requested"
        return 0
    fi
    if [ ! -x ./bin/nanoc_c ]; then
        echo "./bin/nanoc_c not built (run 'make build')"
        return 0
    fi
    if ! command -v perl >/dev/null 2>&1; then
        echo "perl not available to bound the run"
        return 0
    fi
    if ! command -v pkg-config >/dev/null 2>&1; then
        echo "pkg-config not available"
        return 0
    fi
    for pkg in glut glfw3 glew; do
        if ! pkg-config --exists "$pkg" 2>/dev/null; then
            echo "$pkg development package not installed"
            return 0
        fi
    done
    if [ "$(uname -s)" != "Darwin" ] && [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
        echo "no display available (headless)"
        return 0
    fi
    return 1
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

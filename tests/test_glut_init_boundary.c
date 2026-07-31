/*
 * test_glut_init_boundary.c — unit tests for modules/glut/glut_init.c
 *
 * The GLUT examples let GLFW own the window, so GLUT itself is only
 * initialized for its geometry helpers. This test drives that shared
 * initialization boundary against a stub GLUT (tests/stubs/glut) so it runs on
 * any machine, with or without an OpenGL toolchain.
 *
 * Tests:
 *   1. a headless process skips glutInit instead of letting freeglut exit(1)
 *   2. the first call with a display initializes GLUT
 *   3. glutInit gets a real argc/argv pair, not null pointers
 *   4. repeated calls are idempotent (freeglut rejects re-initialization)
 *   5. initialization state survives the display going away
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "glut_init.h"

/* -----------------------------------------------------------------------
 * Stub GLUT: records how the boundary called glutInit
 * ----------------------------------------------------------------------- */

static int  g_init_calls     = 0;
static int  g_last_argc      = -1;
static char g_last_argv0[64] = {0};
static int  g_argv_terminated = 0;

void glutInit(int *pargc, char **argv) {
    g_init_calls++;
    g_last_argc = pargc ? *pargc : -1;
    g_last_argv0[0] = '\0';
    g_argv_terminated = 0;
    if (argv) {
        if (argv[0]) {
            strncpy(g_last_argv0, argv[0], sizeof(g_last_argv0) - 1);
            g_last_argv0[sizeof(g_last_argv0) - 1] = '\0';
            g_argv_terminated = (argv[1] == NULL) ? 1 : 0;
        }
    }
}

/* -----------------------------------------------------------------------
 * Minimal test harness
 * ----------------------------------------------------------------------- */

static int g_pass = 0;
static int g_fail = 0;

#define ASSERT(cond, desc)                                              \
    do {                                                                \
        if (cond) {                                                     \
            printf("  PASS  %s\n", (desc));                             \
            g_pass++;                                                   \
        } else {                                                        \
            printf("  FAIL  %s  (line %d)\n", (desc), __LINE__);        \
            g_fail++;                                                   \
        }                                                               \
    } while (0)

static void clear_display(void) {
    unsetenv("DISPLAY");
    unsetenv("WAYLAND_DISPLAY");
}

/* -----------------------------------------------------------------------
 * Test 1: headless processes skip initialization
 *
 * freeglut's glutInit exits the process when it cannot reach a display, so the
 * boundary must not call it while headless. Skipped on macOS, where GLUT talks
 * to the window server through Cocoa and there is no DISPLAY to inspect.
 * ----------------------------------------------------------------------- */
static void test_headless_skips_init(void) {
#ifdef __APPLE__
    printf("[1] headless skip (not applicable on macOS)\n");
#else
    printf("[1] headless process skips glutInit\n");
    clear_display();
    ASSERT(nl_glut_ensure_init() == 0, "reports GLUT unavailable");
    ASSERT(g_init_calls == 0,          "glutInit not called while headless");
    ASSERT(nl_glut_is_initialized() == 0, "state stays uninitialized");
#endif
}

/* -----------------------------------------------------------------------
 * Test 2 + 3: first call initializes GLUT with a real argc/argv
 * ----------------------------------------------------------------------- */
static void test_first_call_initializes(void) {
    printf("[2] first call initializes GLUT\n");
    setenv("DISPLAY", ":0", 1);

    ASSERT(nl_glut_ensure_init() == 1,    "reports GLUT ready");
    ASSERT(g_init_calls == 1,             "glutInit called exactly once");
    ASSERT(nl_glut_is_initialized() == 1, "state reports initialized");

    printf("[3] glutInit receives a real argc/argv\n");
    ASSERT(g_last_argc == 1,              "argc is 1");
    ASSERT(g_last_argv0[0] != '\0',       "argv[0] is a non-empty program name");
    ASSERT(g_argv_terminated == 1,        "argv is NULL terminated");
}

/* -----------------------------------------------------------------------
 * Test 4: repeated calls are idempotent
 *
 * freeglut treats a second glutInit as a fatal error, so shared code that runs
 * before every primitive must initialize at most once.
 * ----------------------------------------------------------------------- */
static void test_repeated_calls_idempotent(void) {
    printf("[4] repeated calls are idempotent\n");
    ASSERT(nl_glut_ensure_init() == 1, "second call still reports ready");
    ASSERT(nl_glut_ensure_init() == 1, "third call still reports ready");
    ASSERT(g_init_calls == 1,          "glutInit still called only once");
}

/* -----------------------------------------------------------------------
 * Test 5: initialized state survives the display going away
 * ----------------------------------------------------------------------- */
static void test_state_survives_display_loss(void) {
    printf("[5] initialized state survives display loss\n");
    clear_display();
    ASSERT(nl_glut_ensure_init() == 1,    "still reports ready once initialized");
    ASSERT(g_init_calls == 1,             "glutInit not called again");
    ASSERT(nl_glut_is_initialized() == 1, "state still reports initialized");
}

int main(void) {
    printf("=== glut init boundary tests ===\n");
    test_headless_skips_init();
    test_first_call_initializes();
    test_repeated_calls_idempotent();
    test_state_survives_display_loss();
    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

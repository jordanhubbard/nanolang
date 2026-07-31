/* Shared GLUT initialization boundary for NanoLang.
 *
 * GLUT's geometry helpers (glutSolidSphere, glutSolidTeapot, glutSolidTorus,
 * ...) read the library's global state, so freeglut aborts the process with
 *
 *     freeglut ERROR: glutSolidSphere called without first calling glutInit.
 *
 * when a program draws them without initializing GLUT. Examples that let GLFW
 * create the window and drive the event loop still need that state, so this
 * shim initializes GLUT exactly once: it never calls glutCreateWindow and
 * never enters glutMainLoop, which leaves window and event-loop ownership with
 * GLFW.
 *
 * glutInit() takes a pointer to argc plus a NULL-terminated argv and reads
 * through both, so a real (argc, argv) pair is synthesized here instead of
 * passing null pointers down from NanoLang.
 */

#include "glut_platform.h"
#include "glut_init.h"

#include <stdlib.h>

static int nl_glut_initialized = 0;

static int nl_glut_display_available(void) {
#ifdef __APPLE__
    /* GLUT reaches the window server through Cocoa; there is no DISPLAY. */
    return 1;
#else
    const char *display = getenv("DISPLAY");
    if (display && display[0] != '\0') {
        return 1;
    }
    display = getenv("WAYLAND_DISPLAY");
    if (display && display[0] != '\0') {
        return 1;
    }
    return 0;
#endif
}

int64_t nl_glut_ensure_init(void) {
    if (nl_glut_initialized) {
        return 1;
    }

    /* freeglut calls exit(1) from glutInit when it cannot reach a display, so
     * stay out of its way when the process is headless. Callers report the
     * skip instead of dying inside the library. */
    if (!nl_glut_display_available()) {
        return 0;
    }

    {
        static char program_name[] = "nanolang";
        char *argv[2];
        int argc = 1;

        argv[0] = program_name;
        argv[1] = NULL;

        glutInit(&argc, argv);
    }

    nl_glut_initialized = 1;
    return 1;
}

int64_t nl_glut_is_initialized(void) {
    return nl_glut_initialized ? 1 : 0;
}

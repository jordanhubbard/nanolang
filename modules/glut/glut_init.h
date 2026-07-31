/* Shared GLUT initialization boundary for NanoLang.
 *
 * Declares the shim used by modules/glut/glut.nano so that programs which let
 * another toolkit (GLFW) own the window can still use GLUT's geometry
 * helpers.
 */

#ifndef NANOLANG_GLUT_INIT_H
#define NANOLANG_GLUT_INIT_H

#include <stdint.h>

/* Initialize GLUT once, without creating a GLUT window or entering
 * glutMainLoop. Returns 1 when GLUT state is ready (already initialized
 * counts), 0 when the process has no display and initialization was skipped.
 */
int64_t nl_glut_ensure_init(void);

/* Returns 1 once nl_glut_ensure_init has initialized GLUT, 0 otherwise. */
int64_t nl_glut_is_initialized(void);

#endif /* NANOLANG_GLUT_INIT_H */

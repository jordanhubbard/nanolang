/* Stub GLUT header used only by tests/test_glut_init_boundary.c.
 *
 * It stands in for <GL/glut.h> (and, through GLUT/glut.h, for the macOS
 * framework header) so the shared initialization boundary in
 * modules/glut/glut_init.c can be unit tested on machines without an OpenGL
 * toolchain. It is never used to build real programs.
 */

#ifndef NANOLANG_TEST_STUB_GLUT_H
#define NANOLANG_TEST_STUB_GLUT_H

/* Defined by the test, which records how the boundary calls it. */
void glutInit(int *pargc, char **argv);

#endif /* NANOLANG_TEST_STUB_GLUT_H */

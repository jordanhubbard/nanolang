/* GLUT Platform Header Wrapper
 * Handles different GLUT header locations across platforms
 */

#ifndef NANOLANG_GLUT_PLATFORM_H
#define NANOLANG_GLUT_PLATFORM_H

/* Silence OpenGL deprecation warnings on macOS */
#define GL_SILENCE_DEPRECATION 1

#ifdef __APPLE__
    /* macOS: GLUT.framework uses GLUT/glut.h */
    #include <GLUT/glut.h>
#else
    /* Linux/Windows: FreeGLUT uses GL/glut.h */
    #include <GL/glut.h>
#endif

#endif /* NANOLANG_GLUT_PLATFORM_H */

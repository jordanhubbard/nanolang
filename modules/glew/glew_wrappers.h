#ifndef GLEW_WRAPPERS_H
#define GLEW_WRAPPERS_H

#include <stdint.h>

/*
 * Wrappers for OpenGL + GLEW functions.
 *
 * nanolang's FFI uses int64_t for `int` and double for `float`.
 * Many OpenGL entry points take `float` (GLfloat) parameters.
 * These wrappers accept nanolang-friendly types and cast to the
 * correct OpenGL types before calling into the real GL/GLEW APIs.
 */

/* GLEW */
int64_t nlg_glewInit(void);
int64_t nlg_glewIsSupported(const char *extension);
const char* nlg_glewGetString(int64_t name);
const char* nlg_glewGetErrorString(int64_t error);

/* OpenGL error / strings */
int64_t nlg_glGetError(void);
const char* nlg_glGetString(int64_t name);

/* OpenGL float-parameter functions */
void nlg_glClearColor(double r, double g, double b, double a);
void nlg_glVertex2f(double x, double y);
void nlg_glVertex3f(double x, double y, double z);
void nlg_glColor3f(double r, double g, double b);
void nlg_glColor4f(double r, double g, double b, double a);
void nlg_glTranslatef(double x, double y, double z);
void nlg_glRotatef(double angle, double x, double y, double z);
void nlg_glScalef(double x, double y, double z);
void nlg_glLineWidth(double width);
void nlg_glPointSize(double size);
void nlg_glNormal3f(double nx, double ny, double nz);
void nlg_glRasterPos2f(double x, double y);
void nlg_glMaterialf(int64_t face, int64_t pname, double param);

// Wrapper for glLightfv that accepts individual float parameters
void nl_glLightfv4(int64_t light, int64_t pname, double x, double y, double z, double w);

// Wrapper for glMaterialfv that accepts individual float parameters
void nl_glMaterialfv4(int64_t face, int64_t pname, double x, double y, double z, double w);

#endif // GLEW_WRAPPERS_H

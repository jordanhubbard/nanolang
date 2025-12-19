#include <GL/glew.h>
#include "glew_wrappers.h"

int64_t nlg_glewInit(void) {
    return (int64_t)glewInit();
}

int64_t nlg_glewIsSupported(const char *extension) {
    return (int64_t)glewIsSupported(extension);
}

const char* nlg_glewGetString(int64_t name) {
    return (const char*)glewGetString((GLenum)name);
}

const char* nlg_glewGetErrorString(int64_t error) {
    return (const char*)glewGetErrorString((GLenum)error);
}

int64_t nlg_glGetError(void) {
    return (int64_t)glGetError();
}

const char* nlg_glGetString(int64_t name) {
    return (const char*)glGetString((GLenum)name);
}

void nlg_glClearColor(double r, double g, double b, double a) {
    glClearColor((GLclampf)r, (GLclampf)g, (GLclampf)b, (GLclampf)a);
}

void nlg_glVertex2f(double x, double y) {
    glVertex2f((GLfloat)x, (GLfloat)y);
}

void nlg_glVertex3f(double x, double y, double z) {
    glVertex3f((GLfloat)x, (GLfloat)y, (GLfloat)z);
}

void nlg_glColor3f(double r, double g, double b) {
    glColor3f((GLfloat)r, (GLfloat)g, (GLfloat)b);
}

void nlg_glColor4f(double r, double g, double b, double a) {
    glColor4f((GLfloat)r, (GLfloat)g, (GLfloat)b, (GLfloat)a);
}

void nlg_glTranslatef(double x, double y, double z) {
    glTranslatef((GLfloat)x, (GLfloat)y, (GLfloat)z);
}

void nlg_glRotatef(double angle, double x, double y, double z) {
    glRotatef((GLfloat)angle, (GLfloat)x, (GLfloat)y, (GLfloat)z);
}

void nlg_glScalef(double x, double y, double z) {
    glScalef((GLfloat)x, (GLfloat)y, (GLfloat)z);
}

void nlg_glLineWidth(double width) {
    glLineWidth((GLfloat)width);
}

void nlg_glPointSize(double size) {
    glPointSize((GLfloat)size);
}

void nlg_glNormal3f(double nx, double ny, double nz) {
    glNormal3f((GLfloat)nx, (GLfloat)ny, (GLfloat)nz);
}

void nlg_glRasterPos2f(double x, double y) {
    glRasterPos2f((GLfloat)x, (GLfloat)y);
}

void nlg_glMaterialf(int64_t face, int64_t pname, double param) {
    glMaterialf((GLenum)face, (GLenum)pname, (GLfloat)param);
}

// Wrapper for glLightfv that accepts individual float parameters
void nl_glLightfv4(int64_t light, int64_t pname, double x, double y, double z, double w) {
    GLfloat params[4] = {(GLfloat)x, (GLfloat)y, (GLfloat)z, (GLfloat)w};
    glLightfv((GLenum)light, (GLenum)pname, params);
}

// Wrapper for glMaterialfv that accepts individual float parameters
void nl_glMaterialfv4(int64_t face, int64_t pname, double x, double y, double z, double w) {
    GLfloat params[4] = {(GLfloat)x, (GLfloat)y, (GLfloat)z, (GLfloat)w};
    glMaterialfv((GLenum)face, (GLenum)pname, params);
}

#include <GL/glew.h>
#include "glew_wrappers.h"

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

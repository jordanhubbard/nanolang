#ifndef GLEW_WRAPPERS_H
#define GLEW_WRAPPERS_H

#include <stdint.h>

// Wrapper for glLightfv that accepts individual float parameters
void nl_glLightfv4(int64_t light, int64_t pname, double x, double y, double z, double w);

// Wrapper for glMaterialfv that accepts individual float parameters
void nl_glMaterialfv4(int64_t face, int64_t pname, double x, double y, double z, double w);

#endif // GLEW_WRAPPERS_H

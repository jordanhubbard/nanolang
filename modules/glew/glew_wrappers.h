#ifndef GLEW_WRAPPERS_H
#define GLEW_WRAPPERS_H

#include <stdint.h>

/* nanolang runtime arrays */
#include "dyn_array.h"

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

/* === Modern OpenGL helpers (pointer-free) === */

/* Shader/program */
int64_t nl_gl3_create_program_from_sources(const char *vertex_src, const char *fragment_src);
void nl_gl3_use_program(int64_t program);
void nl_gl3_delete_program(int64_t program);
int64_t nl_gl3_get_uniform_location(int64_t program, const char *name);
void nl_gl3_uniform1f(int64_t location, double v);
void nl_gl3_uniform2f(int64_t location, double x, double y);
void nl_gl3_uniform1i(int64_t location, int64_t v);

/* Buffers/VAOs */
int64_t nl_gl3_gen_vertex_array(void);
void nl_gl3_bind_vertex_array(int64_t vao);
int64_t nl_gl3_gen_buffer(void);
void nl_gl3_bind_buffer(int64_t target, int64_t buffer);
void nl_gl3_buffer_data_f32(int64_t target, DynArray *data_f64, int64_t usage);
void nl_gl3_buffer_data_u32(int64_t target, DynArray *data_i64, int64_t usage);
void nl_gl3_enable_vertex_attrib_array(int64_t index);
void nl_gl3_vertex_attrib_pointer_f32(int64_t index, int64_t size, int64_t normalized, int64_t stride_bytes, int64_t offset_bytes);
void nl_gl3_vertex_attrib_divisor(int64_t index, int64_t divisor);

/* Drawing */
void nl_gl3_draw_arrays(int64_t mode, int64_t first, int64_t count);
void nl_gl3_draw_arrays_instanced(int64_t mode, int64_t first, int64_t count, int64_t instance_count);

/* Textures */
int64_t nl_gl3_gen_texture(void);
void nl_gl3_bind_texture(int64_t target, int64_t texture);
void nl_gl3_active_texture(int64_t texture_unit);
void nl_gl3_tex_parami(int64_t target, int64_t pname, int64_t param);
void nl_gl3_tex_image_2d_checker_rgba8(int64_t target, int64_t width, int64_t height, int64_t squares);

/* Framebuffers */
int64_t nl_gl3_gen_framebuffer(void);
void nl_gl3_bind_framebuffer(int64_t target, int64_t fbo);
void nl_gl3_framebuffer_texture_2d(int64_t target, int64_t attachment, int64_t textarget, int64_t texture, int64_t level);
int64_t nl_gl3_check_framebuffer_status(int64_t target);

#endif // GLEW_WRAPPERS_H

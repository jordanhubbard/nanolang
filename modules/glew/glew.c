#include <GL/glew.h>
#include "glew_wrappers.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dyn_array.h"

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

/* ============================================================================
 * Modern OpenGL helpers (pointer-free)
 * ============================================================================
 */

static int64_t clamp_i64(int64_t v, int64_t lo, int64_t hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static GLuint gen_single_id(void (*gen_fn)(GLsizei, GLuint*)) {
    GLuint id = 0;
    gen_fn(1, &id);
    return id;
}

static char g_gl3_last_log[8192];

static void set_gl3_log(const char *s) {
    if (!s) s = "";
    snprintf(g_gl3_last_log, sizeof(g_gl3_last_log), "%s", s);
}

static GLuint compile_shader(GLenum type, const char *src) {
    if (!src) src = "";
    GLuint sh = glCreateShader(type);
    const GLchar *csrc = (const GLchar*)src;
    glShaderSource(sh, 1, &csrc, NULL);
    glCompileShader(sh);

    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetShaderiv(sh, GL_INFO_LOG_LENGTH, &len);
        len = (GLint)clamp_i64((int64_t)len, 1, (int64_t)sizeof(g_gl3_last_log) - 1);
        glGetShaderInfoLog(sh, len, NULL, g_gl3_last_log);
        glDeleteShader(sh);
        return 0;
    }
    set_gl3_log("");
    return sh;
}

int64_t nl_gl3_create_program_from_sources(const char *vertex_src, const char *fragment_src) {
    GLuint vs = compile_shader(GL_VERTEX_SHADER, vertex_src);
    if (!vs) return 0;
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fragment_src);
    if (!fs) {
        glDeleteShader(vs);
        return 0;
    }

    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);

    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    glDeleteShader(vs);
    glDeleteShader(fs);
    if (!ok) {
        GLint len = 0;
        glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
        len = (GLint)clamp_i64((int64_t)len, 1, (int64_t)sizeof(g_gl3_last_log) - 1);
        glGetProgramInfoLog(prog, len, NULL, g_gl3_last_log);
        glDeleteProgram(prog);
        return 0;
    }
    set_gl3_log("");
    return (int64_t)prog;
}

void nl_gl3_use_program(int64_t program) {
    glUseProgram((GLuint)program);
}

void nl_gl3_delete_program(int64_t program) {
    if (program <= 0) return;
    glDeleteProgram((GLuint)program);
}

int64_t nl_gl3_get_uniform_location(int64_t program, const char *name) {
    if (!name) name = "";
    GLint loc = glGetUniformLocation((GLuint)program, name);
    return (int64_t)loc;
}

void nl_gl3_uniform1f(int64_t location, double v) {
    glUniform1f((GLint)location, (GLfloat)v);
}

void nl_gl3_uniform2f(int64_t location, double x, double y) {
    glUniform2f((GLint)location, (GLfloat)x, (GLfloat)y);
}

void nl_gl3_uniform1i(int64_t location, int64_t v) {
    glUniform1i((GLint)location, (GLint)v);
}

int64_t nl_gl3_gen_vertex_array(void) {
    return (int64_t)gen_single_id(glGenVertexArrays);
}

void nl_gl3_bind_vertex_array(int64_t vao) {
    glBindVertexArray((GLuint)vao);
}

int64_t nl_gl3_gen_buffer(void) {
    return (int64_t)gen_single_id(glGenBuffers);
}

void nl_gl3_bind_buffer(int64_t target, int64_t buffer) {
    glBindBuffer((GLenum)target, (GLuint)buffer);
}

void nl_gl3_buffer_data_f32(int64_t target, DynArray *data_f64, int64_t usage) {
    if (!data_f64 || data_f64->length <= 0) {
        glBufferData((GLenum)target, 0, NULL, (GLenum)usage);
        return;
    }

    /* nanolang float arrays are stored as doubles; convert to float32 for GL */
    int64_t n = data_f64->length;
    const double *src = (const double*)data_f64->data;
    float *tmp = (float*)malloc((size_t)n * sizeof(float));
    if (!tmp) {
        glBufferData((GLenum)target, 0, NULL, (GLenum)usage);
        return;
    }
    for (int64_t i = 0; i < n; i++) {
        tmp[i] = (float)src[i];
    }
    glBufferData((GLenum)target, (GLsizeiptr)((size_t)n * sizeof(float)), tmp, (GLenum)usage);
    free(tmp);
}

void nl_gl3_buffer_data_u32(int64_t target, DynArray *data_i64, int64_t usage) {
    if (!data_i64 || data_i64->length <= 0) {
        glBufferData((GLenum)target, 0, NULL, (GLenum)usage);
        return;
    }
    int64_t n = data_i64->length;
    const int64_t *src = (const int64_t*)data_i64->data;
    uint32_t *tmp = (uint32_t*)malloc((size_t)n * sizeof(uint32_t));
    if (!tmp) {
        glBufferData((GLenum)target, 0, NULL, (GLenum)usage);
        return;
    }
    for (int64_t i = 0; i < n; i++) {
        tmp[i] = (uint32_t)src[i];
    }
    glBufferData((GLenum)target, (GLsizeiptr)((size_t)n * sizeof(uint32_t)), tmp, (GLenum)usage);
    free(tmp);
}

void nl_gl3_enable_vertex_attrib_array(int64_t index) {
    glEnableVertexAttribArray((GLuint)index);
}

void nl_gl3_vertex_attrib_pointer_f32(int64_t index, int64_t size, int64_t normalized, int64_t stride_bytes, int64_t offset_bytes) {
    glVertexAttribPointer((GLuint)index,
                          (GLint)size,
                          GL_FLOAT,
                          normalized ? GL_TRUE : GL_FALSE,
                          (GLsizei)stride_bytes,
                          (const void*)(uintptr_t)offset_bytes);
}

void nl_gl3_vertex_attrib_divisor(int64_t index, int64_t divisor) {
    glVertexAttribDivisor((GLuint)index, (GLuint)divisor);
}

void nl_gl3_draw_arrays(int64_t mode, int64_t first, int64_t count) {
    glDrawArrays((GLenum)mode, (GLint)first, (GLsizei)count);
}

void nl_gl3_draw_arrays_instanced(int64_t mode, int64_t first, int64_t count, int64_t instance_count) {
    glDrawArraysInstanced((GLenum)mode, (GLint)first, (GLsizei)count, (GLsizei)instance_count);
}

int64_t nl_gl3_gen_texture(void) {
    return (int64_t)gen_single_id(glGenTextures);
}

void nl_gl3_bind_texture(int64_t target, int64_t texture) {
    glBindTexture((GLenum)target, (GLuint)texture);
}

void nl_gl3_active_texture(int64_t texture_unit) {
    glActiveTexture((GLenum)texture_unit);
}

void nl_gl3_tex_parami(int64_t target, int64_t pname, int64_t param) {
    glTexParameteri((GLenum)target, (GLenum)pname, (GLint)param);
}

void nl_gl3_tex_image_2d_checker_rgba8(int64_t target, int64_t width, int64_t height, int64_t squares) {
    int w = (int)width;
    int h = (int)height;
    int sq = (int)clamp_i64(squares, 1, 512);
    if (w <= 0 || h <= 0) return;

    size_t total = (size_t)w * (size_t)h * 4;
    unsigned char *pixels = (unsigned char*)malloc(total);
    if (!pixels) return;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int cx = (x / sq) & 1;
            int cy = (y / sq) & 1;
            int on = cx ^ cy;
            unsigned char r = on ? 240 : 20;
            unsigned char g = on ? 240 : 20;
            unsigned char b = on ? 240 : 30;
            unsigned char a = 255;
            size_t idx = ((size_t)y * (size_t)w + (size_t)x) * 4;
            pixels[idx + 0] = r;
            pixels[idx + 1] = g;
            pixels[idx + 2] = b;
            pixels[idx + 3] = a;
        }
    }

    glTexImage2D((GLenum)target, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    free(pixels);
}

int64_t nl_gl3_gen_framebuffer(void) {
    return (int64_t)gen_single_id(glGenFramebuffers);
}

void nl_gl3_bind_framebuffer(int64_t target, int64_t fbo) {
    glBindFramebuffer((GLenum)target, (GLuint)fbo);
}

void nl_gl3_framebuffer_texture_2d(int64_t target, int64_t attachment, int64_t textarget, int64_t texture, int64_t level) {
    glFramebufferTexture2D((GLenum)target, (GLenum)attachment, (GLenum)textarget, (GLuint)texture, (GLint)level);
}

int64_t nl_gl3_check_framebuffer_status(int64_t target) {
    return (int64_t)glCheckFramebufferStatus((GLenum)target);
}

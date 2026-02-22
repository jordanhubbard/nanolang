/* pybridge_warp.c – C adapter that forwards fluid-simulation requests to the
 * pybridge subprocess via dynamic linking.  Mirrors the pattern used by
 * pybridge_matplotlib.c.
 *
 * The pybridge subprocess (tools/nanolang_pybridge.py) handles the actual
 * computation using NVIDIA Warp (GPU) when available, or NumPy on the CPU
 * as a fallback.
 */

#include "pybridge.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int64_t    (*nl_pybridge_init_fn)(const char*, int64_t);
typedef const char*(*nl_pybridge_request_fn)(const char*, const char*);
typedef void       (*nl_pybridge_shutdown_fn)(void);

static nl_pybridge_init_fn     g_nl_pybridge_init     = NULL;
static nl_pybridge_request_fn  g_nl_pybridge_request  = NULL;
static nl_pybridge_shutdown_fn g_nl_pybridge_shutdown = NULL;
static int  g_bridge_initialized = 0;
static char g_last_error[2048]   = "";
static char g_last_backend[64]   = "";

/* ── load the shared library once ─────────────────────────────────────────── */

static void pybridge_warp_ensure_loaded(void) {
    static int attempted = 0;
    if (attempted) return;
    attempted = 1;

#ifdef __APPLE__
    void *h = dlopen("modules/pybridge/.build/libpybridge.dylib", RTLD_LAZY | RTLD_GLOBAL);
    if (!h) h = dlopen("modules/pybridge/.build/libpybridge.so", RTLD_LAZY | RTLD_GLOBAL);
#else
    void *h = dlopen("modules/pybridge/.build/libpybridge.so", RTLD_LAZY | RTLD_GLOBAL);
#endif
    if (!h) return;

    g_nl_pybridge_init     = (nl_pybridge_init_fn)    dlsym(h, "nl_pybridge_init");
    g_nl_pybridge_request  = (nl_pybridge_request_fn) dlsym(h, "nl_pybridge_request");
    g_nl_pybridge_shutdown = (nl_pybridge_shutdown_fn)dlsym(h, "nl_pybridge_shutdown");
}

/* ── ensure the Python subprocess is running ──────────────────────────────── */

static int warp_ensure_bridge(void) {
    pybridge_warp_ensure_loaded();
    if (!g_nl_pybridge_init) return 0;
    if (g_bridge_initialized) return 1;
    /* numpy and pillow are always installed; warp is imported optionally in
     * the Python handler if available on the system Python. */
    int64_t ok = g_nl_pybridge_init("[\"numpy\",\"pillow\"]", 0);
    if (ok) g_bridge_initialized = 1;
    return (int)ok;
}

static const char *warp_request(const char *op, const char *params) {
    if (!warp_ensure_bridge()) return "";
    if (!g_nl_pybridge_request) return "";
    return g_nl_pybridge_request(op, params);
}

/* ── error capture ─────────────────────────────────────────────────────────── */

/* Extract the "message" string from an error response and store it in
 * g_last_error.  Example response:
 *   {"v":1,"id":N,"error":{"code":-32602,"message":"CUDA device is unavailable..."}}
 */
static void warp_capture_error(const char *resp) {
    g_last_error[0] = '\0';
    if (!resp) return;
    /* Only interested in error responses. */
    if (!strstr(resp, "\"error\"")) return;
    const char *p = strstr(resp, "\"message\":\"");
    if (!p) return;
    p += 11; /* skip: "message":"  */
    size_t i = 0;
    while (*p && *p != '"' && i < sizeof(g_last_error) - 1) {
        if (p[0] == '\\' && p[1]) {
            /* simple JSON unescape */
            char c = p[1];
            if      (c == 'n') c = '\n';
            else if (c == 't') c = '\t';
            else if (c == '"') c = '"';
            g_last_error[i++] = c;
            p += 2;
        } else {
            g_last_error[i++] = *p++;
        }
    }
    g_last_error[i] = '\0';
}

/* ── simple JSON helpers ───────────────────────────────────────────────────── */

/* Find "key": in a JSON string and return the integer that follows. */
static int64_t json_get_int64(const char *json, const char *key) {
    if (!json || !key) return -1;
    char search[128];
    snprintf(search, sizeof(search), "\"%s\":", key);
    const char *p = strstr(json, search);
    if (!p) return -1;
    p += strlen(search);
    while (*p == ' ' || *p == '\t') p++;
    return (int64_t)atoll(p);
}

static int json_has_ok_true(const char *json) {
    if (!json) return 0;
    return strstr(json, "\"ok\":true") != NULL;
}

/* Find "key": in a JSON string and copy the string value (without quotes).
 * Returns pointer to static buffer; valid until next call. */
static const char *json_get_string(const char *json, const char *key) {
    static char buf[256];
    buf[0] = '\0';
    if (!json || !key) return buf;
    char search[128];
    snprintf(search, sizeof(search), "\"%s\":\"", key);
    const char *p = strstr(json, search);
    if (!p) return buf;
    p += strlen(search);
    size_t i = 0;
    while (*p && *p != '"' && i < sizeof(buf) - 1)
        buf[i++] = *p++;
    buf[i] = '\0';
    return buf;
}

/* ── public API ────────────────────────────────────────────────────────────── */

int64_t warp_fluid_init(int64_t n, const char *device) {
    g_last_error[0] = '\0';
    size_t len = (size_t)snprintf(NULL, 0, "{\"n\":%lld,\"device\":\"%s\"}",
                                  (long long)n, device ? device : "cpu");
    char *params = (char *)malloc(len + 1);
    if (!params) {
        snprintf(g_last_error, sizeof(g_last_error), "Out of memory");
        return -1;
    }
    snprintf(params, len + 1, "{\"n\":%lld,\"device\":\"%s\"}",
             (long long)n, device ? device : "cpu");
    const char *resp = warp_request("warp_fluid_init", params);
    free(params);
    if (!resp || !resp[0]) {
        snprintf(g_last_error, sizeof(g_last_error),
                 "pybridge not available – build it first: "
                 "make -f Makefile.gnu module MODULE=pybridge");
        return -1;
    }
    warp_capture_error(resp);
    int64_t handle = json_get_int64(resp, "handle");
    if (handle >= 0) {
        /* Capture the backend string (e.g. "warp/cuda", "warp/cpu", "numpy-cpu")
         * so callers can query it via warp_fluid_last_backend(). */
        const char *b = json_get_string(resp, "backend");
        snprintf(g_last_backend, sizeof(g_last_backend), "%s", b);
    }
    return handle;
}

void warp_fluid_step(int64_t handle, double dt, int64_t iters) {
    char params[128];
    snprintf(params, sizeof(params),
             "{\"handle\":%lld,\"dt\":%.6f,\"iters\":%lld}",
             (long long)handle, dt, (long long)iters);
    warp_request("warp_fluid_step", params);
}

void warp_fluid_splat(int64_t handle,
                      int64_t cx, int64_t cy,
                      double fx, double fy,
                      int64_t radius, double amount) {
    char params[256];
    snprintf(params, sizeof(params),
             "{\"handle\":%lld,\"cx\":%lld,\"cy\":%lld,"
             "\"fx\":%.4f,\"fy\":%.4f,\"radius\":%lld,\"amount\":%.4f}",
             (long long)handle,
             (long long)cx, (long long)cy,
             fx, fy, (long long)radius, amount);
    warp_request("warp_fluid_splat", params);
}

int64_t warp_fluid_get_image_png(int64_t handle, const char *path) {
    if (!path || !path[0]) return 0;
    size_t len = (size_t)snprintf(NULL, 0,
                                  "{\"handle\":%lld,\"path\":\"%s\"}",
                                  (long long)handle, path);
    char *params = (char *)malloc(len + 1);
    if (!params) return 0;
    snprintf(params, len + 1, "{\"handle\":%lld,\"path\":\"%s\"}",
             (long long)handle, path);
    const char *resp = warp_request("warp_fluid_get_image_png", params);
    free(params);
    return json_has_ok_true(resp) ? 1 : 0;
}

void warp_fluid_reset(int64_t handle) {
    char params[64];
    snprintf(params, sizeof(params), "{\"handle\":%lld}", (long long)handle);
    warp_request("warp_fluid_reset", params);
}

void warp_fluid_destroy(int64_t handle) {
    char params[64];
    snprintf(params, sizeof(params), "{\"handle\":%lld}", (long long)handle);
    warp_request("warp_fluid_destroy", params);
}

void warp_fluid_shutdown(void) {
    pybridge_warp_ensure_loaded();
    if (g_nl_pybridge_shutdown) {
        g_nl_pybridge_shutdown();
        g_bridge_initialized = 0;
    }
}

const char *warp_fluid_backend(void) {
    const char *resp = warp_request("warp_fluid_backend", "{}");
    const char *backend = json_get_string(resp, "backend");
    /* Return pointer to static buffer in json_get_string. */
    return backend;
}

/* Returns the human-readable error message from the last failed operation,
 * or an empty string if the last operation succeeded. */
const char *warp_fluid_last_error(void) {
    return g_last_error;
}

/* Returns the backend string that was selected by the most recent successful
 * warp_fluid_init call (e.g. "warp/cuda", "warp/cpu", "numpy-cpu").
 * Returns an empty string if no simulation has been initialised yet. */
const char *warp_fluid_last_backend(void) {
    return g_last_backend;
}

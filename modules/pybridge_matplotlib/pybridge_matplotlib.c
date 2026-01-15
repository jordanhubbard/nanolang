#include "pybridge.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cJSON.h"

typedef int64_t (*nl_pybridge_init_fn)(const char*, int64_t);
typedef const char* (*nl_pybridge_request_fn)(const char*, const char*);
typedef void (*nl_pybridge_shutdown_fn)(void);

static nl_pybridge_init_fn g_nl_pybridge_init = NULL;
static nl_pybridge_request_fn g_nl_pybridge_request = NULL;
static nl_pybridge_shutdown_fn g_nl_pybridge_shutdown = NULL;

static void pybridge_matplotlib_ensure_pybridge_loaded(void) {
    /* Adapter depends on the base pybridge module's C symbols (nl_pybridge_*).
     * Ensure the base module shared library is loaded so those symbols resolve. */
    static int attempted = 0;
    if (attempted) return;
    attempted = 1;

    #ifdef __APPLE__
    void *h = dlopen("modules/pybridge/.build/libpybridge.dylib", RTLD_LAZY | RTLD_GLOBAL);
    if (!h) {
        h = dlopen("modules/pybridge/.build/libpybridge.so", RTLD_LAZY | RTLD_GLOBAL);
    }
    #else
    void *h = dlopen("modules/pybridge/.build/libpybridge.so", RTLD_LAZY | RTLD_GLOBAL);
    #endif

    if (!h) return;

    g_nl_pybridge_init = (nl_pybridge_init_fn)dlsym(h, "nl_pybridge_init");
    g_nl_pybridge_request = (nl_pybridge_request_fn)dlsym(h, "nl_pybridge_request");
    g_nl_pybridge_shutdown = (nl_pybridge_shutdown_fn)dlsym(h, "nl_pybridge_shutdown");
}

int64_t mpl_init(int64_t privileged) {
    pybridge_matplotlib_ensure_pybridge_loaded();
    if (!g_nl_pybridge_init) return 0;
    return g_nl_pybridge_init("[\"matplotlib\"]", privileged);
}

const char* mpl_render_png(const char* spec_json, int64_t inline_base64) {
    pybridge_matplotlib_ensure_pybridge_loaded();
    if (!g_nl_pybridge_request) return "";

    cJSON *params_obj = cJSON_CreateObject();
    if (!params_obj) return "";

    cJSON *spec = NULL;
    if (spec_json && spec_json[0]) {
        spec = cJSON_Parse(spec_json);
    }
    if (!spec) {
        spec = cJSON_CreateNull();
    }
    cJSON_AddItemToObject(params_obj, "spec", spec);
    cJSON_AddBoolToObject(params_obj, "inline", inline_base64 ? 1 : 0);

    char *params = cJSON_PrintUnformatted(params_obj);
    cJSON_Delete(params_obj);
    if (!params) return "";

    const char* response = g_nl_pybridge_request("mpl_render_png", params);
    free(params);
    return response;
}

void mpl_shutdown(void) {
    pybridge_matplotlib_ensure_pybridge_loaded();
    if (!g_nl_pybridge_shutdown) return;
    g_nl_pybridge_shutdown();
}

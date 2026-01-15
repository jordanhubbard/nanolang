#include "pybridge.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int64_t mpl_init(int64_t privileged) {
    return nl_pybridge_init("[\"matplotlib\"]", privileged);
}

const char* mpl_render_png(const char* spec_json, int64_t inline_base64) {
    const char* spec = spec_json && spec_json[0] ? spec_json : "null";
    const char* inline_flag = inline_base64 ? "true" : "false";
    size_t len = strlen(spec) + strlen(inline_flag) + 32;
    char* params = (char*)malloc(len);
    if (!params) return "";
    snprintf(params, len, "{\"spec\":%s,\"inline\":%s}", spec, inline_flag);
    const char* response = nl_pybridge_request("mpl_render_png", params);
    free(params);
    return response;
}

void mpl_shutdown(void) {
    nl_pybridge_shutdown();
}

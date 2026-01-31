#ifndef NANOLANG_PYBRIDGE_H
#define NANOLANG_PYBRIDGE_H

#include <stdint.h>
#include "nanolang.h"

int64_t nl_pybridge_init(const char* requirements_json, int64_t privileged);
const char* nl_pybridge_request(const char* op, const char* params_json);
int64_t nl_pybridge_import(const char* module_name);
const char* nl_pybridge_call(int64_t handle, const char* method, const char* args_json, const char* kwargs_json);
const char* nl_pybridge_get(int64_t handle, const char* attr);
int64_t nl_pybridge_set(int64_t handle, const char* attr, const char* value_json);
int64_t nl_pybridge_release(int64_t handle);
const char* nl_pybridge_ping(void);
const char* nl_pybridge_sysinfo(void);
const char* nl_pybridge_deps(void);
void nl_pybridge_shutdown(void);

#endif

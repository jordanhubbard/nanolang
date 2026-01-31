#pragma once

#include <stdint.h>

const char* nl_uv_version_string(void);
int64_t nl_uv_version(void);
int64_t nl_uv_default_loop(void);
int64_t nl_uv_loop_new(void);
int64_t nl_uv_loop_close(int64_t loop_handle);
int64_t nl_uv_run(int64_t loop_handle, int64_t mode);
void nl_uv_stop(int64_t loop_handle);
int64_t nl_uv_loop_alive(int64_t loop_handle);
int64_t nl_uv_loop_get_active_handles(int64_t loop_handle);
int64_t nl_uv_now(int64_t loop_handle);
void nl_uv_update_time(int64_t loop_handle);
int64_t nl_uv_hrtime(void);
void nl_uv_sleep(int64_t msec);
int64_t nl_uv_backend_timeout(int64_t loop_handle);
const char* nl_uv_strerror(int64_t err);
const char* nl_uv_err_name(int64_t err);
int64_t nl_uv_translate_sys_error(int64_t sys_errno);
int64_t nl_uv_get_total_memory(void);
int64_t nl_uv_get_free_memory(void);
int64_t nl_uv_cpu_count(void);
int64_t nl_uv_loadavg_1min(void);
int64_t nl_uv_os_getpid(void);
int64_t nl_uv_os_getppid(void);
const char* nl_uv_cwd(void);
const char* nl_uv_os_gethostname(void);

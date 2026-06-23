/**
 * uv_helpers.c - Simplified libuv wrapper for nanolang
 * 
 * Provides cross-platform async I/O and event loop (Node.js-style).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <uv.h>

/**
 * Get libuv version as string
 */
const char* nl_uv_version_string(void) {
    return uv_version_string();
}

/**
 * Get libuv version as number
 */
int64_t nl_uv_version(void) {
    return (int64_t)uv_version();
}

/**
 * Create default event loop
 * Returns loop handle or 0 on failure
 */
int64_t nl_uv_default_loop(void) {
    uv_loop_t *loop = uv_default_loop();
    return (int64_t)loop;
}

/**
 * Create a new event loop
 * Returns loop handle or 0 on failure
 */
int64_t nl_uv_loop_new(void) {
    uv_loop_t *loop = malloc(sizeof(uv_loop_t));
    if (!loop) return 0;
    
    int r = uv_loop_init(loop);
    if (r != 0) {
        free(loop);
        return 0;
    }
    
    return (int64_t)loop;
}

/**
 * Close and free a loop
 * Returns 0 on success
 */
int64_t nl_uv_loop_close(int64_t loop_handle) {
    uv_loop_t *loop = (uv_loop_t *)loop_handle;
    if (!loop) return -1;
    
    int r = uv_loop_close(loop);
    if (loop != uv_default_loop()) {
        free(loop);
    }
    
    return (int64_t)r;
}

/**
 * Run the event loop
 * mode: 0 = UV_RUN_DEFAULT, 1 = UV_RUN_ONCE, 2 = UV_RUN_NOWAIT
 * Returns 0 when no active handles/requests, or 1 otherwise
 */
int64_t nl_uv_run(int64_t loop_handle, int64_t mode) {
    uv_loop_t *loop = (uv_loop_t *)loop_handle;
    if (!loop) return -1;
    
    uv_run_mode run_mode = UV_RUN_DEFAULT;
    if (mode == 1) {
        run_mode = UV_RUN_ONCE;
    } else if (mode == 2) {
        run_mode = UV_RUN_NOWAIT;
    }
    
    return (int64_t)uv_run(loop, run_mode);
}

/**
 * Stop the event loop
 */
void nl_uv_stop(int64_t loop_handle) {
    uv_loop_t *loop = (uv_loop_t *)loop_handle;
    if (loop) {
        uv_stop(loop);
    }
}

/**
 * Check if loop is alive (has active handles/requests)
 * Returns 1 if alive, 0 otherwise
 */
int64_t nl_uv_loop_alive(int64_t loop_handle) {
    uv_loop_t *loop = (uv_loop_t *)loop_handle;
    if (!loop) return 0;
    return (int64_t)uv_loop_alive(loop);
}

/**
 * Get number of active handles
 */
int64_t nl_uv_loop_get_active_handles(int64_t loop_handle) {
    uv_loop_t *loop = (uv_loop_t *)loop_handle;
    if (!loop) return 0;
    
    /* Count active handles */
    uint64_t count = 0;
    uv_walk(loop, NULL, &count);
    return (int64_t)count;
}

/**
 * Get current timestamp in milliseconds
 */
int64_t nl_uv_now(int64_t loop_handle) {
    uv_loop_t *loop = (uv_loop_t *)loop_handle;
    if (!loop) return 0;
    return (int64_t)uv_now(loop);
}

/**
 * Update the event loop's concept of "now"
 */
void nl_uv_update_time(int64_t loop_handle) {
    uv_loop_t *loop = (uv_loop_t *)loop_handle;
    if (loop) {
        uv_update_time(loop);
    }
}

/**
 * Get high-resolution time in nanoseconds
 */
int64_t nl_uv_hrtime(void) {
    return (int64_t)uv_hrtime();
}

/**
 * Sleep for milliseconds (blocking)
 */
void nl_uv_sleep(int64_t msec) {
    uv_sleep((unsigned int)msec);
}

/**
 * Get backend timeout (for diagnostics)
 * Returns timeout in milliseconds
 */
int64_t nl_uv_backend_timeout(int64_t loop_handle) {
    uv_loop_t *loop = (uv_loop_t *)loop_handle;
    if (!loop) return -1;
    return (int64_t)uv_backend_timeout(loop);
}

/**
 * Get error message from error code
 */
const char* nl_uv_strerror(int64_t err) {
    return uv_strerror((int)err);
}

/**
 * Get error name from error code
 */
const char* nl_uv_err_name(int64_t err) {
    return uv_err_name((int)err);
}

/**
 * Get system error from libuv error
 */
int64_t nl_uv_translate_sys_error(int64_t sys_errno) {
    return (int64_t)uv_translate_sys_error((int)sys_errno);
}

/**
 * Get total system memory in bytes
 */
int64_t nl_uv_get_total_memory(void) {
    return (int64_t)uv_get_total_memory();
}

/**
 * Get free system memory in bytes
 */
int64_t nl_uv_get_free_memory(void) {
    return (int64_t)uv_get_free_memory();
}

/**
 * Get number of CPUs
 */
int64_t nl_uv_cpu_count(void) {
    uv_cpu_info_t *cpu_infos;
    int count;
    
    int r = uv_cpu_info(&cpu_infos, &count);
    if (r != 0) {
        return 0;
    }
    
    uv_free_cpu_info(cpu_infos, count);
    return (int64_t)count;
}

/**
 * Get load average (1 min)
 * Returns load * 100 as integer
 */
int64_t nl_uv_loadavg_1min(void) {
    double avg[3];
    uv_loadavg(avg);
    return (int64_t)(avg[0] * 100.0);
}

/**
 * Get current process ID
 */
int64_t nl_uv_os_getpid(void) {
    return (int64_t)uv_os_getpid();
}

/**
 * Get parent process ID
 */
int64_t nl_uv_os_getppid(void) {
    return (int64_t)uv_os_getppid();
}

/**
 * Get current working directory
 */
const char* nl_uv_cwd(void) {
    static char buf[2048];
    size_t size = sizeof(buf);
    
    int r = uv_cwd(buf, &size);
    if (r != 0) {
        return "";
    }
    
    return buf;
}

/**
 * Get hostname
 */
const char* nl_uv_os_gethostname(void) {
    static char buf[256];
    size_t size = sizeof(buf);
    
    int r = uv_os_gethostname(buf, &size);
    if (r != 0) {
        return "";
    }
    
    return buf;
}

/* =============================================================================
 * Standard Library: Logging C Backend Header
 * =============================================================================
 * Public API for logging C backend.
 * ============================================================================= */

#ifndef NL_LOG_H
#define NL_LOG_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Configuration */
void nl_log_set_level(int64_t level);
int64_t nl_log_get_level(void);
void nl_log_set_output_mode(int64_t mode);
int64_t nl_log_get_output_mode(void);
void nl_log_set_file(const char *path);

/* Logging */
void nl_log_write(int64_t level, const char *message);

/* Tracing */
int64_t nl_log_trace_enter(const char *fn_name);
void nl_log_trace_exit(int64_t trace_id, const char *fn_name);
void nl_log_trace_event(const char *name, const char *data);

/* Cleanup */
void nl_log_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* NL_LOG_H */

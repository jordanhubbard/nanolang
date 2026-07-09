#pragma once

#include <stdint.h>

int64_t nl_event_base_new(void);
void nl_event_base_free(int64_t base_handle);
int64_t nl_event_base_dispatch(int64_t base_handle);
int64_t nl_event_base_loop(int64_t base_handle, int64_t flags);
int64_t nl_event_base_loopexit(int64_t base_handle, int64_t timeout_secs);
int64_t nl_event_base_loopbreak(int64_t base_handle);
const char* nl_event_base_get_method(int64_t base_handle);
const char* nl_event_get_version(void);
int64_t nl_event_get_version_number(void);
int64_t nl_evtimer_new(int64_t base_handle);
void nl_event_free(int64_t event_handle);
int64_t nl_evtimer_add_timeout(int64_t event_handle, int64_t timeout_secs);
int64_t nl_event_del(int64_t event_handle);
int64_t nl_event_base_get_num_events(int64_t base_handle);
void nl_event_enable_debug_mode(void);
int64_t nl_event_sleep(int64_t base_handle, int64_t seconds);

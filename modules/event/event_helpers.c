/**
 * event_helpers.c - Simplified libevent wrapper for nanolang
 * 
 * Provides event loop and async I/O capabilities.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <event2/event.h>
#include <event2/buffer.h>
#include <event2/bufferevent.h>
#include <event2/listener.h>

/**
 * Create a new event base (event loop)
 * Returns handle (pointer as int64) or 0 on failure
 */
int64_t nl_event_base_new(void) {
    struct event_base *base = event_base_new();
    return (int64_t)base;
}

/**
 * Free an event base
 */
void nl_event_base_free(int64_t base_handle) {
    struct event_base *base = (struct event_base *)base_handle;
    if (base) {
        event_base_free(base);
    }
}

/**
 * Run the event loop
 * Returns 0 on success, -1 on error, 1 if no events registered
 */
int64_t nl_event_base_dispatch(int64_t base_handle) {
    struct event_base *base = (struct event_base *)base_handle;
    if (!base) return -1;
    return (int64_t)event_base_dispatch(base);
}

/**
 * Run event loop with flags (0 = block until event, 1 = non-blocking, 2 = run once)
 * Returns 0 on success, -1 on error
 */
int64_t nl_event_base_loop(int64_t base_handle, int64_t flags) {
    struct event_base *base = (struct event_base *)base_handle;
    if (!base) return -1;
    
    int ev_flags = EVLOOP_ONCE;
    if (flags == 0) {
        ev_flags = 0; /* block */
    } else if (flags == 1) {
        ev_flags = EVLOOP_NONBLOCK;
    }
    
    return (int64_t)event_base_loop(base, ev_flags);
}

/**
 * Exit the event loop after current event
 * Returns 0 on success
 */
int64_t nl_event_base_loopexit(int64_t base_handle, int64_t timeout_secs) {
    struct event_base *base = (struct event_base *)base_handle;
    if (!base) return -1;
    
    if (timeout_secs <= 0) {
        return (int64_t)event_base_loopexit(base, NULL);
    }
    
    struct timeval tv;
    tv.tv_sec = timeout_secs;
    tv.tv_usec = 0;
    return (int64_t)event_base_loopexit(base, &tv);
}

/**
 * Break out of event loop immediately
 * Returns 0 on success
 */
int64_t nl_event_base_loopbreak(int64_t base_handle) {
    struct event_base *base = (struct event_base *)base_handle;
    if (!base) return -1;
    return (int64_t)event_base_loopbreak(base);
}

/**
 * Get the backend method name (epoll, kqueue, select, etc.)
 */
const char* nl_event_base_get_method(int64_t base_handle) {
    struct event_base *base = (struct event_base *)base_handle;
    if (!base) return "unknown";
    return event_base_get_method(base);
}

/**
 * Get libevent version string
 */
const char* nl_event_get_version(void) {
    return event_get_version();
}

/**
 * Get libevent version number
 */
int64_t nl_event_get_version_number(void) {
    return (int64_t)event_get_version_number();
}

/**
 * Create a new timer event
 * Returns event handle or 0 on failure
 * Note: Callback must be registered via C code or use evtimer_add
 */
int64_t nl_evtimer_new(int64_t base_handle) {
    struct event_base *base = (struct event_base *)base_handle;
    if (!base) return 0;
    
    /* For now, we create a persistent timer that needs to be configured */
    struct event *ev = evtimer_new(base, NULL, NULL);
    return (int64_t)ev;
}

/**
 * Free an event
 */
void nl_event_free(int64_t event_handle) {
    struct event *ev = (struct event *)event_handle;
    if (ev) {
        event_free(ev);
    }
}

/**
 * Add a timeout event (simplified one-shot timer)
 * Returns 0 on success
 * Note: This is a simplified version; full callback support needs more infrastructure
 */
int64_t nl_evtimer_add_timeout(int64_t event_handle, int64_t timeout_secs) {
    struct event *ev = (struct event *)event_handle;
    if (!ev) return -1;
    
    struct timeval tv;
    tv.tv_sec = timeout_secs;
    tv.tv_usec = 0;
    return (int64_t)evtimer_add(ev, &tv);
}

/**
 * Delete/remove an event from the event loop
 * Returns 0 on success
 */
int64_t nl_event_del(int64_t event_handle) {
    struct event *ev = (struct event *)event_handle;
    if (!ev) return -1;
    return (int64_t)event_del(ev);
}

/**
 * Get number of active events
 */
int64_t nl_event_base_get_num_events(int64_t base_handle) {
    struct event_base *base = (struct event_base *)base_handle;
    if (!base) return 0;
    return (int64_t)event_base_get_num_events(base, EVENT_BASE_COUNT_ACTIVE);
}

/**
 * Enable debug mode
 */
void nl_event_enable_debug_mode(void) {
    event_enable_debug_mode();
}

/**
 * Simple sleep using libevent (alternative to blocking sleep)
 * Returns 0 on success
 */
int64_t nl_event_sleep(int64_t base_handle, int64_t seconds) {
    struct event_base *base = (struct event_base *)base_handle;
    if (!base) return -1;
    
    struct timeval tv;
    tv.tv_sec = seconds;
    tv.tv_usec = 0;
    
    return (int64_t)event_base_loopexit(base, &tv);
}

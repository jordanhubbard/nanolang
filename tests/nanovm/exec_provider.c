/* I keep process-local state so inherited-image assumptions are observable. */
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
static int64_t counter;
int64_t exec_add(int64_t amount) { counter += amount; return counter; }
int64_t exec_text(const char *text) { counter += (int64_t)strlen(text); return counter; }
int64_t exec_fd_closed(int64_t fd) { errno = 0; return fcntl((int)fd, F_GETFD) == -1 && errno == EBADF; }

#include "runtime/nano_callback.h"
int64_t exec_retained_callback(NanoCallbackV1 *callback) {
    callback->retain(callback);
    NanoCallbackValue out = {0};
    NanoCallbackStatus status = callback->invoke(callback, NULL, 0, &out);
    callback->release(callback);
    return status == NANO_CALLBACK_OK && out.tag == NANO_CALLBACK_INT ? out.as.integer : -1;
}

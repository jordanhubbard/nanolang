// Timing helper for nanolang pi calculator
// Provides microsecond-resolution timing using gettimeofday()

#include <sys/time.h>
#include <stdint.h>

// Returns current time in microseconds since epoch
int64_t gettimeofday_wrapper(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((int64_t)tv.tv_sec * 1000000LL) + (int64_t)tv.tv_usec;
}

/* I capture diagnostic module identity without retaining the OpenCL loader. */
#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <unistd.h>

extern int __real_dlclose(void *handle);

static void maps_write_all(const char *bytes, size_t count) {
    while (count) {
        ssize_t written = write(STDERR_FILENO, bytes, count);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) _exit(92);
        bytes += (size_t)written;
        count -= (size_t)written;
    }
}

int __wrap_dlclose(void *handle) {
    int original_errno = errno;
    int fd = open("/proc/self/maps", O_RDONLY);
    if (fd < 0) _exit(92);
    static const char begin[] = "OPENCL DIAGNOSTIC MAPS BEFORE DLCLOSE BEGIN\n";
    static const char end[] = "OPENCL DIAGNOSTIC MAPS BEFORE DLCLOSE END\n";
    maps_write_all(begin, sizeof begin - 1);
    char bytes[4096];
    for (;;) {
        ssize_t count = read(fd, bytes, sizeof bytes);
        if (count < 0 && errno == EINTR) continue;
        if (count < 0) _exit(92);
        if (count == 0) break;
        maps_write_all(bytes, (size_t)count);
    }
    if (close(fd) != 0) _exit(92);
    maps_write_all(end, sizeof end - 1);
    errno = original_errno;
    return __real_dlclose(handle);
}

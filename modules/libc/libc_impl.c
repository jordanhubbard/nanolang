#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int64_t nl_libc_printf(const char *format) {
    if (!format) {
        return 0;
    }
    return (int64_t)printf("%s", format);
}

int64_t nl_libc_puts(const char *s) {
    if (!s) {
        return (int64_t)puts("");
    }
    return (int64_t)puts(s);
}

void nl_libc_exit(int64_t status) {
    exit((int)status);
}

int64_t nl_libc_atoi(const char *s) {
    if (!s) {
        return 0;
    }
    return (int64_t)atoi(s);
}

double nl_libc_atof(const char *s) {
    if (!s) {
        return 0.0;
    }
    return atof(s);
}

int64_t nl_libc_strlen(const char *s) {
    if (!s) {
        return 0;
    }
    return (int64_t)strlen(s);
}

int64_t nl_libc_strcmp(const char *s1, const char *s2) {
    if (!s1 || !s2) {
        if (s1 == s2) {
            return 0;
        }
        return s1 ? 1 : -1;
    }
    return (int64_t)strcmp(s1, s2);
}

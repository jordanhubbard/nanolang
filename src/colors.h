/* colors.h - ANSI color codes for terminal output */

#ifndef COLORS_H
#define COLORS_H

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/* ANSI color codes */
#define COLOR_RESET   "\033[0m"
#define COLOR_BOLD    "\033[1m"
#define COLOR_DIM     "\033[2m"

/* Foreground colors */
#define COLOR_RED     "\033[31m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_BLUE    "\033[34m"
#define COLOR_MAGENTA "\033[35m"
#define COLOR_CYAN    "\033[36m"
#define COLOR_WHITE   "\033[37m"
#define COLOR_GRAY    "\033[90m"

/* Bold colors */
#define COLOR_BOLD_RED     "\033[1;31m"
#define COLOR_BOLD_GREEN   "\033[1;32m"
#define COLOR_BOLD_YELLOW  "\033[1;33m"
#define COLOR_BOLD_BLUE    "\033[1;34m"
#define COLOR_BOLD_MAGENTA "\033[1;35m"
#define COLOR_BOLD_CYAN    "\033[1;36m"

/* Check if we should use colors (is this a TTY?) */
static inline bool colors_enabled(FILE *stream) {
    static int cached = -1;  /* -1 = not checked, 0 = disabled, 1 = enabled */

    if (cached == -1) {
        /* Check if output is to a terminal and NO_COLOR env var is not set */
        const char *no_color = getenv("NO_COLOR");
        cached = (isatty(fileno(stream)) && (!no_color || no_color[0] == '\0')) ? 1 : 0;
    }

    return cached == 1;
}

/* Color wrapper macros - automatically disable colors if not TTY */
#define CFMT_ERROR(str)   (colors_enabled(stderr) ? COLOR_BOLD_RED str COLOR_RESET : str)
#define CFMT_WARNING(str) (colors_enabled(stderr) ? COLOR_BOLD_YELLOW str COLOR_RESET : str)
#define CFMT_INFO(str)    (colors_enabled(stderr) ? COLOR_BOLD_CYAN str COLOR_RESET : str)
#define CFMT_HINT(str)    (colors_enabled(stderr) ? COLOR_CYAN str COLOR_RESET : str)
#define CFMT_SUCCESS(str) (colors_enabled(stderr) ? COLOR_BOLD_GREEN str COLOR_RESET : str)

/* Conditional color output - use these for dynamic strings */
#define CSTART_ERROR   (colors_enabled(stderr) ? COLOR_BOLD_RED : "")
#define CSTART_WARNING (colors_enabled(stderr) ? COLOR_BOLD_YELLOW : "")
#define CSTART_INFO    (colors_enabled(stderr) ? COLOR_BOLD_CYAN : "")
#define CSTART_HINT    (colors_enabled(stderr) ? COLOR_CYAN : "")
#define CSTART_SUCCESS (colors_enabled(stderr) ? COLOR_BOLD_GREEN : "")
#define CSTART_DIM     (colors_enabled(stderr) ? COLOR_DIM : "")
#define CSTART_BOLD    (colors_enabled(stderr) ? COLOR_BOLD : "")
#define CEND           (colors_enabled(stderr) ? COLOR_RESET : "")

#endif /* COLORS_H */

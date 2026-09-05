/*
 * readline_helpers.c - GNU Readline FFI bindings for NanoLang
 * 
 * On macOS, libedit provides readline-compatible API.
 * On Linux, uses GNU readline.
 */

#include "readline_helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/select.h>
#include <unistd.h>

#ifdef __APPLE__
/* macOS uses libedit with readline compatibility */
#include <editline/readline.h>
#else
/* Linux uses GNU readline */
#include <readline/readline.h>
#include <readline/history.h>
#endif

/* Track initialization state */
static int rl_initialized = 0;
static int rl_got_eof = 0;
static char rl_raw_line_buf[4096];

static void strip_line_endings(char *line) {
    size_t len;
    if (!line) return;
    len = strlen(line);
    while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
        line[len - 1] = '\0';
        len--;
    }
}

int64_t rl_hit_eof_wrapper(void) {
    return rl_got_eof ? 1 : 0;
}

/* Core readline function - reads a line with editing and prompt */
const char* rl_readline_wrapper(const char* prompt) {
    char* line;
    char* result;

    rl_got_eof = 0;
    if (!rl_initialized) {
        rl_initialize_wrapper();
    }

    line = readline(prompt ? prompt : "");
    if (!line) {
        rl_got_eof = 1;
        return "";
    }

    /* Return the line - caller should not free this as NanoLang
     * expects GC-managed or static strings. We'll strdup to be safe. */
    result = strdup(line);
    free(line);
    return result ? result : "";
}

const char* rl_raw_getline_wrapper(const char* prompt) {
    rl_got_eof = 0;
    if (prompt && prompt[0] != '\0') {
        fputs(prompt, stdout);
        fflush(stdout);
    }
    if (fgets(rl_raw_line_buf, (int)sizeof(rl_raw_line_buf), stdin) == NULL) {
        rl_got_eof = 1;
        rl_raw_line_buf[0] = '\0';
        return rl_raw_line_buf;
    }
    strip_line_endings(rl_raw_line_buf);
    return rl_raw_line_buf;
}

/* Add a line to the history */
void rl_add_history_wrapper(const char* line) {
    if (line && *line) {
        add_history(line);
    }
}

/* Clear all history entries */
void rl_clear_history_wrapper(void) {
#ifdef __APPLE__
    /* libedit uses clear_history */
    clear_history();
#else
    /* GNU readline uses rl_clear_history or clear_history */
    #ifdef rl_clear_history
    rl_clear_history();
    #else
    clear_history();
    #endif
#endif
}

/* Get the number of history entries */
int64_t rl_history_length_wrapper(void) {
#ifdef __APPLE__
    /* libedit: use history_length global */
    extern int history_length;
    return (int64_t)history_length;
#else
    /* GNU readline has history_length */
    return (int64_t)history_length;
#endif
}

/* Get a history entry by index (0 = oldest) */
const char* rl_history_get_wrapper(int64_t index) {
    HIST_ENTRY* entry = history_get((int)index + 1); /* history_get is 1-indexed */
    if (entry && entry->line) {
        return entry->line;
    }
    return "";
}

/* Initialize readline */
int64_t rl_initialize_wrapper(void) {
    if (!rl_initialized) {
        /* Set up readline */
        rl_initialized = 1;
        
        /* Use standard input/output */
        rl_instream = stdin;
        rl_outstream = stdout;
        
        /* Initialize readline internals */
        rl_initialize();
    }
    return 0;
}

/* Cleanup readline resources */
void rl_cleanup_wrapper(void) {
    if (rl_initialized) {
        /* Clear history to free memory */
        rl_clear_history_wrapper();
        rl_initialized = 0;
    }
}

/* Check if input is available (non-blocking) */
int64_t rl_input_available_wrapper(void) {
    /* Use select() to check if stdin has data */
    fd_set fds;
    struct timeval tv;
    
    FD_ZERO(&fds);
    FD_SET(fileno(stdin), &fds);
    
    tv.tv_sec = 0;
    tv.tv_usec = 0;
    
    return select(fileno(stdin) + 1, &fds, NULL, NULL, &tv) > 0 ? 1 : 0;
}

/* Set the prompt (for next readline call) */
void rl_set_prompt_wrapper(const char* prompt) {
    /* This is used internally - readline() takes prompt as argument */
    (void)prompt;
}

/* Read history from a file */
int64_t rl_read_history_wrapper(const char* filename) {
    if (!filename) return -1;
    return (int64_t)read_history(filename);
}

/* Write history to a file */
int64_t rl_write_history_wrapper(const char* filename) {
    if (!filename) return -1;
    return (int64_t)write_history(filename);
}

/* Stifle history to max entries (keeps last N entries) */
void rl_stifle_history_wrapper(int64_t max_entries) {
    stifle_history((int)max_entries);
}

/* Append history to a file (instead of overwriting) */
int64_t rl_append_history_wrapper(int64_t nelements, const char* filename) {
    if (!filename) return -1;
#ifdef __APPLE__
    /* libedit doesn't have append_history; fall back to write */
    (void)nelements;
    return (int64_t)write_history(filename);
#else
    return (int64_t)append_history((int)nelements, filename);
#endif
}

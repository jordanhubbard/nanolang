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

/* Core readline function - reads a line with editing and prompt */
const char* rl_readline_wrapper(const char* prompt) {
    if (!rl_initialized) {
        rl_initialize_wrapper();
    }
    
    char* line = readline(prompt ? prompt : "");
    if (!line) {
        /* EOF received (Ctrl-D) */
        return "";
    }
    
    /* Return the line - caller should not free this as NanoLang
     * expects GC-managed or static strings. We'll strdup to be safe. */
    char* result = strdup(line);
    free(line);
    return result ? result : "";
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

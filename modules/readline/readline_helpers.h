/*
 * readline_helpers.h - GNU Readline FFI bindings for NanoLang
 * 
 * Provides line editing with history support for interactive programs.
 * On macOS, uses libedit which provides readline-compatible API.
 * On Linux, uses GNU readline.
 */

#ifndef READLINE_HELPERS_H
#define READLINE_HELPERS_H

#include <stdint.h>

/* Core readline function */
const char* rl_readline_wrapper(const char* prompt);

/* History management */
void rl_add_history_wrapper(const char* line);
void rl_clear_history_wrapper(void);
int64_t rl_history_length_wrapper(void);
const char* rl_history_get_wrapper(int64_t index);

/* Readline state */
int64_t rl_initialize_wrapper(void);
void rl_cleanup_wrapper(void);
int64_t rl_input_available_wrapper(void);

/* Prompt control */
void rl_set_prompt_wrapper(const char* prompt);

/* History file I/O */
int64_t rl_read_history_wrapper(const char* filename);
int64_t rl_write_history_wrapper(const char* filename);

#endif /* READLINE_HELPERS_H */

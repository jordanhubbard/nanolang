#ifndef NANO_EVAL_H
#define NANO_EVAL_H

#include <stdint.h>

#define NANO_EVAL_CMD_NONE 0
#define NANO_EVAL_CMD_MESSAGE 1
#define NANO_EVAL_CMD_FIND_FILE 2
#define NANO_EVAL_CMD_SAVE 3
#define NANO_EVAL_CMD_SPLIT 4
#define NANO_EVAL_CMD_OTHER_WINDOW 5

int64_t nano_eval_create(void);
void nano_eval_destroy(int64_t session);
const char *nano_eval_string(int64_t session, const char *source);
const char *nano_eval_error(int64_t session);
void nano_eval_bind_buffer(int64_t session, const char *text, int64_t point);
const char *nano_eval_buffer(int64_t session);
int64_t nano_eval_point(int64_t session);
int64_t nano_eval_cmd_count(int64_t session);
int64_t nano_eval_cmd_kind(int64_t session, int64_t index);
const char *nano_eval_cmd_arg(int64_t session, int64_t index);
void nano_eval_cmd_clear(int64_t session);

int64_t ed_message(const char *text);
int64_t ed_insert(const char *text);
int64_t ed_buffer_string(void);
int64_t ed_point(void);
int64_t ed_goto_char(int64_t pos);
int64_t ed_find_file(const char *path);
int64_t ed_save_buffer(void);
int64_t ed_split_window(void);
int64_t ed_other_window(void);

#endif

/*
 * Forth session runtime.
 *
 * I keep one mutable NvmModule and one persistent VmState for a Forth
 * interpreter session. NanoVM's operand stack is the calling convention for
 * verified NanoISA functions. It is not my Forth data stack.
 *
 * This is a C host runtime, not a NanoLang language feature. There is no
 * src_nano twin.
 */

#ifndef NANOLANG_FORTH_SESSION_H
#define NANOLANG_FORTH_SESSION_H

#include "nanovm/vm.h"

#include <stdbool.h>
#include <stdint.h>

#define FORTH_CELL_BYTES 8
#define FORTH_STACK_CELLS 1024
#define FORTH_RETURN_STACK_CELLS 1024
#define FORTH_FLOAT_STACK_CELLS 256
#define FORTH_CONTROL_STACK_CELLS 64

typedef enum {
    FORTH_CTRL_ORIG = 1,
    FORTH_CTRL_DEST = 2,
    FORTH_CTRL_DO = 3,
    FORTH_CTRL_CASE = 4
} ForthCtrlKind;

typedef struct ForthSession ForthSession;

ForthSession *forth_session_create(void);
void forth_session_destroy(ForthSession *session);

NvmModule *forth_session_module(ForthSession *session);
VmState *forth_session_vm(ForthSession *session);

bool forth_session_rebuild(ForthSession *session);
VmResult forth_session_invoke(ForthSession *session, uint32_t fn_idx,
                              const NanoValue *args, uint16_t arg_count,
                              NanoValue *out_result);

bool forth_data_push(ForthSession *session, int64_t cell);
bool forth_data_pop(ForthSession *session, int64_t *out);
uint32_t forth_data_depth(const ForthSession *session);

bool forth_return_push(ForthSession *session, int64_t cell);
bool forth_return_pop(ForthSession *session, int64_t *out);
uint32_t forth_return_depth(const ForthSession *session);

bool forth_float_push(ForthSession *session, double value);
bool forth_float_pop(ForthSession *session, double *out);
uint32_t forth_float_depth(const ForthSession *session);

bool forth_control_push(ForthSession *session, ForthCtrlKind kind, uint32_t value);
bool forth_control_pop(ForthSession *session, ForthCtrlKind *kind, uint32_t *value);
uint32_t forth_control_depth(const ForthSession *session);

bool forth_allocate(ForthSession *session, uint64_t bytes, uint64_t *addr);
bool forth_free(ForthSession *session, uint64_t addr);
bool forth_store_cell(ForthSession *session, uint64_t addr, int64_t cell);
bool forth_fetch_cell(ForthSession *session, uint64_t addr, int64_t *out);
bool forth_store_byte(ForthSession *session, uint64_t addr, uint8_t byte);
bool forth_fetch_byte(ForthSession *session, uint64_t addr, uint8_t *out);

bool forth_file_open(ForthSession *session, const char *path, const char *mode,
                     uint32_t *fileid);
bool forth_file_close(ForthSession *session, uint32_t fileid);
bool forth_file_is_open(const ForthSession *session, uint32_t fileid);

#endif

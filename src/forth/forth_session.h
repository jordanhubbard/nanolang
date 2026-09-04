/*
 * Forth session runtime.
 *
 * I keep one mutable NvmModule and one persistent VmState for a Forth
 * interpreter session. NanoVM's operand stack is the calling convention for
 * verified NanoISA functions. It is not my Forth data stack. Dictionary
 * headers, word lists, nested input sources, colon compilation, and
 * CATCH/THROW live on the same session.
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
#define FORTH_NAME_MAX 255
#define FORTH_TIB_SIZE 256
#define FORTH_BLOCK_SIZE 1024
#define FORTH_BLOCK_COUNT 32
#define FORTH_ORDER_MAX 8
#define FORTH_WORDLIST_MAX 16
#define FORTH_SOURCE_NEST 16
#define FORTH_COLON_CODE_MAX 8192

typedef uint32_t ForthNt;
typedef uint32_t ForthXt;
typedef uint32_t ForthWid;

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

ForthWid forth_forth_wordlist(const ForthSession *session);
ForthWid forth_get_current(const ForthSession *session);
bool forth_set_current(ForthSession *session, ForthWid wid);
bool forth_wordlist_create(ForthSession *session, ForthWid *wid);
bool forth_get_order(const ForthSession *session, ForthWid *wids, uint32_t cap,
                     uint32_t *count);
bool forth_set_order(ForthSession *session, const ForthWid *wids, uint32_t count);

bool forth_define(ForthSession *session, const char *name, uint32_t name_len,
                  ForthXt xt, bool immediate, bool hidden, ForthNt *nt);
bool forth_reveal(ForthSession *session, ForthNt nt);
bool forth_mark_immediate(ForthSession *session, ForthNt nt);
bool forth_find(const ForthSession *session, const char *name, uint32_t name_len,
                ForthNt *nt, ForthXt *xt, bool *immediate);
bool forth_nt_xt(const ForthSession *session, ForthNt nt, ForthXt *xt);
bool forth_nt_name(const ForthSession *session, ForthNt nt, uint64_t *addr,
                   uint32_t *len);
bool forth_nt_immediate(const ForthSession *session, ForthNt nt);
bool forth_nt_hidden(const ForthSession *session, ForthNt nt);
ForthWid forth_nt_wid(const ForthSession *session, ForthNt nt);
ForthNt forth_latest(const ForthSession *session);

uint64_t forth_to_in_addr(const ForthSession *session);
uint64_t forth_blk_addr(const ForthSession *session);
uint64_t forth_state_addr(const ForthSession *session);
bool forth_source(const ForthSession *session, uint64_t *caddr, uint64_t *u);
int64_t forth_source_id(const ForthSession *session);
uint32_t forth_source_depth(const ForthSession *session);
bool forth_source_load_terminal(ForthSession *session, const uint8_t *bytes,
                                uint32_t len);
bool forth_source_push_evaluate(ForthSession *session, uint64_t caddr, uint64_t u);
bool forth_source_push_file(ForthSession *session, uint32_t fileid);
bool forth_source_push_block(ForthSession *session, uint32_t blk);
bool forth_source_pop(ForthSession *session);
bool forth_refill(ForthSession *session);

bool forth_colon_begin(ForthSession *session, const char *name, uint32_t name_len);
bool forth_colon_literal(ForthSession *session, int64_t cell);
bool forth_colon_call(ForthSession *session, ForthXt xt);
bool forth_colon_recurse(ForthSession *session);
bool forth_colon_if(ForthSession *session);
bool forth_colon_else(ForthSession *session);
bool forth_colon_then(ForthSession *session);
bool forth_colon_cs_begin(ForthSession *session);
bool forth_colon_until(ForthSession *session);
bool forth_colon_again(ForthSession *session);
bool forth_colon_while(ForthSession *session);
bool forth_colon_repeat(ForthSession *session);
bool forth_colon_throw(ForthSession *session);
bool forth_catch(ForthSession *session, ForthXt xt, int64_t *code);
bool forth_colon_finish(ForthSession *session, ForthNt *nt);
bool forth_colon_abort(ForthSession *session);
bool forth_colon_is_open(const ForthSession *session);
ForthXt forth_colon_xt(const ForthSession *session);

#endif

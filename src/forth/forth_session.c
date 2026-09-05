/*
 * Forth session runtime — one module, one VM, Forth stacks, virtual memory.
 */

#include "forth_session.h"
#include "nanoisa/disassembler.h"
#include "nanoisa/verifier.h"
#include "nanovm/vm_ffi.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FORTH_ADDR_INITIAL 65536u
#define FORTH_ADDR_MAX (16u * 1024u * 1024u)
#define FORTH_FILE_SLOTS 32
#define FORTH_REGION_INITIAL 16

typedef struct {
    uint64_t addr;
    uint64_t size;
    bool used;
    bool pinned;
} ForthRegion;

typedef struct {
    ForthCtrlKind kind;
    uint32_t value;
    uint32_t aux;
} ForthCtrlItem;

typedef struct {
    uint32_t generation;
    FILE *fp;
    bool used;
} ForthFile;

typedef struct {
    uint64_t name_addr;
    uint32_t name_len;
    ForthXt xt;
    ForthWid wid;
    bool immediate;
    bool hidden;
    bool compile_only;
    bool used;
    uint8_t host_kind;
    uint8_t name[FORTH_NAME_MAX];
    uint64_t data_addr;
} ForthHeader;

enum {
    FORTH_HOST_NONE = 0,
    FORTH_HOST_COLON,
    FORTH_HOST_SEMI,
    FORTH_HOST_IF,
    FORTH_HOST_ELSE,
    FORTH_HOST_THEN,
    FORTH_HOST_BEGIN,
    FORTH_HOST_UNTIL,
    FORTH_HOST_AGAIN,
    FORTH_HOST_WHILE,
    FORTH_HOST_REPEAT,
    FORTH_HOST_DO,
    FORTH_HOST_LOOP,
    FORTH_HOST_PLUS_LOOP,
    FORTH_HOST_RECURSE,
    FORTH_HOST_LBRACKET,
    FORTH_HOST_RBRACKET,
    FORTH_HOST_LITERAL,
    FORTH_HOST_IMMEDIATE,
    FORTH_HOST_TICK,
    FORTH_HOST_BRACKET_TICK,
    FORTH_HOST_CHAR,
    FORTH_HOST_BRACKET_CHAR,
    FORTH_HOST_CONSTANT,
    FORTH_HOST_VARIABLE,
    FORTH_HOST_ALLOT,
    FORTH_HOST_COMMA,
    FORTH_HOST_ALIGN,
    FORTH_HOST_EXECUTE,
    FORTH_HOST_BACKSLASH,
    FORTH_HOST_PAREN,
    FORTH_HOST_QDO,
    FORTH_HOST_LEAVE,
    FORTH_HOST_EXIT,
    FORTH_HOST_CREATE,
    FORTH_HOST_DOES,
    FORTH_HOST_SOURCE,
    FORTH_HOST_EVALUATE,
    FORTH_HOST_FIND,
    FORTH_HOST_WORD,
    FORTH_HOST_PARSE,
    FORTH_HOST_S_QUOTE,
    FORTH_HOST_DOT_QUOTE,
    FORTH_HOST_EMIT,
    FORTH_HOST_TYPE,
    FORTH_HOST_CR,
    FORTH_HOST_ENVIRONMENT,
    FORTH_HOST_ABORT,
    FORTH_HOST_UM_MOD,
    FORTH_HOST_SM_REM,
    FORTH_HOST_FM_MOD,
    FORTH_HOST_FILL,
    FORTH_HOST_MOVE,
    FORTH_HOST_C_COMMA,
    FORTH_HOST_PICK,
    FORTH_HOST_ROLL,
    FORTH_HOST_LESS_NUM,
    FORTH_HOST_HOLD,
    FORTH_HOST_SIGN,
    FORTH_HOST_HASH,
    FORTH_HOST_HASH_S,
    FORTH_HOST_NUM_END,
    FORTH_HOST_DOT,
    FORTH_HOST_UDOT,
    FORTH_HOST_CATCH,
    FORTH_HOST_BYE,
    FORTH_HOST_TO_BODY,
    FORTH_HOST_TO_NUMBER,
    FORTH_HOST_POSTPONE,
    FORTH_HOST_COMPILE_COMMA,
    FORTH_HOST_ABORT_QUOTE,
    FORTH_HOST_ACCEPT,
    FORTH_HOST_KEY,
    FORTH_HOST_QUIT
};

typedef enum {
    FORTH_SRC_TERMINAL = 0,
    FORTH_SRC_EVALUATE,
    FORTH_SRC_FILE,
    FORTH_SRC_BLOCK
} ForthSourceKind;

typedef struct {
    ForthSourceKind kind;
    uint64_t caddr;
    uint64_t u;
    int64_t source_id;
    uint32_t fileid;
    int64_t blk;
    int64_t saved_to_in;
    int64_t saved_blk;
} ForthSourceFrame;

struct ForthSession {
    NvmModule *module;
    VmState vm;
    uint64_t data_stack_addr;
    uint64_t data_depth_addr;
    uint64_t ret_stack_addr;
    uint64_t ret_depth_addr;
    uint32_t dpush_fn;
    uint32_t dpop_fn;
    uint32_t rpush_fn;
    uint32_t rpop_fn;
    uint32_t throw_fn;
    uint64_t throw_code_addr;
    uint32_t do_enter_fn;
    uint32_t loop_step_fn;
    uint32_t plusloop_step_fn;
    uint32_t unloop_fn;
    uint32_t qdo_enter_fn;
    uint32_t runtime_import;
    double fp[FORTH_FLOAT_STACK_CELLS];
    uint32_t fp_depth;
    ForthCtrlItem control[FORTH_CONTROL_STACK_CELLS];
    uint32_t control_depth;
    ForthRegion *regions;
    uint32_t region_count;
    uint32_t region_cap;
    uint64_t bump;
    ForthFile files[FORTH_FILE_SLOTS];
    ForthHeader *headers;
    uint32_t header_count;
    uint32_t header_cap;
    ForthNt latest;
    ForthWid wordlist_count;
    ForthWid current;
    ForthWid order[FORTH_ORDER_MAX];
    uint32_t order_count;
    uint64_t sysvars;
    uint64_t here_cell_addr;
    uint64_t pad_addr;
    uint64_t word_addr;
    uint64_t hold_addr;
    uint64_t hld_addr;
    char *out_buf;
    uint32_t out_len;
    uint32_t out_cap;
    bool echo_output;
    uint64_t tib_addr;
    uint64_t file_tib_addr;
    uint64_t blocks_addr;
    ForthSourceFrame sources[FORTH_SOURCE_NEST];
    uint32_t source_depth;
    bool colon_open;
    uint32_t colon_fn_idx;
    ForthNt colon_nt;
    uint32_t colon_saved_fn_count;
    uint32_t colon_saved_code_size;
    uint32_t colon_saved_header_count;
    ForthNt colon_saved_latest;
    uint32_t colon_saved_control_depth;
    uint8_t colon_code[FORTH_COLON_CODE_MAX];
    uint32_t colon_code_len;
    bool colon_does_pending;
    uint32_t colon_does_off;
    bool exit_requested;
    bool quit_requested;
};

static bool forth_allocate_ex(ForthSession *session, uint64_t bytes, uint64_t *addr,
                              bool pinned);
static bool forth_install_dpush(ForthSession *session);
static bool forth_install_kernel(ForthSession *session);
static void wrap_patch_rel(uint8_t *code, uint32_t instr_off, uint32_t target_off);
static bool wrap_emit(uint8_t *code, uint32_t *off, uint32_t cap, NanoOpcode op, ...);
static bool forth_interpret_loop(ForthSession *session);
static int forth_run_host(ForthSession *session, uint8_t host, int64_t state);

static ForthSession *g_forth = NULL;

int64_t nl_forth_runtime(int64_t kind);

static uint64_t align_cells(uint64_t bytes) {
    if (bytes > UINT64_MAX - (FORTH_CELL_BYTES - 1)) return UINT64_MAX;
    return (bytes + (FORTH_CELL_BYTES - 1)) & ~(uint64_t)(FORTH_CELL_BYTES - 1);
}

static bool regions_reserve(ForthSession *session, uint32_t extra) {
    uint32_t needed;
    uint32_t cap;
    ForthRegion *grown;

    if (extra > UINT32_MAX - session->region_count) return false;
    needed = session->region_count + extra;
    if (needed <= session->region_cap) return true;
    cap = session->region_cap ? session->region_cap : FORTH_REGION_INITIAL;
    while (cap < needed) {
        if (cap > (UINT32_MAX / 2)) return false;
        cap *= 2;
    }
    grown = realloc(session->regions, (size_t)cap * sizeof(*grown));
    if (!grown) return false;
    session->regions = grown;
    session->region_cap = cap;
    return true;
}

static bool ensure_memory(ForthSession *session, uint64_t needed_end) {
    uint64_t size;

    if (needed_end <= session->vm.memory_size) return true;
    if (needed_end > FORTH_ADDR_MAX) return false;
    size = session->vm.memory_size ? session->vm.memory_size : FORTH_ADDR_INITIAL;
    while (size < needed_end) {
        if (size > FORTH_ADDR_MAX / 2) {
            size = FORTH_ADDR_MAX;
            break;
        }
        size *= 2;
    }
    if (needed_end > size) return false;
    return vm_memory_resize(&session->vm, size);
}

static ForthRegion *region_covering(ForthSession *session, uint64_t addr,
                                    uint64_t size) {
    uint32_t i;

    if (size != 0 && addr > UINT64_MAX - size) return NULL;
    for (i = 0; i < session->region_count; i++) {
        ForthRegion *region = &session->regions[i];
        if (!region->used) continue;
        if (addr < region->addr) continue;
        if ((addr - region->addr) > region->size) continue;
        if (size > region->size - (addr - region->addr)) continue;
        return region;
    }
    return NULL;
}

static int find_region_at(ForthSession *session, uint64_t addr) {
    uint32_t i;
    for (i = 0; i < session->region_count; i++) {
        if (session->regions[i].used && session->regions[i].addr == addr)
            return (int)i;
    }
    return -1;
}

static bool valid_file_mode(const char *mode) {
    size_t i;
    if (!mode || mode[0] == '\0') return false;
    for (i = 0; mode[i] != '\0'; i++) {
        char c = mode[i];
        if (c != 'r' && c != 'w' && c != 'a' && c != 'b' && c != '+')
            return false;
    }
    return true;
}

static bool decode_fileid(const ForthSession *session, uint32_t fileid,
                          uint32_t *slot_out) {
    uint32_t slot;
    uint32_t gen;

    if (!session || fileid == 0) return false;
    slot = (fileid & 0xFFFFu) - 1u;
    gen = fileid >> 16;
    if (slot >= FORTH_FILE_SLOTS) return false;
    if (!session->files[slot].used) return false;
    if ((session->files[slot].generation & 0xFFFFu) != gen) return false;
    if (slot_out) *slot_out = slot;
    return true;
}

static ForthSourceFrame *source_top(ForthSession *session) {
    if (!session || session->source_depth == 0) return NULL;
    return &session->sources[session->source_depth - 1];
}

static const ForthSourceFrame *source_top_const(const ForthSession *session) {
    if (!session || session->source_depth == 0) return NULL;
    return &session->sources[session->source_depth - 1];
}

static bool valid_wid(const ForthSession *session, ForthWid wid) {
    return session && wid >= 1 && wid <= session->wordlist_count;
}

static ForthHeader *header_at(ForthSession *session, ForthNt nt) {
    if (!session || nt == 0 || nt > session->header_count) return NULL;
    if (!session->headers[nt - 1].used) return NULL;
    return &session->headers[nt - 1];
}

static ForthHeader *header_by_xt(ForthSession *session, ForthXt xt) {
    uint32_t i;
    if (!session || !session->headers) return NULL;
    for (i = 0; i < session->header_count; i++) {
        if (session->headers[i].used && session->headers[i].xt == xt)
            return &session->headers[i];
    }
    return NULL;
}

static const ForthHeader *header_at_const(const ForthSession *session, ForthNt nt) {
    if (!session || nt == 0 || nt > session->header_count) return NULL;
    if (!session->headers[nt - 1].used) return NULL;
    return &session->headers[nt - 1];
}

static int ascii_fold(unsigned char c) {
    if (c >= 'A' && c <= 'Z') return (int)c + 32;
    return (int)c;
}

static bool names_equal(const uint8_t *a, uint32_t na, const uint8_t *b, uint32_t nb) {
    uint32_t i;
    if (na != nb) return false;
    for (i = 0; i < na; i++) {
        if (ascii_fold(a[i]) != ascii_fold(b[i])) return false;
    }
    return true;
}

static bool snapshot_source(ForthSession *session) {
    ForthSourceFrame *frame = source_top(session);
    int64_t to_in = 0;
    int64_t blk = 0;
    if (!frame) return false;
    if (!forth_fetch_cell(session, session->sysvars, &to_in)) return false;
    if (!forth_fetch_cell(session, session->sysvars + FORTH_CELL_BYTES, &blk))
        return false;
    frame->saved_to_in = to_in;
    frame->saved_blk = blk;
    return true;
}

static bool restore_source(ForthSession *session) {
    ForthSourceFrame *frame = source_top(session);
    if (!frame) return false;
    if (!forth_store_cell(session, session->sysvars, frame->saved_to_in))
        return false;
    if (!forth_store_cell(session, session->sysvars + FORTH_CELL_BYTES,
                          frame->saved_blk))
        return false;
    return true;
}

static bool forth_session_init_language(ForthSession *session) {
    ForthSourceFrame *base;

    session->wordlist_count = 1;
    session->current = 1;
    session->order[0] = 1;
    session->order_count = 1;

    if (!forth_allocate_ex(session, FORTH_CELL_BYTES * 4, &session->sysvars, true))
        return false;
    if (!forth_allocate_ex(session, FORTH_TIB_SIZE, &session->tib_addr, true))
        return false;
    if (!forth_allocate_ex(session, FORTH_TIB_SIZE, &session->file_tib_addr, true))
        return false;
    if (!forth_allocate_ex(session,
                           (uint64_t)FORTH_BLOCK_SIZE * (uint64_t)FORTH_BLOCK_COUNT,
                           &session->blocks_addr, true))
        return false;

    if (!forth_store_cell(session, session->sysvars, 0)) return false;
    if (!forth_store_cell(session, session->sysvars + FORTH_CELL_BYTES, 0))
        return false;
    if (!forth_store_cell(session, session->sysvars + FORTH_CELL_BYTES * 2, 0))
        return false;
    if (!forth_store_cell(session, session->sysvars + FORTH_CELL_BYTES * 3, 10))
        return false;

    if (!forth_allocate_ex(session, FORTH_CELL_BYTES, &session->here_cell_addr, true))
        return false;
    if (!forth_store_cell(session, session->here_cell_addr, (int64_t)session->bump))
        return false;

    if (!forth_allocate_ex(session, FORTH_CELL_BYTES, &session->data_depth_addr, true))
        return false;
    if (!forth_allocate_ex(session,
                           (uint64_t)FORTH_STACK_CELLS * (uint64_t)FORTH_CELL_BYTES,
                           &session->data_stack_addr, true))
        return false;
    if (!forth_store_cell(session, session->data_depth_addr, 0)) return false;
    if (!forth_allocate_ex(session, FORTH_CELL_BYTES, &session->ret_depth_addr, true))
        return false;
    if (!forth_allocate_ex(session,
                           (uint64_t)FORTH_RETURN_STACK_CELLS * (uint64_t)FORTH_CELL_BYTES,
                           &session->ret_stack_addr, true))
        return false;
    if (!forth_store_cell(session, session->ret_depth_addr, 0)) return false;
    if (!forth_allocate_ex(session, FORTH_CELL_BYTES, &session->throw_code_addr, true))
        return false;
    if (!forth_store_cell(session, session->throw_code_addr, 0)) return false;

    if (!forth_allocate_ex(session, FORTH_PAD_MAX + 1, &session->pad_addr, true))
        return false;
    if (!forth_allocate_ex(session, FORTH_WORD_MAX, &session->word_addr, true))
        return false;
    if (!forth_allocate_ex(session, FORTH_HOLD_MAX, &session->hold_addr, true))
        return false;
    if (!forth_allocate_ex(session, FORTH_CELL_BYTES, &session->hld_addr, true))
        return false;
    if (!forth_store_cell(session, session->hld_addr,
                          (int64_t)(session->hold_addr + FORTH_HOLD_MAX)))
        return false;
    session->echo_output = false;

    session->source_depth = 1;
    base = &session->sources[0];
    memset(base, 0, sizeof(*base));
    base->kind = FORTH_SRC_TERMINAL;
    base->caddr = session->tib_addr;
    base->u = 0;
    base->source_id = 0;
    base->saved_to_in = 0;
    base->saved_blk = 0;
    return true;
}

ForthSession *forth_session_create(void) {
    ForthSession *session = calloc(1, sizeof(*session));
    if (!session) return NULL;

    session->module = nvm_module_new();
    if (!session->module) {
        free(session);
        return NULL;
    }

    vm_init(&session->vm, session->module);
    if (!session->vm.decoded_module_valid) {
        forth_session_destroy(session);
        return NULL;
    }
    if (!vm_memory_resize(&session->vm, FORTH_ADDR_INITIAL)) {
        forth_session_destroy(session);
        return NULL;
    }
    session->bump = FORTH_CELL_BYTES;
    if (!forth_session_init_language(session)) {
        forth_session_destroy(session);
        return NULL;
    }
    if (!forth_install_dpush(session)) {
        forth_session_destroy(session);
        return NULL;
    }
    if (!forth_install_kernel(session)) {
        forth_session_destroy(session);
        return NULL;
    }
    return session;
}

void forth_session_destroy(ForthSession *session) {
    uint32_t i;
    if (!session) return;
    for (i = 0; i < FORTH_FILE_SLOTS; i++) {
        if (session->files[i].fp) fclose(session->files[i].fp);
    }
    vm_ffi_cop_stop(&session->vm);
    vm_destroy(&session->vm);
    nvm_module_free(session->module);
    free(session->regions);
    free(session->headers);
    free(session->out_buf);
    free(session);
}

NvmModule *forth_session_module(ForthSession *session) {
    return session ? session->module : NULL;
}

VmState *forth_session_vm(ForthSession *session) {
    return session ? &session->vm : NULL;
}

bool forth_session_rebuild(ForthSession *session) {
    if (!session) return false;
    return vm_rebuild_module(&session->vm, session->module);
}

VmResult forth_session_invoke(ForthSession *session, uint32_t fn_idx,
                              const NanoValue *args, uint16_t arg_count,
                              NanoValue *out_result) {
    ForthSession *prev;
    VmResult ran;
    if (!session) return VM_ERR_UNDEFINED_FUNCTION;
    prev = g_forth;
    g_forth = session;
    ran = vm_invoke(&session->vm, fn_idx, args, arg_count, out_result);
    g_forth = prev;
    return ran;
}

static VmResult forth_invoke_nested(ForthSession *session, ForthXt xt) {
    ForthSession *prev;
    VmState *vm;
    uint32_t saved_frames;
    uint32_t saved_ip;
    uint32_t saved_fn;
    VmCallFrame copy[VM_MAX_FRAMES];
    VmResult ran;

    if (!session || !session->module) return VM_ERR_UNDEFINED_FUNCTION;
    if (xt >= session->module->function_count) return VM_ERR_UNDEFINED_FUNCTION;
    vm = &session->vm;
    saved_frames = vm->frame_count;
    saved_ip = vm->ip;
    saved_fn = vm->current_fn;
    if (saved_frames > VM_MAX_FRAMES) return VM_ERR_CALL_DEPTH;
    if (saved_frames > 0)
        memcpy(copy, vm->frames, (size_t)saved_frames * sizeof(VmCallFrame));
    vm->frame_count = 0;
    prev = g_forth;
    g_forth = session;
    ran = vm_invoke(vm, xt, NULL, 0, NULL);
    g_forth = prev;
    if (saved_frames > 0)
        memcpy(vm->frames, copy, (size_t)saved_frames * sizeof(VmCallFrame));
    vm->frame_count = saved_frames;
    vm->ip = saved_ip;
    vm->current_fn = saved_fn;
    return ran;
}

static bool forth_emit_char(ForthSession *session, uint8_t ch) {
    char *grown;
    if (!session) return false;
    if (session->out_len + 1 >= session->out_cap) {
        uint32_t cap = session->out_cap ? session->out_cap * 2 : 256;
        if (cap < session->out_len + 2) cap = session->out_len + 2;
        grown = realloc(session->out_buf, cap);
        if (!grown) return false;
        session->out_buf = grown;
        session->out_cap = cap;
    }
    session->out_buf[session->out_len++] = (char)ch;
    session->out_buf[session->out_len] = '\0';
    if (session->echo_output) {
        fputc((int)ch, stdout);
        fflush(stdout);
    }
    return true;
}

const char *forth_output(const ForthSession *session) {
    if (!session || !session->out_buf) return "";
    return session->out_buf;
}

void forth_output_clear(ForthSession *session) {
    if (!session) return;
    session->out_len = 0;
    if (session->out_buf) session->out_buf[0] = '\0';
}

bool forth_data_push(ForthSession *session, int64_t cell) {
    int64_t depth = 0;
    uint64_t addr;

    if (!session) return false;
    if (!forth_fetch_cell(session, session->data_depth_addr, &depth)) return false;
    if (depth < 0 || depth >= (int64_t)FORTH_STACK_CELLS) return false;
    addr = session->data_stack_addr + (uint64_t)depth * (uint64_t)FORTH_CELL_BYTES;
    if (!forth_store_cell(session, addr, cell)) return false;
    return forth_store_cell(session, session->data_depth_addr, depth + 1);
}

bool forth_data_pop(ForthSession *session, int64_t *out) {
    int64_t depth = 0;
    uint64_t addr;

    if (!session || !out) return false;
    if (!forth_fetch_cell(session, session->data_depth_addr, &depth)) return false;
    if (depth <= 0) return false;
    addr = session->data_stack_addr
        + (uint64_t)(depth - 1) * (uint64_t)FORTH_CELL_BYTES;
    if (!forth_fetch_cell(session, addr, out)) return false;
    return forth_store_cell(session, session->data_depth_addr, depth - 1);
}

uint32_t forth_data_depth(const ForthSession *session) {
    uint64_t bits = 0;
    uint32_t i;

    if (!session || !session->vm.memory) return 0;
    if (session->data_depth_addr + FORTH_CELL_BYTES > session->vm.memory_size)
        return 0;
    for (i = 0; i < FORTH_CELL_BYTES; i++) {
        bits |= (uint64_t)session->vm.memory[session->data_depth_addr + i] << (i * 8);
    }
    if ((int64_t)bits < 0 || (int64_t)bits > (int64_t)FORTH_STACK_CELLS) return 0;
    return (uint32_t)bits;
}

bool forth_return_push(ForthSession *session, int64_t cell) {
    int64_t depth = 0;
    uint64_t addr;

    if (!session) return false;
    if (!forth_fetch_cell(session, session->ret_depth_addr, &depth)) return false;
    if (depth < 0 || depth >= (int64_t)FORTH_RETURN_STACK_CELLS) return false;
    addr = session->ret_stack_addr + (uint64_t)depth * (uint64_t)FORTH_CELL_BYTES;
    if (!forth_store_cell(session, addr, cell)) return false;
    return forth_store_cell(session, session->ret_depth_addr, depth + 1);
}

bool forth_return_pop(ForthSession *session, int64_t *out) {
    int64_t depth = 0;
    uint64_t addr;

    if (!session || !out) return false;
    if (!forth_fetch_cell(session, session->ret_depth_addr, &depth)) return false;
    if (depth <= 0) return false;
    addr = session->ret_stack_addr
        + (uint64_t)(depth - 1) * (uint64_t)FORTH_CELL_BYTES;
    if (!forth_fetch_cell(session, addr, out)) return false;
    return forth_store_cell(session, session->ret_depth_addr, depth - 1);
}

uint32_t forth_return_depth(const ForthSession *session) {
    uint64_t bits = 0;
    uint32_t i;

    if (!session || !session->vm.memory) return 0;
    if (session->ret_depth_addr + FORTH_CELL_BYTES > session->vm.memory_size)
        return 0;
    for (i = 0; i < FORTH_CELL_BYTES; i++) {
        bits |= (uint64_t)session->vm.memory[session->ret_depth_addr + i] << (i * 8);
    }
    if ((int64_t)bits < 0 || (int64_t)bits > (int64_t)FORTH_RETURN_STACK_CELLS)
        return 0;
    return (uint32_t)bits;
}

bool forth_float_push(ForthSession *session, double value) {
    if (!session || session->fp_depth >= FORTH_FLOAT_STACK_CELLS) return false;
    session->fp[session->fp_depth++] = value;
    return true;
}

bool forth_float_pop(ForthSession *session, double *out) {
    if (!session || session->fp_depth == 0 || !out) return false;
    *out = session->fp[--session->fp_depth];
    return true;
}

uint32_t forth_float_depth(const ForthSession *session) {
    return session ? session->fp_depth : 0;
}

bool forth_control_push(ForthSession *session, ForthCtrlKind kind, uint32_t value) {
    if (!session || session->control_depth >= FORTH_CONTROL_STACK_CELLS)
        return false;
    session->control[session->control_depth].kind = kind;
    session->control[session->control_depth].value = value;
    session->control[session->control_depth].aux = UINT32_MAX;
    session->control_depth++;
    return true;
}

bool forth_control_pop(ForthSession *session, ForthCtrlKind *kind, uint32_t *value) {
    ForthCtrlItem item;
    if (!session || session->control_depth == 0 || !kind || !value) return false;
    item = session->control[--session->control_depth];
    *kind = item.kind;
    *value = item.value;
    return true;
}

uint32_t forth_control_depth(const ForthSession *session) {
    return session ? session->control_depth : 0;
}

static bool forth_allocate_ex(ForthSession *session, uint64_t bytes, uint64_t *addr,
                              bool pinned) {
    uint64_t size;
    uint32_t i;

    if (!session || !addr || bytes == 0) return false;
    size = align_cells(bytes);
    if (size == 0 || size == UINT64_MAX) return false;

    for (i = 0; i < session->region_count; i++) {
        ForthRegion *region = &session->regions[i];
        if (region->used || region->size < size) continue;
        if (region->size - size >= FORTH_CELL_BYTES) {
            if (!regions_reserve(session, 1)) return false;
            region = &session->regions[i];
            session->regions[session->region_count].addr = region->addr + size;
            session->regions[session->region_count].size = region->size - size;
            session->regions[session->region_count].used = false;
            session->regions[session->region_count].pinned = false;
            session->region_count++;
            region->size = size;
        }
        region->used = true;
        region->pinned = pinned;
        *addr = region->addr;
        return true;
    }

    if (session->bump > UINT64_MAX - size) return false;
    if (!ensure_memory(session, session->bump + size)) return false;
    if (!regions_reserve(session, 1)) return false;
    session->regions[session->region_count].addr = session->bump;
    session->regions[session->region_count].size = size;
    session->regions[session->region_count].used = true;
    session->regions[session->region_count].pinned = pinned;
    session->region_count++;
    *addr = session->bump;
    session->bump += size;
    if (session->here_cell_addr != 0) {
        if (!forth_store_cell(session, session->here_cell_addr, (int64_t)session->bump))
            return false;
    }
    return true;
}

bool forth_allocate(ForthSession *session, uint64_t bytes, uint64_t *addr) {
    return forth_allocate_ex(session, bytes, addr, false);
}

bool forth_free(ForthSession *session, uint64_t addr) {
    int idx;
    if (!session || addr == 0) return false;
    idx = find_region_at(session, addr);
    if (idx < 0) return false;
    if (session->regions[idx].pinned) return false;
    session->regions[idx].used = false;
    return true;
}

bool forth_store_cell(ForthSession *session, uint64_t addr, int64_t cell) {
    uint64_t bits;
    uint32_t i;
    if (!session) return false;
    if ((addr % FORTH_CELL_BYTES) != 0) return false;
    if (!region_covering(session, addr, FORTH_CELL_BYTES)) return false;
    bits = (uint64_t)cell;
    for (i = 0; i < FORTH_CELL_BYTES; i++) {
        session->vm.memory[addr + i] = (uint8_t)(bits >> (i * 8));
    }
    return true;
}

bool forth_fetch_cell(ForthSession *session, uint64_t addr, int64_t *out) {
    uint64_t bits = 0;
    uint32_t i;
    if (!session || !out) return false;
    if ((addr % FORTH_CELL_BYTES) != 0) return false;
    if (!region_covering(session, addr, FORTH_CELL_BYTES)) return false;
    for (i = 0; i < FORTH_CELL_BYTES; i++) {
        bits |= (uint64_t)session->vm.memory[addr + i] << (i * 8);
    }
    *out = (int64_t)bits;
    return true;
}

bool forth_store_byte(ForthSession *session, uint64_t addr, uint8_t byte) {
    if (!session) return false;
    if (!region_covering(session, addr, 1)) return false;
    session->vm.memory[addr] = byte;
    return true;
}

bool forth_fetch_byte(ForthSession *session, uint64_t addr, uint8_t *out) {
    if (!session || !out) return false;
    if (!region_covering(session, addr, 1)) return false;
    *out = session->vm.memory[addr];
    return true;
}

bool forth_file_open(ForthSession *session, const char *path, const char *mode,
                     uint32_t *fileid) {
    uint32_t slot;
    FILE *fp;
    uint32_t gen;

    if (!session || !path || path[0] == '\0' || !fileid) return false;
    if (!valid_file_mode(mode)) return false;

    for (slot = 0; slot < FORTH_FILE_SLOTS; slot++) {
        if (!session->files[slot].used) break;
    }
    if (slot >= FORTH_FILE_SLOTS) return false;

    fp = fopen(path, mode);
    if (!fp) return false;

    gen = (session->files[slot].generation + 1u) & 0xFFFFu;
    if (gen == 0) gen = 1;
    session->files[slot].generation = gen;
    session->files[slot].fp = fp;
    session->files[slot].used = true;
    *fileid = (gen << 16) | (slot + 1u);
    return true;
}

bool forth_file_close(ForthSession *session, uint32_t fileid) {
    uint32_t slot;
    if (!decode_fileid(session, fileid, &slot)) return false;
    fclose(session->files[slot].fp);
    session->files[slot].fp = NULL;
    session->files[slot].used = false;
    return true;
}

bool forth_file_is_open(const ForthSession *session, uint32_t fileid) {
    return decode_fileid(session, fileid, NULL);
}

ForthWid forth_forth_wordlist(const ForthSession *session) {
    return session ? 1u : 0;
}

ForthWid forth_get_current(const ForthSession *session) {
    return session ? session->current : 0;
}

bool forth_set_current(ForthSession *session, ForthWid wid) {
    if (!valid_wid(session, wid)) return false;
    session->current = wid;
    return true;
}

bool forth_wordlist_create(ForthSession *session, ForthWid *wid) {
    if (!session || !wid) return false;
    if (session->wordlist_count >= FORTH_WORDLIST_MAX) return false;
    session->wordlist_count++;
    *wid = session->wordlist_count;
    return true;
}

bool forth_get_order(const ForthSession *session, ForthWid *wids, uint32_t cap,
                     uint32_t *count) {
    uint32_t i;
    if (!session || !wids || !count) return false;
    if (cap < session->order_count) return false;
    for (i = 0; i < session->order_count; i++) wids[i] = session->order[i];
    *count = session->order_count;
    return true;
}

bool forth_set_order(ForthSession *session, const ForthWid *wids, uint32_t count) {
    uint32_t i;
    if (!session) return false;
    if (count > FORTH_ORDER_MAX) return false;
    if (count > 0 && !wids) return false;
    for (i = 0; i < count; i++) {
        if (!valid_wid(session, wids[i])) return false;
    }
    for (i = 0; i < count; i++) session->order[i] = wids[i];
    session->order_count = count;
    return true;
}

bool forth_define(ForthSession *session, const char *name, uint32_t name_len,
                  ForthXt xt, bool immediate, bool hidden, ForthNt *nt) {
    ForthHeader *header;
    uint64_t name_addr = 0;
    uint32_t i;

    if (!session || !name || !nt) return false;
    if (name_len == 0 || name_len > FORTH_NAME_MAX) return false;
    if (!valid_wid(session, session->current)) return false;

    if (session->header_count == session->header_cap) {
        uint32_t cap = session->header_cap ? session->header_cap * 2 : 16;
        ForthHeader *grown = realloc(session->headers, (size_t)cap * sizeof(*grown));
        if (!grown) return false;
        session->headers = grown;
        session->header_cap = cap;
    }

    if (!forth_allocate(session, name_len, &name_addr)) return false;
    for (i = 0; i < name_len; i++) {
        if (!forth_store_byte(session, name_addr + i, (uint8_t)name[i]))
            return false;
    }

    header = &session->headers[session->header_count];
    memset(header, 0, sizeof(*header));
    header->name_addr = name_addr;
    header->name_len = name_len;
    header->xt = xt;
    header->wid = session->current;
    header->immediate = immediate;
    header->hidden = hidden;
    header->used = true;
    memcpy(header->name, name, name_len);
    session->header_count++;
    session->latest = session->header_count;
    *nt = session->latest;
    return true;
}

bool forth_reveal(ForthSession *session, ForthNt nt) {
    ForthHeader *header = header_at(session, nt);
    if (!header) return false;
    header->hidden = false;
    return true;
}

bool forth_mark_immediate(ForthSession *session, ForthNt nt) {
    ForthHeader *header = header_at(session, nt);
    if (!header) return false;
    header->immediate = true;
    return true;
}

bool forth_find(const ForthSession *session, const char *name, uint32_t name_len,
                ForthNt *nt, ForthXt *xt, bool *immediate) {
    uint32_t o, i;

    if (!session || !name) return false;
    if (name_len == 0 || name_len > FORTH_NAME_MAX) return false;

    for (o = 0; o < session->order_count; o++) {
        ForthWid wid = session->order[o];
        for (i = session->header_count; i > 0; i--) {
            const ForthHeader *header = &session->headers[i - 1];
            if (!header->used || header->hidden || header->wid != wid) continue;
            if (!names_equal(header->name, header->name_len,
                             (const uint8_t *)name, name_len))
                continue;
            if (nt) *nt = i;
            if (xt) *xt = header->xt;
            if (immediate) *immediate = header->immediate;
            return true;
        }
    }
    return false;
}

bool forth_nt_xt(const ForthSession *session, ForthNt nt, ForthXt *xt) {
    const ForthHeader *header = header_at_const(session, nt);
    if (!header || !xt) return false;
    *xt = header->xt;
    return true;
}

bool forth_nt_name(const ForthSession *session, ForthNt nt, uint64_t *addr,
                   uint32_t *len) {
    const ForthHeader *header = header_at_const(session, nt);
    if (!header || !addr || !len) return false;
    *addr = header->name_addr;
    *len = header->name_len;
    return true;
}

bool forth_nt_immediate(const ForthSession *session, ForthNt nt) {
    const ForthHeader *header = header_at_const(session, nt);
    return header && header->immediate;
}

bool forth_nt_hidden(const ForthSession *session, ForthNt nt) {
    const ForthHeader *header = header_at_const(session, nt);
    return header && header->hidden;
}

ForthWid forth_nt_wid(const ForthSession *session, ForthNt nt) {
    const ForthHeader *header = header_at_const(session, nt);
    return header ? header->wid : 0;
}

ForthNt forth_latest(const ForthSession *session) {
    return session ? session->latest : 0;
}

uint64_t forth_to_in_addr(const ForthSession *session) {
    return session ? session->sysvars : 0;
}

uint64_t forth_blk_addr(const ForthSession *session) {
    return session ? session->sysvars + FORTH_CELL_BYTES : 0;
}

uint64_t forth_state_addr(const ForthSession *session) {
    return session ? session->sysvars + FORTH_CELL_BYTES * 2 : 0;
}

uint64_t forth_base_addr(const ForthSession *session) {
    return session ? session->sysvars + FORTH_CELL_BYTES * 3 : 0;
}

bool forth_source(const ForthSession *session, uint64_t *caddr, uint64_t *u) {
    const ForthSourceFrame *frame = source_top_const(session);
    if (!frame || !caddr || !u) return false;
    *caddr = frame->caddr;
    *u = frame->u;
    return true;
}

int64_t forth_source_id(const ForthSession *session) {
    const ForthSourceFrame *frame = source_top_const(session);
    return frame ? frame->source_id : 0;
}

uint32_t forth_source_depth(const ForthSession *session) {
    return session ? session->source_depth : 0;
}

bool forth_source_load_terminal(ForthSession *session, const uint8_t *bytes,
                                uint32_t len) {
    ForthSourceFrame *frame = source_top(session);
    uint32_t i;
    if (!session || !frame) return false;
    if (frame->kind != FORTH_SRC_TERMINAL) return false;
    if (len > FORTH_TIB_SIZE) return false;
    if (len > 0 && !bytes) return false;
    for (i = 0; i < len; i++) {
        if (!forth_store_byte(session, session->tib_addr + i, bytes[i]))
            return false;
    }
    frame->caddr = session->tib_addr;
    frame->u = len;
    frame->source_id = 0;
    if (!forth_store_cell(session, session->sysvars, 0)) return false;
    if (!forth_store_cell(session, session->sysvars + FORTH_CELL_BYTES, 0))
        return false;
    return true;
}

bool forth_source_push_evaluate(ForthSession *session, uint64_t caddr, uint64_t u) {
    ForthSourceFrame *child;
    if (!session) return false;
    if (session->source_depth >= FORTH_SOURCE_NEST) return false;
    if (u > 0 && !region_covering(session, caddr, u)) return false;
    if (!snapshot_source(session)) return false;
    child = &session->sources[session->source_depth];
    memset(child, 0, sizeof(*child));
    child->kind = FORTH_SRC_EVALUATE;
    child->caddr = caddr;
    child->u = u;
    child->source_id = -1;
    child->blk = 0;
    session->source_depth++;
    if (!forth_store_cell(session, session->sysvars, 0)) return false;
    if (!forth_store_cell(session, session->sysvars + FORTH_CELL_BYTES, 0))
        return false;
    return true;
}

bool forth_source_push_file(ForthSession *session, uint32_t fileid) {
    ForthSourceFrame *child;
    if (!decode_fileid(session, fileid, NULL)) return false;
    if (session->source_depth >= FORTH_SOURCE_NEST) return false;
    if (!snapshot_source(session)) return false;
    child = &session->sources[session->source_depth];
    memset(child, 0, sizeof(*child));
    child->kind = FORTH_SRC_FILE;
    child->caddr = session->file_tib_addr;
    child->u = 0;
    child->source_id = (int64_t)fileid;
    child->fileid = fileid;
    session->source_depth++;
    if (!forth_store_cell(session, session->sysvars, 0)) return false;
    if (!forth_store_cell(session, session->sysvars + FORTH_CELL_BYTES, 0))
        return false;
    return true;
}

bool forth_source_push_block(ForthSession *session, uint32_t blk) {
    ForthSourceFrame *child;
    if (!session || blk >= FORTH_BLOCK_COUNT) return false;
    if (session->source_depth >= FORTH_SOURCE_NEST) return false;
    if (!snapshot_source(session)) return false;
    child = &session->sources[session->source_depth];
    memset(child, 0, sizeof(*child));
    child->kind = FORTH_SRC_BLOCK;
    child->caddr = session->blocks_addr + (uint64_t)blk * FORTH_BLOCK_SIZE;
    child->u = FORTH_BLOCK_SIZE;
    child->source_id = 0;
    child->blk = (int64_t)blk;
    session->source_depth++;
    if (!forth_store_cell(session, session->sysvars, 0)) return false;
    if (!forth_store_cell(session, session->sysvars + FORTH_CELL_BYTES,
                          (int64_t)blk))
        return false;
    return true;
}

bool forth_source_pop(ForthSession *session) {
    if (!session || session->source_depth <= 1) return false;
    session->source_depth--;
    return restore_source(session);
}

static bool refill_file_line(ForthSession *session, ForthSourceFrame *frame) {
    uint32_t slot;
    FILE *fp;
    uint8_t buf[FORTH_TIB_SIZE];
    uint32_t n = 0;
    bool any = false;
    int c;
    uint32_t i;

    if (!decode_fileid(session, frame->fileid, &slot)) return false;
    fp = session->files[slot].fp;
    if (!fp) return false;

    while ((c = fgetc(fp)) != EOF) {
        any = true;
        if (c == '\n') break;
        if (c == '\r') {
            int next = fgetc(fp);
            if (next != '\n' && next != EOF) ungetc(next, fp);
            break;
        }
        if (n < FORTH_TIB_SIZE) buf[n++] = (uint8_t)c;
    }
    if (!any) return false;
    for (i = 0; i < n; i++) {
        if (!forth_store_byte(session, session->file_tib_addr + i, buf[i]))
            return false;
    }
    frame->caddr = session->file_tib_addr;
    frame->u = n;
    if (!forth_store_cell(session, session->sysvars, 0)) return false;
    return true;
}

bool forth_refill(ForthSession *session) {
    ForthSourceFrame *frame = source_top(session);
    if (!frame) return false;
    if (frame->kind == FORTH_SRC_FILE) return refill_file_line(session, frame);
    return false;
}

static uint32_t forth_emit_into(uint8_t *buf, NanoOpcode op, ...) {
    DecodedInstruction instr;
    const InstructionInfo *info;
    va_list args;
    int i;

    memset(&instr, 0, sizeof(instr));
    instr.opcode = op;
    info = isa_get_info(op);
    if (!info) return 0;
    va_start(args, op);
    for (i = 0; i < info->operand_count; i++) {
        switch (info->operands[i]) {
            case OPERAND_U8:  instr.operands[i].u8  = (uint8_t)va_arg(args, int);      break;
            case OPERAND_U16: instr.operands[i].u16 = (uint16_t)va_arg(args, int);     break;
            case OPERAND_U32: instr.operands[i].u32 = va_arg(args, uint32_t);          break;
            case OPERAND_I32: instr.operands[i].i32 = va_arg(args, int32_t);           break;
            case OPERAND_I64: instr.operands[i].i64 = va_arg(args, int64_t);           break;
            case OPERAND_F64: instr.operands[i].f64 = va_arg(args, double);            break;
            default:
                va_end(args);
                return 0;
        }
    }
    va_end(args);
    return isa_encode(&instr, buf, ISA_MAX_INSTRUCTION_SIZE);
}

static bool forth_install_dpush(ForthSession *session) {
    uint8_t code[512];
    uint32_t off = 0;
    uint32_t n;
    NvmFunctionEntry fn;
    NvmVerifyResult verified;
    NvmModule *mod;
    int64_t depth_addr;
    int64_t stack_base;

    if (!session || !session->module) return false;
    mod = session->module;
    depth_addr = (int64_t)session->data_depth_addr;
    stack_base = (int64_t)session->data_stack_addr;
    memset(&fn, 0, sizeof(fn));

    n = forth_emit_into(code + off, OP_PUSH_I64, depth_addr); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_MEM_LOAD64); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_STORE_LOCAL, 1); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_LOAD_LOCAL, 1); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_PUSH_I64, (int64_t)FORTH_CELL_BYTES); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_I64_MUL); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_PUSH_I64, stack_base); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_I64_ADD); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_STORE_LOCAL, 2); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_LOAD_LOCAL, 2); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_LOAD_LOCAL, 0); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_MEM_STORE64); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_PUSH_I64, depth_addr); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_LOAD_LOCAL, 1); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_PUSH_I64, (int64_t)1); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_I64_ADD); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_MEM_STORE64); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_RET); if (!n) return false; off += n;

    fn.name_idx = nvm_add_string(mod, "nl_forth_dpush", 14);
    fn.arity = 1;
    fn.code_offset = nvm_append_code(mod, code, off);
    fn.code_length = off;
    fn.local_count = 3;
    fn.result_tag = TAG_VOID;
    fn.result_count = 0;
    session->dpush_fn = nvm_add_function(mod, &fn);
    if (session->dpush_fn >= mod->function_count) return false;
    verified = nvm_verify_function(mod, session->dpush_fn);
    if (!verified.ok) return false;

    off = 0;
    memset(&fn, 0, sizeof(fn));
    n = forth_emit_into(code + off, OP_PUSH_I64, depth_addr); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_MEM_LOAD64); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_STORE_LOCAL, 0); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_LOAD_LOCAL, 0); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_PUSH_I64, (int64_t)1); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_I64_SUB); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_STORE_LOCAL, 1); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_PUSH_I64, depth_addr); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_LOAD_LOCAL, 1); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_MEM_STORE64); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_LOAD_LOCAL, 1); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_PUSH_I64, (int64_t)FORTH_CELL_BYTES); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_I64_MUL); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_PUSH_I64, stack_base); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_I64_ADD); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_MEM_LOAD64); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_RET); if (!n) return false; off += n;

    fn.name_idx = nvm_add_string(mod, "nl_forth_dpop", 13);
    fn.arity = 0;
    fn.code_offset = nvm_append_code(mod, code, off);
    fn.code_length = off;
    fn.local_count = 2;
    fn.result_tag = TAG_INT;
    fn.result_count = 1;
    session->dpop_fn = nvm_add_function(mod, &fn);
    if (session->dpop_fn >= mod->function_count) return false;
    verified = nvm_verify_function(mod, session->dpop_fn);
    if (!verified.ok) return false;

    off = 0;
    memset(&fn, 0, sizeof(fn));
    n = forth_emit_into(code + off, OP_LOAD_LOCAL, 0); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_PUSH_I64, (int64_t)0); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_I64_EQ); if (!n) return false; off += n;
    {
        uint32_t jmp_off = off;
        uint32_t done_off;
        int32_t rel;
        n = forth_emit_into(code + off, OP_JMP_TRUE, (int32_t)0); if (!n) return false; off += n;
        n = forth_emit_into(code + off, OP_PUSH_I64, (int64_t)session->throw_code_addr); if (!n) return false; off += n;
        n = forth_emit_into(code + off, OP_LOAD_LOCAL, 0); if (!n) return false; off += n;
        n = forth_emit_into(code + off, OP_MEM_STORE64); if (!n) return false; off += n;
        n = forth_emit_into(code + off, OP_HALT); if (!n) return false; off += n;
        done_off = off;
        rel = (int32_t)done_off - (int32_t)jmp_off;
        code[jmp_off + 1] = (uint8_t)(rel & 0xFF);
        code[jmp_off + 2] = (uint8_t)((rel >> 8) & 0xFF);
        code[jmp_off + 3] = (uint8_t)((rel >> 16) & 0xFF);
        code[jmp_off + 4] = (uint8_t)((rel >> 24) & 0xFF);
    }
    n = forth_emit_into(code + off, OP_RET); if (!n) return false; off += n;

    fn.name_idx = nvm_add_string(mod, "nl_forth_throw", 14);
    fn.arity = 1;
    fn.code_offset = nvm_append_code(mod, code, off);
    fn.code_length = off;
    fn.local_count = 1;
    fn.result_tag = TAG_VOID;
    fn.result_count = 0;
    session->throw_fn = nvm_add_function(mod, &fn);
    if (session->throw_fn >= mod->function_count) return false;
    verified = nvm_verify_function(mod, session->throw_fn);
    if (!verified.ok) return false;

    depth_addr = (int64_t)session->ret_depth_addr;
    stack_base = (int64_t)session->ret_stack_addr;
    off = 0;
    memset(&fn, 0, sizeof(fn));
    n = forth_emit_into(code + off, OP_PUSH_I64, depth_addr); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_MEM_LOAD64); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_STORE_LOCAL, 1); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_LOAD_LOCAL, 1); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_PUSH_I64, (int64_t)FORTH_CELL_BYTES); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_I64_MUL); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_PUSH_I64, stack_base); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_I64_ADD); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_STORE_LOCAL, 2); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_LOAD_LOCAL, 2); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_LOAD_LOCAL, 0); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_MEM_STORE64); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_PUSH_I64, depth_addr); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_LOAD_LOCAL, 1); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_PUSH_I64, (int64_t)1); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_I64_ADD); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_MEM_STORE64); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_RET); if (!n) return false; off += n;
    fn.name_idx = nvm_add_string(mod, "nl_forth_rpush", 14);
    fn.arity = 1;
    fn.code_offset = nvm_append_code(mod, code, off);
    fn.code_length = off;
    fn.local_count = 3;
    fn.result_tag = TAG_VOID;
    fn.result_count = 0;
    session->rpush_fn = nvm_add_function(mod, &fn);
    if (session->rpush_fn >= mod->function_count) return false;
    verified = nvm_verify_function(mod, session->rpush_fn);
    if (!verified.ok) return false;

    off = 0;
    memset(&fn, 0, sizeof(fn));
    n = forth_emit_into(code + off, OP_PUSH_I64, depth_addr); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_MEM_LOAD64); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_STORE_LOCAL, 0); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_LOAD_LOCAL, 0); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_PUSH_I64, (int64_t)1); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_I64_SUB); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_STORE_LOCAL, 1); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_PUSH_I64, depth_addr); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_LOAD_LOCAL, 1); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_MEM_STORE64); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_LOAD_LOCAL, 1); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_PUSH_I64, (int64_t)FORTH_CELL_BYTES); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_I64_MUL); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_PUSH_I64, stack_base); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_I64_ADD); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_MEM_LOAD64); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_RET); if (!n) return false; off += n;
    fn.name_idx = nvm_add_string(mod, "nl_forth_rpop", 13);
    fn.arity = 0;
    fn.code_offset = nvm_append_code(mod, code, off);
    fn.code_length = off;
    fn.local_count = 2;
    fn.result_tag = TAG_INT;
    fn.result_count = 1;
    session->rpop_fn = nvm_add_function(mod, &fn);
    if (session->rpop_fn >= mod->function_count) return false;
    verified = nvm_verify_function(mod, session->rpop_fn);
    if (!verified.ok) return false;

    off = 0;
    memset(&fn, 0, sizeof(fn));
    n = forth_emit_into(code + off, OP_CALL, session->dpop_fn); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_STORE_LOCAL, 0); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_CALL, session->dpop_fn); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_STORE_LOCAL, 1); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_LOAD_LOCAL, 1); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_CALL, session->rpush_fn); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_LOAD_LOCAL, 0); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_CALL, session->rpush_fn); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_RET); if (!n) return false; off += n;
    fn.name_idx = nvm_add_string(mod, "nl_forth_do_enter", 17);
    fn.arity = 0;
    fn.code_offset = nvm_append_code(mod, code, off);
    fn.code_length = off;
    fn.local_count = 2;
    fn.result_tag = TAG_VOID;
    fn.result_count = 0;
    session->do_enter_fn = nvm_add_function(mod, &fn);
    if (session->do_enter_fn >= mod->function_count) return false;
    verified = nvm_verify_function(mod, session->do_enter_fn);
    if (!verified.ok) return false;

    off = 0;
    memset(&fn, 0, sizeof(fn));
    n = forth_emit_into(code + off, OP_CALL, session->rpop_fn); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_STORE_LOCAL, 0); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_CALL, session->rpop_fn); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_STORE_LOCAL, 1); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_LOAD_LOCAL, 0); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_PUSH_I64, (int64_t)1); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_I64_ADD); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_STORE_LOCAL, 0); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_LOAD_LOCAL, 0); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_LOAD_LOCAL, 1); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_I64_EQ); if (!n) return false; off += n;
    {
        uint32_t jmp_done = off;
        uint32_t cont_off;
        n = forth_emit_into(code + off, OP_JMP_TRUE, (int32_t)0); if (!n) return false; off += n;
        n = forth_emit_into(code + off, OP_LOAD_LOCAL, 1); if (!n) return false; off += n;
        n = forth_emit_into(code + off, OP_CALL, session->rpush_fn); if (!n) return false; off += n;
        n = forth_emit_into(code + off, OP_LOAD_LOCAL, 0); if (!n) return false; off += n;
        n = forth_emit_into(code + off, OP_CALL, session->rpush_fn); if (!n) return false; off += n;
        n = forth_emit_into(code + off, OP_PUSH_I64, (int64_t)-1); if (!n) return false; off += n;
        n = forth_emit_into(code + off, OP_RET); if (!n) return false; off += n;
        cont_off = off;
        wrap_patch_rel(code, jmp_done, cont_off);
        n = forth_emit_into(code + off, OP_PUSH_I64, (int64_t)0); if (!n) return false; off += n;
        n = forth_emit_into(code + off, OP_RET); if (!n) return false; off += n;
    }
    fn.name_idx = nvm_add_string(mod, "nl_forth_loop_step", 18);
    fn.arity = 0;
    fn.code_offset = nvm_append_code(mod, code, off);
    fn.code_length = off;
    fn.local_count = 2;
    fn.result_tag = TAG_INT;
    fn.result_count = 1;
    session->loop_step_fn = nvm_add_function(mod, &fn);
    if (session->loop_step_fn >= mod->function_count) return false;
    verified = nvm_verify_function(mod, session->loop_step_fn);
    if (!verified.ok) return false;

    off = 0;
    memset(&fn, 0, sizeof(fn));
    n = forth_emit_into(code + off, OP_CALL, session->dpop_fn); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_STORE_LOCAL, 2); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_CALL, session->rpop_fn); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_STORE_LOCAL, 0); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_CALL, session->rpop_fn); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_STORE_LOCAL, 1); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_LOAD_LOCAL, 0); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_LOAD_LOCAL, 2); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_I64_ADD); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_STORE_LOCAL, 0); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_LOAD_LOCAL, 2); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_PUSH_I64, (int64_t)0); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_I64_LT_S); if (!n) return false; off += n;
    {
        uint32_t jmp_neg = off;
        uint32_t jmp_pos_done;
        uint32_t jmp_to_done;
        uint32_t jmp_neg_term;
        uint32_t cont;
        uint32_t done;
        n = forth_emit_into(code + off, OP_JMP_TRUE, (int32_t)0); if (!n) return false; off += n;
        n = forth_emit_into(code + off, OP_LOAD_LOCAL, 0); if (!n) return false; off += n;
        n = forth_emit_into(code + off, OP_LOAD_LOCAL, 1); if (!n) return false; off += n;
        n = forth_emit_into(code + off, OP_I64_LT_S); if (!n) return false; off += n;
        jmp_pos_done = off;
        n = forth_emit_into(code + off, OP_JMP_FALSE, (int32_t)0); if (!n) return false; off += n;
        jmp_to_done = off;
        n = forth_emit_into(code + off, OP_JMP, (int32_t)0); if (!n) return false; off += n;
        wrap_patch_rel(code, jmp_neg, off);
        n = forth_emit_into(code + off, OP_LOAD_LOCAL, 0); if (!n) return false; off += n;
        n = forth_emit_into(code + off, OP_LOAD_LOCAL, 1); if (!n) return false; off += n;
        n = forth_emit_into(code + off, OP_I64_LT_S); if (!n) return false; off += n;
        jmp_neg_term = off;
        n = forth_emit_into(code + off, OP_JMP_TRUE, (int32_t)0); if (!n) return false; off += n;
        cont = off;
        wrap_patch_rel(code, jmp_to_done, cont);
        n = forth_emit_into(code + off, OP_LOAD_LOCAL, 1); if (!n) return false; off += n;
        n = forth_emit_into(code + off, OP_CALL, session->rpush_fn); if (!n) return false; off += n;
        n = forth_emit_into(code + off, OP_LOAD_LOCAL, 0); if (!n) return false; off += n;
        n = forth_emit_into(code + off, OP_CALL, session->rpush_fn); if (!n) return false; off += n;
        n = forth_emit_into(code + off, OP_PUSH_I64, (int64_t)-1); if (!n) return false; off += n;
        n = forth_emit_into(code + off, OP_RET); if (!n) return false; off += n;
        done = off;
        wrap_patch_rel(code, jmp_pos_done, done);
        wrap_patch_rel(code, jmp_neg_term, done);
        n = forth_emit_into(code + off, OP_PUSH_I64, (int64_t)0); if (!n) return false; off += n;
        n = forth_emit_into(code + off, OP_RET); if (!n) return false; off += n;
    }
    fn.name_idx = nvm_add_string(mod, "nl_forth_plusloop_step", 22);
    fn.arity = 0;
    fn.code_offset = nvm_append_code(mod, code, off);
    fn.code_length = off;
    fn.local_count = 3;
    fn.result_tag = TAG_INT;
    fn.result_count = 1;
    session->plusloop_step_fn = nvm_add_function(mod, &fn);
    if (session->plusloop_step_fn >= mod->function_count) return false;
    verified = nvm_verify_function(mod, session->plusloop_step_fn);
    if (!verified.ok) return false;

    off = 0;
    memset(&fn, 0, sizeof(fn));
    n = forth_emit_into(code + off, OP_CALL, session->rpop_fn); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_STORE_LOCAL, 0); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_CALL, session->rpop_fn); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_STORE_LOCAL, 0); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_RET); if (!n) return false; off += n;
    fn.name_idx = nvm_add_string(mod, "nl_forth_unloop", 15);
    fn.arity = 0;
    fn.code_offset = nvm_append_code(mod, code, off);
    fn.code_length = off;
    fn.local_count = 1;
    fn.result_tag = TAG_VOID;
    fn.result_count = 0;
    session->unloop_fn = nvm_add_function(mod, &fn);
    if (session->unloop_fn >= mod->function_count) return false;
    verified = nvm_verify_function(mod, session->unloop_fn);
    if (!verified.ok) return false;

    off = 0;
    memset(&fn, 0, sizeof(fn));
    n = forth_emit_into(code + off, OP_CALL, session->dpop_fn); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_STORE_LOCAL, 0); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_CALL, session->dpop_fn); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_STORE_LOCAL, 1); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_LOAD_LOCAL, 0); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_LOAD_LOCAL, 1); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_I64_EQ); if (!n) return false; off += n;
    {
        uint32_t jmp_eq = off;
        n = forth_emit_into(code + off, OP_JMP_TRUE, (int32_t)0); if (!n) return false; off += n;
        n = forth_emit_into(code + off, OP_LOAD_LOCAL, 1); if (!n) return false; off += n;
        n = forth_emit_into(code + off, OP_CALL, session->rpush_fn); if (!n) return false; off += n;
        n = forth_emit_into(code + off, OP_LOAD_LOCAL, 0); if (!n) return false; off += n;
        n = forth_emit_into(code + off, OP_CALL, session->rpush_fn); if (!n) return false; off += n;
        n = forth_emit_into(code + off, OP_PUSH_I64, (int64_t)0); if (!n) return false; off += n;
        n = forth_emit_into(code + off, OP_RET); if (!n) return false; off += n;
        wrap_patch_rel(code, jmp_eq, off);
        n = forth_emit_into(code + off, OP_PUSH_I64, (int64_t)-1); if (!n) return false; off += n;
        n = forth_emit_into(code + off, OP_RET); if (!n) return false; off += n;
    }
    fn.name_idx = nvm_add_string(mod, "nl_forth_qdo_enter", 18);
    fn.arity = 0;
    fn.code_offset = nvm_append_code(mod, code, off);
    fn.code_length = off;
    fn.local_count = 2;
    fn.result_tag = TAG_INT;
    fn.result_count = 1;
    session->qdo_enter_fn = nvm_add_function(mod, &fn);
    if (session->qdo_enter_fn >= mod->function_count) return false;
    verified = nvm_verify_function(mod, session->qdo_enter_fn);
    if (!verified.ok) return false;

    return forth_session_rebuild(session);
}

static bool colon_emit(ForthSession *session, NanoOpcode op, ...) {
    uint8_t buf[ISA_MAX_INSTRUCTION_SIZE];
    DecodedInstruction instr;
    const InstructionInfo *info;
    va_list args;
    uint32_t n;
    int i;

    if (!session || !session->colon_open) return false;
    memset(&instr, 0, sizeof(instr));
    instr.opcode = op;
    info = isa_get_info(op);
    if (!info) return false;
    va_start(args, op);
    for (i = 0; i < info->operand_count; i++) {
        switch (info->operands[i]) {
            case OPERAND_U8:  instr.operands[i].u8  = (uint8_t)va_arg(args, int);      break;
            case OPERAND_U16: instr.operands[i].u16 = (uint16_t)va_arg(args, int);     break;
            case OPERAND_U32: instr.operands[i].u32 = va_arg(args, uint32_t);          break;
            case OPERAND_I32: instr.operands[i].i32 = va_arg(args, int32_t);           break;
            case OPERAND_I64: instr.operands[i].i64 = va_arg(args, int64_t);           break;
            case OPERAND_F64: instr.operands[i].f64 = va_arg(args, double);            break;
            default:
                va_end(args);
                return false;
        }
    }
    va_end(args);
    n = isa_encode(&instr, buf, sizeof(buf));
    if (n == 0) return false;
    if (session->colon_code_len > FORTH_COLON_CODE_MAX - n) return false;
    memcpy(session->colon_code + session->colon_code_len, buf, n);
    session->colon_code_len += n;
    return true;
}

static bool colon_rollback(ForthSession *session) {
    uint32_t i;
    NvmModule *mod;

    if (!session || !session->colon_open) return false;
    mod = session->module;
    for (i = session->colon_saved_header_count; i < session->header_count; i++) {
        if (session->headers[i].used && session->headers[i].name_addr != 0)
            forth_free(session, session->headers[i].name_addr);
        session->headers[i].used = false;
    }
    session->header_count = session->colon_saved_header_count;
    session->latest = session->colon_saved_latest;
    session->control_depth = session->colon_saved_control_depth;
    if (mod) {
        if (session->colon_saved_fn_count <= mod->function_count)
            mod->function_count = session->colon_saved_fn_count;
        if (session->colon_saved_code_size <= mod->code_size)
            mod->code_size = session->colon_saved_code_size;
    }
    session->colon_open = false;
    session->colon_code_len = 0;
    session->colon_fn_idx = 0;
    session->colon_nt = 0;
    return true;
}

bool forth_colon_begin(ForthSession *session, const char *name, uint32_t name_len) {
    uint8_t stub[ISA_MAX_INSTRUCTION_SIZE];
    uint32_t stub_len;
    NvmFunctionEntry fn;
    NvmModule *mod;
    ForthNt nt = 0;
    uint32_t before;

    if (!session || !name || session->colon_open) return false;
    if (name_len == 0 || name_len > FORTH_NAME_MAX) return false;
    mod = session->module;
    if (!mod) return false;

    session->colon_saved_fn_count = mod->function_count;
    session->colon_saved_code_size = mod->code_size;
    session->colon_saved_header_count = session->header_count;
    session->colon_saved_latest = session->latest;
    session->colon_saved_control_depth = session->control_depth;

    stub_len = forth_emit_into(stub, OP_RET);
    if (stub_len == 0) return false;
    memset(&fn, 0, sizeof(fn));
    fn.name_idx = nvm_add_string(mod, name, name_len);
    fn.arity = 0;
    fn.code_offset = nvm_append_code(mod, stub, stub_len);
    fn.code_length = stub_len;
    fn.local_count = 0;
    fn.result_tag = TAG_VOID;
    fn.result_count = 0;
    before = mod->function_count;
    session->colon_fn_idx = nvm_add_function(mod, &fn);
    if (mod->function_count != before + 1) return false;

    session->colon_open = true;
    session->colon_code_len = 0;
    session->colon_does_pending = false;
    session->colon_does_off = 0;
    if (!forth_define(session, name, name_len, (ForthXt)session->colon_fn_idx,
                      false, true, &nt)) {
        colon_rollback(session);
        return false;
    }
    session->colon_nt = nt;
    return true;
}

bool forth_colon_literal(ForthSession *session, int64_t cell) {
    if (!session || !session->colon_open) return false;
    if (!colon_emit(session, OP_PUSH_I64, cell)) return false;
    return colon_emit(session, OP_CALL, session->dpush_fn);
}

bool forth_colon_call(ForthSession *session, ForthXt xt) {
    if (!session || !session->colon_open || !session->module) return false;
    if (xt >= session->module->function_count) return false;
    if (xt == session->colon_fn_idx) return false;
    return colon_emit(session, OP_CALL, xt);
}

bool forth_colon_recurse(ForthSession *session) {
    if (!session || !session->colon_open) return false;
    return colon_emit(session, OP_CALL, session->colon_fn_idx);
}

static bool colon_patch_jump(ForthSession *session, uint32_t instr_off,
                             uint32_t target_off) {
    int32_t rel;
    uint32_t patch_off;

    if (!session) return false;
    if (instr_off + 5 > session->colon_code_len) return false;
    if (session->colon_code[instr_off] != (uint8_t)OP_JMP
            && session->colon_code[instr_off] != (uint8_t)OP_JMP_FALSE
            && session->colon_code[instr_off] != (uint8_t)OP_JMP_TRUE)
        return false;
    rel = (int32_t)target_off - (int32_t)instr_off;
    patch_off = instr_off + 1;
    session->colon_code[patch_off] = (uint8_t)(rel & 0xFF);
    session->colon_code[patch_off + 1] = (uint8_t)((rel >> 8) & 0xFF);
    session->colon_code[patch_off + 2] = (uint8_t)((rel >> 16) & 0xFF);
    session->colon_code[patch_off + 3] = (uint8_t)((rel >> 24) & 0xFF);
    return true;
}

static bool colon_pop_ctrl(ForthSession *session, ForthCtrlKind expected,
                           uint32_t *value) {
    ForthCtrlKind kind = FORTH_CTRL_ORIG;
    uint32_t got = 0;
    if (!forth_control_pop(session, &kind, &got)) return false;
    if (kind != expected) {
        forth_control_push(session, kind, got);
        return false;
    }
    *value = got;
    return true;
}

static bool colon_emit_dpop(ForthSession *session) {
    return colon_emit(session, OP_CALL, session->dpop_fn);
}

bool forth_colon_if(ForthSession *session) {
    uint32_t instr_off;
    if (!session || !session->colon_open) return false;
    if (!colon_emit_dpop(session)) return false;
    instr_off = session->colon_code_len;
    if (!colon_emit(session, OP_JMP_FALSE, (int32_t)0)) return false;
    return forth_control_push(session, FORTH_CTRL_ORIG, instr_off);
}

bool forth_colon_else(ForthSession *session) {
    uint32_t if_off = 0;
    uint32_t skip_off;
    if (!session || !session->colon_open) return false;
    if (!colon_pop_ctrl(session, FORTH_CTRL_ORIG, &if_off)) return false;
    skip_off = session->colon_code_len;
    if (!colon_emit(session, OP_JMP, (int32_t)0)) return false;
    if (!colon_patch_jump(session, if_off, session->colon_code_len)) return false;
    return forth_control_push(session, FORTH_CTRL_ORIG, skip_off);
}

bool forth_colon_then(ForthSession *session) {
    uint32_t orig = 0;
    if (!session || !session->colon_open) return false;
    if (!colon_pop_ctrl(session, FORTH_CTRL_ORIG, &orig)) return false;
    return colon_patch_jump(session, orig, session->colon_code_len);
}

bool forth_colon_cs_begin(ForthSession *session) {
    if (!session || !session->colon_open) return false;
    return forth_control_push(session, FORTH_CTRL_DEST, session->colon_code_len);
}

bool forth_colon_until(ForthSession *session) {
    uint32_t dest = 0;
    uint32_t instr_off;
    int32_t rel;
    if (!session || !session->colon_open) return false;
    if (!colon_pop_ctrl(session, FORTH_CTRL_DEST, &dest)) return false;
    if (!colon_emit_dpop(session)) return false;
    instr_off = session->colon_code_len;
    rel = (int32_t)dest - (int32_t)instr_off;
    return colon_emit(session, OP_JMP_FALSE, rel);
}

bool forth_colon_again(ForthSession *session) {
    uint32_t dest = 0;
    uint32_t instr_off;
    int32_t rel;
    if (!session || !session->colon_open) return false;
    if (!colon_pop_ctrl(session, FORTH_CTRL_DEST, &dest)) return false;
    instr_off = session->colon_code_len;
    rel = (int32_t)dest - (int32_t)instr_off;
    return colon_emit(session, OP_JMP, rel);
}

bool forth_colon_while(ForthSession *session) {
    uint32_t instr_off;
    if (!session || !session->colon_open) return false;
    if (!colon_emit_dpop(session)) return false;
    instr_off = session->colon_code_len;
    if (!colon_emit(session, OP_JMP_FALSE, (int32_t)0)) return false;
    return forth_control_push(session, FORTH_CTRL_ORIG, instr_off);
}

bool forth_colon_repeat(ForthSession *session) {
    uint32_t orig = 0;
    uint32_t dest = 0;
    uint32_t instr_off;
    int32_t rel;
    if (!session || !session->colon_open) return false;
    if (!colon_pop_ctrl(session, FORTH_CTRL_ORIG, &orig)) return false;
    if (!colon_pop_ctrl(session, FORTH_CTRL_DEST, &dest)) return false;
    instr_off = session->colon_code_len;
    rel = (int32_t)dest - (int32_t)instr_off;
    if (!colon_emit(session, OP_JMP, rel)) return false;
    return colon_patch_jump(session, orig, session->colon_code_len);
}

bool forth_colon_do(ForthSession *session) {
    if (!session || !session->colon_open) return false;
    if (!colon_emit(session, OP_CALL, session->do_enter_fn)) return false;
    return forth_control_push(session, FORTH_CTRL_DO, session->colon_code_len);
}

bool forth_colon_qdo(ForthSession *session) {
    uint32_t skip_off;
    if (!session || !session->colon_open) return false;
    if (!colon_emit(session, OP_CALL, session->qdo_enter_fn)) return false;
    skip_off = session->colon_code_len;
    if (!colon_emit(session, OP_JMP_TRUE, (int32_t)0)) return false;
    if (!forth_control_push(session, FORTH_CTRL_QDO, skip_off)) return false;
    return forth_control_push(session, FORTH_CTRL_DO, session->colon_code_len);
}

static int32_t colon_read_i32(ForthSession *session, uint32_t instr_off) {
    uint32_t patch_off = instr_off + 1;
    uint32_t bits;
    bits = (uint32_t)session->colon_code[patch_off]
        | ((uint32_t)session->colon_code[patch_off + 1] << 8)
        | ((uint32_t)session->colon_code[patch_off + 2] << 16)
        | ((uint32_t)session->colon_code[patch_off + 3] << 24);
    return (int32_t)bits;
}

static bool colon_close_loop(ForthSession *session, uint32_t step_fn) {
    ForthCtrlItem item;
    uint32_t dest = 0;
    uint32_t instr_off;
    uint32_t leave;
    int32_t rel;

    if (!session || !session->colon_open) return false;
    if (session->control_depth <= session->colon_saved_control_depth) return false;
    item = session->control[--session->control_depth];
    if (item.kind != FORTH_CTRL_DO) {
        session->control[session->control_depth++] = item;
        return false;
    }
    dest = item.value;
    if (!colon_emit(session, OP_CALL, step_fn)) return false;
    instr_off = session->colon_code_len;
    rel = (int32_t)dest - (int32_t)instr_off;
    if (!colon_emit(session, OP_JMP_TRUE, rel)) return false;
    leave = item.aux;
    while (leave != UINT32_MAX) {
        int32_t link = colon_read_i32(session, leave);
        uint32_t next = (link < 0) ? UINT32_MAX : (uint32_t)link;
        if (!colon_patch_jump(session, leave, session->colon_code_len))
            return false;
        leave = next;
    }
    if (session->control_depth > session->colon_saved_control_depth) {
        ForthCtrlItem top = session->control[session->control_depth - 1];
        if (top.kind == FORTH_CTRL_QDO) {
            session->control_depth--;
            if (!colon_patch_jump(session, top.value, session->colon_code_len))
                return false;
        }
    }
    return true;
}

bool forth_colon_loop(ForthSession *session) {
    return colon_close_loop(session, session->loop_step_fn);
}

bool forth_colon_plus_loop(ForthSession *session) {
    return colon_close_loop(session, session->plusloop_step_fn);
}

bool forth_colon_leave(ForthSession *session) {
    ForthCtrlItem *do_item = NULL;
    uint32_t instr_off;
    int32_t prev;
    uint32_t i;

    if (!session || !session->colon_open) return false;
    if (session->control_depth == 0) return false;
    i = session->control_depth;
    while (i > session->colon_saved_control_depth) {
        i--;
        if (session->control[i].kind == FORTH_CTRL_DO) {
            do_item = &session->control[i];
            break;
        }
    }
    if (!do_item) return false;
    if (!colon_emit(session, OP_CALL, session->unloop_fn)) return false;
    instr_off = session->colon_code_len;
    prev = (do_item->aux == UINT32_MAX) ? (int32_t)-1 : (int32_t)do_item->aux;
    if (!colon_emit(session, OP_JMP, prev)) return false;
    do_item->aux = instr_off;
    return true;
}

bool forth_colon_unloop(ForthSession *session) {
    if (!session || !session->colon_open) return false;
    return colon_emit(session, OP_CALL, session->unloop_fn);
}

bool forth_colon_exit(ForthSession *session) {
    if (!session || !session->colon_open) return false;
    return colon_emit(session, OP_RET);
}

bool forth_colon_finish(ForthSession *session, ForthNt *nt) {
    NvmModule *mod;
    NvmFunctionEntry *fn;
    NvmVerifyResult verified;
    uint32_t code_off;

    if (!session || !nt || !session->colon_open || !session->module) return false;
    if (session->control_depth != session->colon_saved_control_depth) {
        colon_rollback(session);
        return false;
    }
    if (session->colon_does_pending) {
        uint8_t does_code[FORTH_COLON_CODE_MAX];
        uint32_t does_len = session->colon_code_len - session->colon_does_off;
        uint32_t does_off = 0;
        NvmFunctionEntry dfn;
        NvmVerifyResult dver;
        uint32_t does_xt;
        if (session->colon_does_off > session->colon_code_len) {
            colon_rollback(session);
            return false;
        }
        memcpy(does_code, session->colon_code + session->colon_does_off, does_len);
        does_off = does_len;
        if (!wrap_emit(does_code, &does_off, sizeof(does_code), OP_RET)) {
            colon_rollback(session);
            return false;
        }
        mod = session->module;
        memset(&dfn, 0, sizeof(dfn));
        dfn.name_idx = nvm_add_string(mod, "nl_forth_does", 13);
        dfn.arity = 0;
        dfn.code_offset = nvm_append_code(mod, does_code, does_off);
        dfn.code_length = does_off;
        dfn.local_count = 0;
        dfn.result_tag = TAG_VOID;
        dfn.result_count = 0;
        does_xt = nvm_add_function(mod, &dfn);
        if (does_xt >= mod->function_count) {
            colon_rollback(session);
            return false;
        }
        dver = nvm_verify_function(mod, does_xt);
        if (!dver.ok) {
            colon_rollback(session);
            return false;
        }
        session->colon_code_len = session->colon_does_off;
        if (!forth_colon_literal(session, (int64_t)does_xt)
                || !colon_emit(session, OP_PUSH_I64, (int64_t)FORTH_HOST_DOES)
                || !colon_emit(session, OP_CALL_EXTERN, session->runtime_import)
                || !colon_emit(session, OP_POP)) {
            colon_rollback(session);
            return false;
        }
        session->colon_does_pending = false;
    }
    if (!colon_emit(session, OP_RET)) {
        colon_rollback(session);
        return false;
    }
    mod = session->module;
    fn = &mod->functions[session->colon_fn_idx];
    code_off = nvm_append_code(mod, session->colon_code, session->colon_code_len);
    fn->code_offset = code_off;
    fn->code_length = session->colon_code_len;
    verified = nvm_verify_function(mod, session->colon_fn_idx);
    if (!verified.ok) {
        colon_rollback(session);
        return false;
    }
    if (!forth_reveal(session, session->colon_nt)) {
        colon_rollback(session);
        return false;
    }
    *nt = session->colon_nt;
    session->colon_open = false;
    session->colon_code_len = 0;
    if (!forth_session_rebuild(session)) {
        session->colon_open = true;
        colon_rollback(session);
        return false;
    }
    return true;
}

bool forth_colon_abort(ForthSession *session) {
    return colon_rollback(session);
}

bool forth_colon_is_open(const ForthSession *session) {
    return session && session->colon_open;
}

ForthXt forth_colon_xt(const ForthSession *session) {
    if (!session || !session->colon_open) return 0;
    return session->colon_fn_idx;
}

bool forth_colon_throw(ForthSession *session) {
    if (!session || !session->colon_open) return false;
    if (!colon_emit(session, OP_CALL, session->dpop_fn)) return false;
    return colon_emit(session, OP_CALL, session->throw_fn);
}

bool forth_catch(ForthSession *session, ForthXt xt, int64_t *code) {
    uint32_t saved_data;
    uint32_t saved_ret;
    uint32_t saved_fp;
    uint32_t saved_control;
    uint32_t saved_source;
    int64_t saved_to_in = 0;
    int64_t saved_blk = 0;
    int64_t thrown = 0;
    VmResult ran;

    if (!session || !code || !session->module) return false;
    if (xt >= session->module->function_count) return false;

    saved_data = forth_data_depth(session);
    saved_ret = forth_return_depth(session);
    saved_fp = session->fp_depth;
    saved_control = session->control_depth;
    saved_source = session->source_depth;
    if (!forth_fetch_cell(session, session->sysvars, &saved_to_in)) return false;
    if (!forth_fetch_cell(session, session->sysvars + FORTH_CELL_BYTES, &saved_blk))
        return false;
    if (!forth_store_cell(session, session->throw_code_addr, 0)) return false;

    ran = forth_invoke_nested(session, xt);
    if (ran != VM_OK) return false;
    if (!forth_fetch_cell(session, session->throw_code_addr, &thrown)) return false;
    if (thrown == 0) {
        *code = 0;
        return true;
    }

    while (session->source_depth > saved_source) {
        if (!forth_source_pop(session)) return false;
    }
    session->fp_depth = saved_fp;
    session->control_depth = saved_control;
    if (!forth_store_cell(session, session->data_depth_addr, (int64_t)saved_data))
        return false;
    if (!forth_store_cell(session, session->ret_depth_addr, (int64_t)saved_ret))
        return false;
    if (!forth_store_cell(session, session->sysvars, saved_to_in)) return false;
    if (!forth_store_cell(session, session->sysvars + FORTH_CELL_BYTES, saved_blk))
        return false;
    if (!forth_store_cell(session, session->throw_code_addr, 0)) return false;
    *code = thrown;
    return true;
}

static bool forth_tag_is_cell(uint8_t tag) {
    return tag == TAG_INT || tag == TAG_BOOL || tag == TAG_OPAQUE || tag == TAG_U8;
}

static bool forth_import_signature_ok(const uint8_t *param_tags, uint16_t param_count,
                                      uint8_t return_tag) {
    uint16_t i;
    bool any_float = (return_tag == TAG_FLOAT);
    bool any_nonfloat = (return_tag != TAG_FLOAT && return_tag != TAG_VOID);

    if (param_count > NANO_MAX_FFI_ARGS) return false;
    if (param_count > 0 && param_tags == NULL) return false;
    if (return_tag != TAG_VOID && !forth_tag_is_cell(return_tag)
            && return_tag != TAG_FLOAT)
        return false;
    for (i = 0; i < param_count; i++) {
        if (param_tags[i] == TAG_FLOAT) {
            any_float = true;
        } else {
            any_nonfloat = true;
            if (!forth_tag_is_cell(param_tags[i])) return false;
        }
    }
    if (any_float && any_nonfloat) return false;
    if (any_float) return false;
    return true;
}

static bool wrap_emit(uint8_t *code, uint32_t *off, uint32_t cap, NanoOpcode op, ...) {
    uint8_t buf[ISA_MAX_INSTRUCTION_SIZE];
    DecodedInstruction instr;
    const InstructionInfo *info;
    va_list args;
    uint32_t n;
    int i;

    memset(&instr, 0, sizeof(instr));
    instr.opcode = op;
    info = isa_get_info(op);
    if (!info) return false;
    va_start(args, op);
    for (i = 0; i < info->operand_count; i++) {
        switch (info->operands[i]) {
            case OPERAND_U8:  instr.operands[i].u8  = (uint8_t)va_arg(args, int);      break;
            case OPERAND_U16: instr.operands[i].u16 = (uint16_t)va_arg(args, int);     break;
            case OPERAND_U32: instr.operands[i].u32 = va_arg(args, uint32_t);          break;
            case OPERAND_I32: instr.operands[i].i32 = va_arg(args, int32_t);           break;
            case OPERAND_I64: instr.operands[i].i64 = va_arg(args, int64_t);           break;
            case OPERAND_F64: instr.operands[i].f64 = va_arg(args, double);            break;
            default:
                va_end(args);
                return false;
        }
    }
    va_end(args);
    n = isa_encode(&instr, buf, sizeof(buf));
    if (n == 0 || *off > cap - n) return false;
    memcpy(code + *off, buf, n);
    *off += n;
    return true;
}

static void wrap_patch_rel(uint8_t *code, uint32_t instr_off, uint32_t target_off) {
    int32_t rel = (int32_t)target_off - (int32_t)instr_off;
    code[instr_off + 1] = (uint8_t)(rel & 0xFF);
    code[instr_off + 2] = (uint8_t)((rel >> 8) & 0xFF);
    code[instr_off + 3] = (uint8_t)((rel >> 16) & 0xFF);
    code[instr_off + 4] = (uint8_t)((rel >> 24) & 0xFF);
}

static bool forth_publish_prim(ForthSession *session, const char *name,
                              const uint8_t *code, uint32_t off,
                              uint16_t locals, bool immediate, uint8_t host) {
    NvmFunctionEntry fn;
    NvmVerifyResult verified;
    NvmModule *mod;
    ForthNt nt = 0;
    uint32_t xt;
    ForthHeader *header;

    if (!session || !name || !code) return false;
    mod = session->module;
    memset(&fn, 0, sizeof(fn));
    fn.name_idx = nvm_add_string(mod, name, (uint32_t)strlen(name));
    fn.arity = 0;
    fn.code_offset = nvm_append_code(mod, code, off);
    fn.code_length = off;
    fn.local_count = locals;
    fn.result_tag = TAG_VOID;
    fn.result_count = 0;
    xt = nvm_add_function(mod, &fn);
    if (xt >= mod->function_count) return false;
    verified = nvm_verify_function(mod, xt);
    if (!verified.ok) return false;
    if (!forth_define(session, name, (uint32_t)strlen(name), xt, immediate,
                      false, &nt))
        return false;
    header = header_at(session, nt);
    if (!header) return false;
    header->host_kind = host;
    return true;
}

static bool wrap_dpop(ForthSession *session, uint8_t *code, uint32_t *off,
                      uint32_t cap, int local) {
    if (!wrap_emit(code, off, cap, OP_CALL, session->dpop_fn)) return false;
    return wrap_emit(code, off, cap, OP_STORE_LOCAL, local);
}

static bool wrap_dpush_local(ForthSession *session, uint8_t *code, uint32_t *off,
                             uint32_t cap, int local) {
    if (!wrap_emit(code, off, cap, OP_LOAD_LOCAL, local)) return false;
    return wrap_emit(code, off, cap, OP_CALL, session->dpush_fn);
}

static bool forth_install_binop(ForthSession *session, const char *name,
                                NanoOpcode op) {
    uint8_t code[128];
    uint32_t off = 0;
    if (!wrap_dpop(session, code, &off, sizeof(code), 1)) return false;
    if (!wrap_dpop(session, code, &off, sizeof(code), 0)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_LOAD_LOCAL, 0)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_LOAD_LOCAL, 1)) return false;
    if (!wrap_emit(code, &off, sizeof(code), op)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_CALL, session->dpush_fn))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_RET)) return false;
    return forth_publish_prim(session, name, code, off, 2, false, FORTH_HOST_NONE);
}

static bool forth_install_unop(ForthSession *session, const char *name,
                               NanoOpcode op) {
    uint8_t code[128];
    uint32_t off = 0;
    if (!wrap_dpop(session, code, &off, sizeof(code), 0)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_LOAD_LOCAL, 0)) return false;
    if (!wrap_emit(code, &off, sizeof(code), op)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_CALL, session->dpush_fn))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_RET)) return false;
    return forth_publish_prim(session, name, code, off, 1, false, FORTH_HOST_NONE);
}

static bool forth_install_cmp(ForthSession *session, const char *name,
                              NanoOpcode op) {
    uint8_t code[128];
    uint32_t off = 0;
    if (!wrap_dpop(session, code, &off, sizeof(code), 1)) return false;
    if (!wrap_dpop(session, code, &off, sizeof(code), 0)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_LOAD_LOCAL, 0)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_LOAD_LOCAL, 1)) return false;
    if (!wrap_emit(code, &off, sizeof(code), op)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_CAST_INT)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_I64_NEG)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_CALL, session->dpush_fn))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_RET)) return false;
    return forth_publish_prim(session, name, code, off, 2, false, FORTH_HOST_NONE);
}

static bool forth_install_host_flags(ForthSession *session, const char *name,
                                     bool immediate, bool compile_only,
                                     uint8_t host) {
    uint8_t code[16];
    uint32_t off = 0;
    ForthHeader *header;
    ForthNt nt = 0;
    ForthXt xt = 0;
    bool imm = false;

    if (!wrap_emit(code, &off, sizeof(code), OP_RET)) return false;
    if (!forth_publish_prim(session, name, code, off, 0, immediate, host))
        return false;
    if (!forth_find(session, name, (uint32_t)strlen(name), &nt, &xt, &imm))
        return false;
    header = header_at(session, nt);
    if (!header) return false;
    header->compile_only = compile_only;
    return true;
}

static bool forth_install_host(ForthSession *session, const char *name,
                               bool immediate, uint8_t host) {
    return forth_install_host_flags(session, name, immediate, false, host);
}

static bool forth_install_compile_only(ForthSession *session, const char *name,
                                       uint8_t host) {
    return forth_install_host_flags(session, name, true, true, host);
}

static bool forth_install_runtime_import(ForthSession *session) {
    NvmModule *mod;
    uint32_t mod_idx;
    uint32_t fn_idx;
    uint8_t tags[1];
    uint32_t before;

    if (!session || !session->module) return false;
    mod = session->module;
    before = mod->import_count;
    tags[0] = TAG_INT;
    mod_idx = nvm_add_string(mod, "", 0);
    fn_idx = nvm_add_string(mod, "nl_forth_runtime", 16);
    session->runtime_import = nvm_add_import(mod, mod_idx, fn_idx, 1, TAG_INT, tags);
    return mod->import_count == before + 1;
}

static bool forth_install_runtime_host(ForthSession *session, const char *name,
                                       uint8_t host) {
    uint8_t code[64];
    uint32_t off = 0;
    if (!wrap_emit(code, &off, sizeof(code), OP_PUSH_I64, (int64_t)host))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_CALL_EXTERN, session->runtime_import))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_POP)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_RET)) return false;
    return forth_publish_prim(session, name, code, off, 0, false, host);
}

static bool forth_install_throw_word(ForthSession *session) {
    uint8_t code[64];
    uint32_t off = 0;
    if (!wrap_emit(code, &off, sizeof(code), OP_CALL, session->dpop_fn))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_CALL, session->throw_fn))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_RET)) return false;
    return forth_publish_prim(session, "THROW", code, off, 0, false, FORTH_HOST_NONE);
}

static bool forth_install_bye(ForthSession *session) {
    uint8_t code[64];
    uint32_t off = 0;
    if (!wrap_emit(code, &off, sizeof(code), OP_PUSH_I64, (int64_t)FORTH_HOST_BYE))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_CALL_EXTERN, session->runtime_import))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_POP)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_HALT)) return false;
    return forth_publish_prim(session, "BYE", code, off, 0, false, FORTH_HOST_BYE);
}

int64_t nl_forth_runtime(int64_t kind) {
    ForthSession *session = g_forth;
    int rc;
    if (!session) return 0;
    rc = forth_run_host(session, (uint8_t)kind, 0);
    if (rc < 0) {
        forth_store_cell(session, session->throw_code_addr, -1);
    }
    return 0;
}

static bool forth_install_slashmod(ForthSession *session) {
    uint8_t code[512];
    uint32_t off = 0;
    uint32_t jmp_zero;
    uint32_t jmp_same;
    uint32_t pushq;

    if (!wrap_dpop(session, code, &off, sizeof(code), 1)) return false;
    if (!wrap_dpop(session, code, &off, sizeof(code), 0)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_LOAD_LOCAL, 0)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_LOAD_LOCAL, 1)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_I64_DIV_S)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_STORE_LOCAL, 2)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_LOAD_LOCAL, 0)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_LOAD_LOCAL, 1)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_I64_REM_S)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_STORE_LOCAL, 3)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_LOAD_LOCAL, 3)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_PUSH_I64, (int64_t)0)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_I64_EQ)) return false;
    jmp_zero = off;
    if (!wrap_emit(code, &off, sizeof(code), OP_JMP_TRUE, (int32_t)0)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_LOAD_LOCAL, 0)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_PUSH_I64, (int64_t)0)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_I64_LT_S)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_CAST_INT)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_LOAD_LOCAL, 1)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_PUSH_I64, (int64_t)0)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_I64_LT_S)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_CAST_INT)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_I64_XOR)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_PUSH_I64, (int64_t)0)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_I64_EQ)) return false;
    jmp_same = off;
    if (!wrap_emit(code, &off, sizeof(code), OP_JMP_TRUE, (int32_t)0)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_LOAD_LOCAL, 2)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_PUSH_I64, (int64_t)1)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_I64_SUB)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_STORE_LOCAL, 2)) return false;
    pushq = off;
    wrap_patch_rel(code, jmp_zero, pushq);
    wrap_patch_rel(code, jmp_same, pushq);
    if (!wrap_emit(code, &off, sizeof(code), OP_LOAD_LOCAL, 0)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_LOAD_LOCAL, 2)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_LOAD_LOCAL, 1)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_I64_MUL)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_I64_SUB)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_STORE_LOCAL, 3)) return false;
    if (!wrap_dpush_local(session, code, &off, sizeof(code), 3)) return false;
    if (!wrap_dpush_local(session, code, &off, sizeof(code), 2)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_RET)) return false;
    return forth_publish_prim(session, "/MOD", code, off, 4, false, FORTH_HOST_NONE);
}

static bool forth_install_core_colon(ForthSession *session) {
    static const char *const defs[] = {
        ": NIP SWAP DROP ;",
        ": TUCK SWAP OVER ;",
        ": 2DUP OVER OVER ;",
        ": 2DROP DROP DROP ;",
        ": 1+ 1 + ;",
        ": 1- 1 - ;",
        ": 2* DUP + ;",
        ": CELL+ 8 + ;",
        ": CELLS 8 * ;",
        ": CHAR+ 1 + ;",
        ": CHARS ;",
        ": 0= 0 = ;",
        ": 0< 0 < ;",
        ": 0> 0 > ;",
        ": <> = INVERT ;",
        ": 0<> 0= INVERT ;",
        ": U> SWAP U< ;",
        ": ABS DUP 0< IF NEGATE THEN ;",
        ": MIN 2DUP > IF SWAP THEN DROP ;",
        ": MAX 2DUP < IF SWAP THEN DROP ;",
        ": TRUE -1 ;",
        ": FALSE 0 ;",
        ": BL 32 ;",
        ": DECIMAL 10 BASE ! ;",
        ": HEX 16 BASE ! ;",
        ": +! DUP @ ROT + SWAP ! ;",
        ": / /MOD SWAP DROP ;",
        ": MOD /MOD DROP ;",
        ": ?DUP DUP IF DUP THEN ;",
        ": S>D DUP 0< ;",
        ": ALIGNED 7 + -8 AND ;",
        ": COUNT DUP C@ SWAP CHAR+ SWAP ;",
        ": 2! SWAP OVER ! CELL+ ! ;",
        ": 2@ DUP CELL+ @ SWAP @ ;",
        ": 2OVER 3 PICK 3 PICK ;",
        ": 2SWAP 3 ROLL 3 ROLL ;",
        ": -ROT ROT ROT ;",
        ": WITHIN OVER - >R - R> U< ;",
        ": SPACE BL EMIT ;",
        ": SPACES DUP 0> IF 0 DO BL EMIT LOOP ELSE DROP THEN ;",
        ": */MOD >R M* R> FM/MOD ;",
        ": */ */MOD SWAP DROP ;",
        NULL
    };
    uint32_t i;

    for (i = 0; defs[i] != NULL; i++) {
        const char *line = defs[i];
        if (!forth_interpret(session, (const uint8_t *)line, (uint32_t)strlen(line)))
            return false;
    }
    return true;
}

static bool forth_install_kernel(ForthSession *session) {
    uint8_t code[512];
    uint32_t off = 0;
    ForthNt nt = 0;

    if (!session) return false;
    if (!forth_install_binop(session, "+", OP_I64_ADD)) return false;
    if (!forth_install_binop(session, "-", OP_I64_SUB)) return false;
    if (!forth_install_binop(session, "*", OP_I64_MUL)) return false;
    if (!forth_install_binop(session, "AND", OP_I64_AND)) return false;
    if (!forth_install_binop(session, "OR", OP_I64_OR)) return false;
    if (!forth_install_binop(session, "XOR", OP_I64_XOR)) return false;
    if (!forth_install_binop(session, "LSHIFT", OP_I64_SHL)) return false;
    if (!forth_install_binop(session, "RSHIFT", OP_I64_SHR_U)) return false;
    if (!forth_install_unop(session, "NEGATE", OP_I64_NEG)) return false;
    if (!forth_install_unop(session, "INVERT", OP_I64_INVERT)) return false;
    if (!forth_install_cmp(session, "=", OP_I64_EQ)) return false;
    if (!forth_install_cmp(session, "<", OP_I64_LT_S)) return false;
    if (!forth_install_cmp(session, ">", OP_I64_GT_S)) return false;
    if (!forth_install_cmp(session, "U<", OP_I64_LT_U)) return false;
    if (!forth_install_slashmod(session)) return false;

    off = 0;
    if (!wrap_dpop(session, code, &off, sizeof(code), 0)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_LOAD_LOCAL, 0)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_PUSH_I64, (int64_t)1)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_I64_SHR_S)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_CALL, session->dpush_fn))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_RET)) return false;
    if (!forth_publish_prim(session, "2/", code, off, 1, false, FORTH_HOST_NONE))
        return false;

    off = 0;
    if (!wrap_dpop(session, code, &off, sizeof(code), 0)) return false;
    if (!wrap_dpush_local(session, code, &off, sizeof(code), 0)) return false;
    if (!wrap_dpush_local(session, code, &off, sizeof(code), 0)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_RET)) return false;
    if (!forth_publish_prim(session, "DUP", code, off, 1, false, FORTH_HOST_NONE))
        return false;

    off = 0;
    if (!wrap_dpop(session, code, &off, sizeof(code), 0)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_RET)) return false;
    if (!forth_publish_prim(session, "DROP", code, off, 1, false, FORTH_HOST_NONE))
        return false;

    off = 0;
    if (!wrap_dpop(session, code, &off, sizeof(code), 1)) return false;
    if (!wrap_dpop(session, code, &off, sizeof(code), 0)) return false;
    if (!wrap_dpush_local(session, code, &off, sizeof(code), 1)) return false;
    if (!wrap_dpush_local(session, code, &off, sizeof(code), 0)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_RET)) return false;
    if (!forth_publish_prim(session, "SWAP", code, off, 2, false, FORTH_HOST_NONE))
        return false;

    off = 0;
    if (!wrap_dpop(session, code, &off, sizeof(code), 1)) return false;
    if (!wrap_dpop(session, code, &off, sizeof(code), 0)) return false;
    if (!wrap_dpush_local(session, code, &off, sizeof(code), 0)) return false;
    if (!wrap_dpush_local(session, code, &off, sizeof(code), 1)) return false;
    if (!wrap_dpush_local(session, code, &off, sizeof(code), 0)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_RET)) return false;
    if (!forth_publish_prim(session, "OVER", code, off, 2, false, FORTH_HOST_NONE))
        return false;

    off = 0;
    if (!wrap_dpop(session, code, &off, sizeof(code), 2)) return false;
    if (!wrap_dpop(session, code, &off, sizeof(code), 1)) return false;
    if (!wrap_dpop(session, code, &off, sizeof(code), 0)) return false;
    if (!wrap_dpush_local(session, code, &off, sizeof(code), 1)) return false;
    if (!wrap_dpush_local(session, code, &off, sizeof(code), 2)) return false;
    if (!wrap_dpush_local(session, code, &off, sizeof(code), 0)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_RET)) return false;
    if (!forth_publish_prim(session, "ROT", code, off, 3, false, FORTH_HOST_NONE))
        return false;

    off = 0;
    if (!wrap_dpop(session, code, &off, sizeof(code), 0)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_LOAD_LOCAL, 0)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_MEM_LOAD64)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_CALL, session->dpush_fn))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_RET)) return false;
    if (!forth_publish_prim(session, "@", code, off, 1, false, FORTH_HOST_NONE))
        return false;

    off = 0;
    if (!wrap_dpop(session, code, &off, sizeof(code), 1)) return false;
    if (!wrap_dpop(session, code, &off, sizeof(code), 0)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_LOAD_LOCAL, 1)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_LOAD_LOCAL, 0)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_MEM_STORE64)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_RET)) return false;
    if (!forth_publish_prim(session, "!", code, off, 2, false, FORTH_HOST_NONE))
        return false;

    off = 0;
    if (!wrap_dpop(session, code, &off, sizeof(code), 0)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_LOAD_LOCAL, 0)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_MEM_LOAD8)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_CALL, session->dpush_fn))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_RET)) return false;
    if (!forth_publish_prim(session, "C@", code, off, 1, false, FORTH_HOST_NONE))
        return false;

    off = 0;
    if (!wrap_dpop(session, code, &off, sizeof(code), 1)) return false;
    if (!wrap_dpop(session, code, &off, sizeof(code), 0)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_LOAD_LOCAL, 1)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_LOAD_LOCAL, 0)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_MEM_STORE8)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_RET)) return false;
    if (!forth_publish_prim(session, "C!", code, off, 2, false, FORTH_HOST_NONE))
        return false;

    off = 0;
    if (!wrap_emit(code, &off, sizeof(code), OP_PUSH_I64,
                   (int64_t)session->data_depth_addr))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_MEM_LOAD64)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_CALL, session->dpush_fn))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_RET)) return false;
    if (!forth_publish_prim(session, "DEPTH", code, off, 0, false, FORTH_HOST_NONE))
        return false;

    off = 0;
    if (!wrap_emit(code, &off, sizeof(code), OP_PUSH_I64,
                   (int64_t)session->here_cell_addr))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_MEM_LOAD64)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_CALL, session->dpush_fn))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_RET)) return false;
    if (!forth_publish_prim(session, "HERE", code, off, 0, false, FORTH_HOST_NONE))
        return false;

    off = 0;
    if (!wrap_emit(code, &off, sizeof(code), OP_PUSH_I64,
                   (int64_t)session->ret_depth_addr))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_MEM_LOAD64)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_PUSH_I64, (int64_t)1)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_I64_SUB)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_PUSH_I64, (int64_t)FORTH_CELL_BYTES))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_I64_MUL)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_PUSH_I64,
                   (int64_t)session->ret_stack_addr))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_I64_ADD)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_MEM_LOAD64)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_CALL, session->dpush_fn))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_RET)) return false;
    if (!forth_publish_prim(session, "I", code, off, 0, false, FORTH_HOST_NONE))
        return false;

    off = 0;
    if (!wrap_emit(code, &off, sizeof(code), OP_PUSH_I64,
                   (int64_t)session->ret_depth_addr))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_MEM_LOAD64)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_PUSH_I64, (int64_t)3)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_I64_SUB)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_PUSH_I64, (int64_t)FORTH_CELL_BYTES))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_I64_MUL)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_PUSH_I64,
                   (int64_t)session->ret_stack_addr))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_I64_ADD)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_MEM_LOAD64)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_CALL, session->dpush_fn))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_RET)) return false;
    if (!forth_publish_prim(session, "J", code, off, 0, false, FORTH_HOST_NONE))
        return false;

    off = 0;
    if (!wrap_emit(code, &off, sizeof(code), OP_CALL, session->dpop_fn))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_CALL, session->rpush_fn))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_RET)) return false;
    if (!forth_publish_prim(session, ">R", code, off, 0, false, FORTH_HOST_NONE))
        return false;

    off = 0;
    if (!wrap_emit(code, &off, sizeof(code), OP_CALL, session->rpop_fn))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_CALL, session->dpush_fn))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_RET)) return false;
    if (!forth_publish_prim(session, "R>", code, off, 0, false, FORTH_HOST_NONE))
        return false;

    off = 0;
    if (!wrap_emit(code, &off, sizeof(code), OP_PUSH_I64,
                   (int64_t)session->ret_depth_addr))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_MEM_LOAD64)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_PUSH_I64, (int64_t)1)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_I64_SUB)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_PUSH_I64, (int64_t)FORTH_CELL_BYTES))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_I64_MUL)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_PUSH_I64,
                   (int64_t)session->ret_stack_addr))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_I64_ADD)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_MEM_LOAD64)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_CALL, session->dpush_fn))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_RET)) return false;
    if (!forth_publish_prim(session, "R@", code, off, 0, false, FORTH_HOST_NONE))
        return false;

    off = 0;
    if (!wrap_emit(code, &off, sizeof(code), OP_CALL, session->unloop_fn))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_RET)) return false;
    if (!forth_publish_prim(session, "UNLOOP", code, off, 0, false, FORTH_HOST_NONE))
        return false;

    off = 0;
    if (!wrap_dpop(session, code, &off, sizeof(code), 1)) return false;
    if (!wrap_dpop(session, code, &off, sizeof(code), 0)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_LOAD_LOCAL, 0)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_LOAD_LOCAL, 1)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_I64_MUL_WIDE_S)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_STORE_LOCAL, 1)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_STORE_LOCAL, 0)) return false;
    if (!wrap_dpush_local(session, code, &off, sizeof(code), 0)) return false;
    if (!wrap_dpush_local(session, code, &off, sizeof(code), 1)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_RET)) return false;
    if (!forth_publish_prim(session, "M*", code, off, 2, false, FORTH_HOST_NONE))
        return false;

    off = 0;
    if (!wrap_dpop(session, code, &off, sizeof(code), 1)) return false;
    if (!wrap_dpop(session, code, &off, sizeof(code), 0)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_LOAD_LOCAL, 0)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_LOAD_LOCAL, 1)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_I64_MUL_WIDE_U)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_STORE_LOCAL, 1)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_STORE_LOCAL, 0)) return false;
    if (!wrap_dpush_local(session, code, &off, sizeof(code), 0)) return false;
    if (!wrap_dpush_local(session, code, &off, sizeof(code), 1)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_RET)) return false;
    if (!forth_publish_prim(session, "UM*", code, off, 2, false, FORTH_HOST_NONE))
        return false;

    if (!forth_install_runtime_import(session)) return false;

    if (!forth_install_compile_only(session, ":", FORTH_HOST_COLON)) return false;
    if (!forth_install_compile_only(session, ";", FORTH_HOST_SEMI)) return false;
    if (!forth_install_compile_only(session, "IF", FORTH_HOST_IF)) return false;
    if (!forth_install_compile_only(session, "ELSE", FORTH_HOST_ELSE)) return false;
    if (!forth_install_compile_only(session, "THEN", FORTH_HOST_THEN)) return false;
    if (!forth_install_compile_only(session, "BEGIN", FORTH_HOST_BEGIN)) return false;
    if (!forth_install_compile_only(session, "UNTIL", FORTH_HOST_UNTIL)) return false;
    if (!forth_install_compile_only(session, "AGAIN", FORTH_HOST_AGAIN)) return false;
    if (!forth_install_compile_only(session, "WHILE", FORTH_HOST_WHILE)) return false;
    if (!forth_install_compile_only(session, "REPEAT", FORTH_HOST_REPEAT)) return false;
    if (!forth_install_compile_only(session, "DO", FORTH_HOST_DO)) return false;
    if (!forth_install_compile_only(session, "LOOP", FORTH_HOST_LOOP)) return false;
    if (!forth_install_compile_only(session, "+LOOP", FORTH_HOST_PLUS_LOOP))
        return false;
    if (!forth_install_compile_only(session, "?DO", FORTH_HOST_QDO)) return false;
    if (!forth_install_compile_only(session, "LEAVE", FORTH_HOST_LEAVE)) return false;
    if (!forth_install_compile_only(session, "EXIT", FORTH_HOST_EXIT)) return false;
    if (!forth_install_compile_only(session, "RECURSE", FORTH_HOST_RECURSE))
        return false;
    if (!forth_install_host(session, "[", true, FORTH_HOST_LBRACKET)) return false;
    if (!forth_install_compile_only(session, "]", FORTH_HOST_RBRACKET)) return false;
    if (!forth_install_compile_only(session, "LITERAL", FORTH_HOST_LITERAL))
        return false;
    if (!forth_install_host(session, "IMMEDIATE", false, FORTH_HOST_IMMEDIATE))
        return false;
    if (!forth_install_runtime_host(session, "'", FORTH_HOST_TICK)) return false;
    if (!forth_install_compile_only(session, "[']", FORTH_HOST_BRACKET_TICK))
        return false;
    if (!forth_install_runtime_host(session, "CHAR", FORTH_HOST_CHAR)) return false;
    if (!forth_install_compile_only(session, "[CHAR]", FORTH_HOST_BRACKET_CHAR))
        return false;
    if (!forth_install_runtime_host(session, "CONSTANT", FORTH_HOST_CONSTANT))
        return false;
    if (!forth_install_runtime_host(session, "VARIABLE", FORTH_HOST_VARIABLE))
        return false;
    if (!forth_install_runtime_host(session, "ALLOT", FORTH_HOST_ALLOT)) return false;
    if (!forth_install_runtime_host(session, ",", FORTH_HOST_COMMA)) return false;
    if (!forth_install_runtime_host(session, "ALIGN", FORTH_HOST_ALIGN)) return false;
    if (!forth_install_runtime_host(session, "EXECUTE", FORTH_HOST_EXECUTE))
        return false;
    if (!forth_install_host(session, "\\", true, FORTH_HOST_BACKSLASH)) return false;
    if (!forth_install_host(session, "(", true, FORTH_HOST_PAREN)) return false;
    if (!forth_install_host(session, "S\"", true, FORTH_HOST_S_QUOTE)) return false;
    if (!forth_install_host(session, ".\"", true, FORTH_HOST_DOT_QUOTE)) return false;
    if (!forth_install_host(session, "DOES>", true, FORTH_HOST_DOES)) return false;
    if (!forth_install_runtime_host(session, "CREATE", FORTH_HOST_CREATE))
        return false;
    if (!forth_install_runtime_host(session, "SOURCE", FORTH_HOST_SOURCE))
        return false;
    if (!forth_install_runtime_host(session, "EVALUATE", FORTH_HOST_EVALUATE))
        return false;
    if (!forth_install_runtime_host(session, "FIND", FORTH_HOST_FIND)) return false;
    if (!forth_install_runtime_host(session, "WORD", FORTH_HOST_WORD)) return false;
    if (!forth_install_runtime_host(session, "PARSE", FORTH_HOST_PARSE))
        return false;
    if (!forth_install_runtime_host(session, "EMIT", FORTH_HOST_EMIT)) return false;
    if (!forth_install_runtime_host(session, "TYPE", FORTH_HOST_TYPE)) return false;
    if (!forth_install_runtime_host(session, "CR", FORTH_HOST_CR)) return false;
    if (!forth_install_runtime_host(session, "ENVIRONMENT?", FORTH_HOST_ENVIRONMENT))
        return false;
    if (!forth_install_runtime_host(session, "ABORT", FORTH_HOST_ABORT))
        return false;
    if (!forth_install_runtime_host(session, "UM/MOD", FORTH_HOST_UM_MOD))
        return false;
    if (!forth_install_runtime_host(session, "SM/REM", FORTH_HOST_SM_REM))
        return false;
    if (!forth_install_runtime_host(session, "FM/MOD", FORTH_HOST_FM_MOD))
        return false;
    if (!forth_install_runtime_host(session, "FILL", FORTH_HOST_FILL)) return false;
    if (!forth_install_runtime_host(session, "MOVE", FORTH_HOST_MOVE)) return false;
    if (!forth_install_runtime_host(session, "C,", FORTH_HOST_C_COMMA))
        return false;
    if (!forth_install_runtime_host(session, "PICK", FORTH_HOST_PICK)) return false;
    if (!forth_install_runtime_host(session, "ROLL", FORTH_HOST_ROLL)) return false;
    if (!forth_install_runtime_host(session, "<#", FORTH_HOST_LESS_NUM))
        return false;
    if (!forth_install_runtime_host(session, "HOLD", FORTH_HOST_HOLD)) return false;
    if (!forth_install_runtime_host(session, "SIGN", FORTH_HOST_SIGN)) return false;
    if (!forth_install_runtime_host(session, "#", FORTH_HOST_HASH)) return false;
    if (!forth_install_runtime_host(session, "#S", FORTH_HOST_HASH_S)) return false;
    if (!forth_install_runtime_host(session, "#>", FORTH_HOST_NUM_END))
        return false;
    if (!forth_install_runtime_host(session, ".", FORTH_HOST_DOT)) return false;
    if (!forth_install_runtime_host(session, "U.", FORTH_HOST_UDOT)) return false;
    if (!forth_install_throw_word(session)) return false;
    if (!forth_install_runtime_host(session, "CATCH", FORTH_HOST_CATCH)) return false;
    if (!forth_install_bye(session)) return false;
    if (!forth_install_runtime_host(session, "COMPILE,", FORTH_HOST_COMPILE_COMMA))
        return false;
    if (!forth_install_compile_only(session, "POSTPONE", FORTH_HOST_POSTPONE))
        return false;
    if (!forth_install_runtime_host(session, ">BODY", FORTH_HOST_TO_BODY))
        return false;
    if (!forth_install_runtime_host(session, ">NUMBER", FORTH_HOST_TO_NUMBER))
        return false;
    if (!forth_install_host(session, "ABORT\"", true, FORTH_HOST_ABORT_QUOTE))
        return false;
    if (!forth_install_runtime_host(session, "ACCEPT", FORTH_HOST_ACCEPT))
        return false;
    if (!forth_install_runtime_host(session, "KEY", FORTH_HOST_KEY)) return false;
    if (!forth_install_runtime_host(session, "QUIT", FORTH_HOST_QUIT)) return false;

    if (!forth_session_rebuild(session)) return false;

    if (!forth_colon_begin(session, "BASE", 4)) return false;
    if (!forth_colon_literal(session, (int64_t)forth_base_addr(session)))
        return false;
    if (!forth_colon_finish(session, &nt)) return false;

    if (!forth_colon_begin(session, ">IN", 3)) return false;
    if (!forth_colon_literal(session, (int64_t)forth_to_in_addr(session)))
        return false;
    if (!forth_colon_finish(session, &nt)) return false;

    if (!forth_colon_begin(session, "BLK", 3)) return false;
    if (!forth_colon_literal(session, (int64_t)forth_blk_addr(session)))
        return false;
    if (!forth_colon_finish(session, &nt)) return false;

    if (!forth_colon_begin(session, "STATE", 5)) return false;
    if (!forth_colon_literal(session, (int64_t)forth_state_addr(session)))
        return false;
    if (!forth_colon_finish(session, &nt)) return false;

    if (!forth_colon_begin(session, "PAD", 3)) return false;
    if (!forth_colon_literal(session, (int64_t)session->pad_addr))
        return false;
    if (!forth_colon_finish(session, &nt)) return false;

    return forth_install_core_colon(session);
}

bool forth_import_declare(ForthSession *session, const char *module_name,
                          const char *symbol, const uint8_t *param_tags,
                          uint16_t param_count, uint8_t return_tag,
                          ForthNt *nt) {
    NvmModule *mod;
    uint32_t mod_idx;
    uint32_t fn_idx;
    uint32_t import_idx;
    uint32_t saved_imports;
    uint32_t saved_fns;
    uint32_t saved_code;
    uint8_t code[512];
    uint32_t off = 0;
    uint16_t i;
    NvmFunctionEntry fn;
    NvmVerifyResult verified;
    ForthNt defined = 0;
    uint32_t name_len;

    if (!session || !symbol || !nt || session->colon_open) return false;
    name_len = (uint32_t)strlen(symbol);
    if (name_len == 0 || name_len > FORTH_NAME_MAX) return false;
    if (!forth_import_signature_ok(param_tags, param_count, return_tag))
        return false;
    mod = session->module;
    if (!mod) return false;

    saved_imports = mod->import_count;
    saved_fns = mod->function_count;
    saved_code = mod->code_size;

    if (!module_name) module_name = "";
    mod_idx = nvm_add_string(mod, module_name, (uint32_t)strlen(module_name));
    fn_idx = nvm_add_string(mod, symbol, name_len);
    import_idx = nvm_add_import(mod, mod_idx, fn_idx, param_count, return_tag,
                                param_tags);
    if (mod->import_count != saved_imports + 1) return false;

    for (i = 0; i < param_count; i++) {
        if (!wrap_emit(code, &off, sizeof(code), OP_CALL, session->dpop_fn))
            goto fail;
        if (!wrap_emit(code, &off, sizeof(code), OP_STORE_LOCAL, (int)i))
            goto fail;
    }
    for (i = param_count; i > 0; i--) {
        if (!wrap_emit(code, &off, sizeof(code), OP_LOAD_LOCAL, (int)(i - 1)))
            goto fail;
    }
    if (!wrap_emit(code, &off, sizeof(code), OP_CALL_EXTERN, import_idx))
        goto fail;
    if (return_tag != TAG_VOID) {
        if (!wrap_emit(code, &off, sizeof(code), OP_CALL, session->dpush_fn))
            goto fail;
    }
    if (!wrap_emit(code, &off, sizeof(code), OP_RET)) goto fail;

    memset(&fn, 0, sizeof(fn));
    fn.name_idx = fn_idx;
    fn.arity = 0;
    fn.code_offset = nvm_append_code(mod, code, off);
    fn.code_length = off;
    fn.local_count = param_count;
    fn.result_tag = TAG_VOID;
    fn.result_count = 0;
    if (nvm_add_function(mod, &fn) != saved_fns) goto fail;
    if (mod->function_count != saved_fns + 1) goto fail;
    verified = nvm_verify_function(mod, saved_fns);
    if (!verified.ok) goto fail;
    if (!forth_define(session, symbol, name_len, (ForthXt)saved_fns, false, false,
                      &defined))
        goto fail;
    if (!forth_session_rebuild(session)) goto fail;
    nvm_call_descriptors_reset(mod);
    vm_ffi_cop_stop(&session->vm);
    *nt = defined;
    return true;

fail:
    mod->import_count = saved_imports;
    mod->function_count = saved_fns;
    if (saved_code <= mod->code_size)
        mod->code_size = saved_code;
    return false;
}

char *forth_see(const ForthSession *session, ForthXt xt) {
    const NvmModule *mod;
    const NvmFunctionEntry *fn;
    const uint8_t *code;
    char *buf = NULL;
    size_t size = 0;
    FILE *out;
    uint32_t off = 0;
    const char *word_name = NULL;

    if (!session || !session->module) return NULL;
    mod = session->module;
    if (xt >= mod->function_count) return NULL;
    fn = &mod->functions[xt];
    if (fn->name_idx < mod->string_count)
        word_name = nvm_get_string(mod, fn->name_idx);

    out = open_memstream(&buf, &size);
    if (!out) return NULL;
    if (word_name && word_name[0])
        fprintf(out, "%s\n", word_name);
    while (off < fn->code_length) {
        DecodedInstruction instr;
        uint32_t n = isa_decode(mod->code + fn->code_offset + off,
                                fn->code_length - off, &instr);
        if (n == 0) break;
        if (instr.opcode == OP_CALL_EXTERN
                && instr.operands[0].u32 < mod->import_count) {
            const NvmImportEntry *imp = &mod->imports[instr.operands[0].u32];
            const char *mod_name = nvm_get_string(mod, imp->module_name_idx);
            const char *sym = nvm_get_string(mod, imp->function_name_idx);
            fprintf(out, "imported %s%s%s\n",
                    mod_name && mod_name[0] ? mod_name : "",
                    mod_name && mod_name[0] ? " " : "",
                    sym ? sym : "");
        }
        off += n;
    }
    code = mod->code + fn->code_offset;
    disasm_function_styled(code, fn->code_length, mod, out, DISASM_STYLE_DETAILED);
    fclose(out);
    return buf;
}

static bool forth_is_blank(uint8_t c) {
    return c == (uint8_t)' ' || c == (uint8_t)'\t' || c == (uint8_t)'\n'
        || c == (uint8_t)'\r';
}

static int forth_digit_value(uint8_t c) {
    if (c >= (uint8_t)'0' && c <= (uint8_t)'9') return (int)(c - (uint8_t)'0');
    if (c >= (uint8_t)'a' && c <= (uint8_t)'z') return (int)(c - (uint8_t)'a') + 10;
    if (c >= (uint8_t)'A' && c <= (uint8_t)'Z') return (int)(c - (uint8_t)'A') + 10;
    return -1;
}

static bool forth_parse_number(ForthSession *session, const uint8_t *name,
                               uint32_t len, int64_t *out) {
    uint32_t i = 0;
    int sign = 1;
    int64_t value = 0;
    int64_t base = 10;

    if (!session || !name || !out || len == 0) return false;
    if (!forth_fetch_cell(session, forth_base_addr(session), &base)) return false;
    if (base < 2 || base > 36) return false;
    if (name[0] == (uint8_t)'-' && len > 1) {
        sign = -1;
        i = 1;
    }
    if (i >= len) return false;
    for (; i < len; i++) {
        int digit = forth_digit_value(name[i]);
        if (digit < 0 || (int64_t)digit >= base) return false;
        if (value > (INT64_MAX - digit) / base) return false;
        value = value * base + digit;
    }
    *out = sign < 0 ? -value : value;
    return true;
}

static int forth_take_word(ForthSession *session, uint8_t *name, uint32_t *nlen) {
    uint64_t caddr = 0;
    uint64_t u = 0;
    int64_t to_in = 0;
    uint32_t start;
    uint32_t i;

    if (!session || !name || !nlen) return -1;
    if (!forth_source(session, &caddr, &u)) return -1;
    if (!forth_fetch_cell(session, session->sysvars, &to_in)) return -1;
    if (to_in < 0) return -1;
    while ((uint64_t)to_in < u) {
        uint8_t ch = 0;
        if (!forth_fetch_byte(session, caddr + (uint64_t)to_in, &ch)) return -1;
        if (!forth_is_blank(ch)) break;
        to_in++;
    }
    if ((uint64_t)to_in >= u) {
        if (!forth_store_cell(session, session->sysvars, to_in)) return -1;
        *nlen = 0;
        return 0;
    }
    start = (uint32_t)to_in;
    while ((uint64_t)to_in < u) {
        uint8_t ch = 0;
        if (!forth_fetch_byte(session, caddr + (uint64_t)to_in, &ch)) return -1;
        if (forth_is_blank(ch)) break;
        to_in++;
    }
    *nlen = (uint32_t)to_in - start;
    if (*nlen == 0 || *nlen > FORTH_NAME_MAX) return -1;
    for (i = 0; i < *nlen; i++) {
        if (!forth_fetch_byte(session, caddr + start + i, &name[i])) return -1;
    }
    if ((uint64_t)to_in < u) {
        uint8_t trail = 0;
        if (!forth_fetch_byte(session, caddr + (uint64_t)to_in, &trail)) return -1;
        if (forth_is_blank(trail)) to_in++;
    }
    if (!forth_store_cell(session, session->sysvars, to_in)) return -1;
    return 1;
}

static bool forth_dict_allot(ForthSession *session, int64_t n) {
    uint64_t addr;
    uint64_t size;
    if (!session || n < 0) return false;
    if (n == 0) return true;
    size = (uint64_t)n;
    addr = session->bump;
    if (addr > UINT64_MAX - size) return false;
    if (!ensure_memory(session, addr + size)) return false;
    if (!regions_reserve(session, 1)) return false;
    session->regions[session->region_count].addr = addr;
    session->regions[session->region_count].size = size;
    session->regions[session->region_count].used = true;
    session->regions[session->region_count].pinned = false;
    session->region_count++;
    session->bump = addr + size;
    return forth_store_cell(session, session->here_cell_addr, (int64_t)session->bump);
}

static bool forth_dict_align(ForthSession *session) {
    uint64_t aligned;
    if (!session) return false;
    aligned = align_cells(session->bump);
    if (aligned < session->bump) return false;
    if (aligned == session->bump) return true;
    return forth_dict_allot(session, (int64_t)(aligned - session->bump));
}

static bool forth_skip_until(ForthSession *session, uint8_t closer, bool newline) {
    uint64_t caddr = 0;
    uint64_t u = 0;
    int64_t to_in = 0;

    if (!forth_source(session, &caddr, &u)) return false;
    if (!forth_fetch_cell(session, session->sysvars, &to_in)) return false;
    while ((uint64_t)to_in < u) {
        uint8_t ch = 0;
        if (!forth_fetch_byte(session, caddr + (uint64_t)to_in, &ch)) return false;
        to_in++;
        if (newline && (ch == (uint8_t)'\n' || ch == (uint8_t)'\r')) break;
        if (!newline && ch == closer) break;
    }
    return forth_store_cell(session, session->sysvars, to_in);
}

static bool forth_skip_blanks(ForthSession *session) {
    uint64_t caddr = 0;
    uint64_t u = 0;
    int64_t to_in = 0;

    if (!forth_source(session, &caddr, &u)) return false;
    if (!forth_fetch_cell(session, session->sysvars, &to_in)) return false;
    if (to_in < 0) return false;
    while ((uint64_t)to_in < u) {
        uint8_t ch = 0;
        if (!forth_fetch_byte(session, caddr + (uint64_t)to_in, &ch)) return false;
        if (!forth_is_blank(ch)) break;
        to_in++;
    }
    return forth_store_cell(session, session->sysvars, to_in);
}

static bool forth_parse_delimited(ForthSession *session, uint8_t delim, bool skip_lead,
                                  uint64_t *caddr_out, uint32_t *len_out) {
    uint64_t caddr = 0;
    uint64_t u = 0;
    int64_t to_in = 0;
    uint32_t start;

    if (!forth_source(session, &caddr, &u)) return false;
    if (!forth_fetch_cell(session, session->sysvars, &to_in)) return false;
    if (to_in < 0) return false;
    if (skip_lead) {
        while ((uint64_t)to_in < u) {
            uint8_t ch = 0;
            if (!forth_fetch_byte(session, caddr + (uint64_t)to_in, &ch))
                return false;
            if (ch != delim && !(delim == (uint8_t)' ' && forth_is_blank(ch)))
                break;
            to_in++;
        }
    }
    start = (uint32_t)to_in;
    while ((uint64_t)to_in < u) {
        uint8_t ch = 0;
        if (!forth_fetch_byte(session, caddr + (uint64_t)to_in, &ch)) return false;
        if (ch == delim || (delim == (uint8_t)' ' && forth_is_blank(ch)))
            break;
        to_in++;
    }
    *caddr_out = caddr + start;
    *len_out = (uint32_t)to_in - start;
    if ((uint64_t)to_in < u) to_in++;
    return forth_store_cell(session, session->sysvars, to_in);
}

static bool forth_type_range(ForthSession *session, uint64_t caddr, uint32_t len) {
    uint32_t i;
    for (i = 0; i < len; i++) {
        uint8_t ch = 0;
        if (!forth_fetch_byte(session, caddr + i, &ch)) return false;
        if (!forth_emit_char(session, ch)) return false;
    }
    return true;
}

static bool forth_copy_to_word(ForthSession *session, uint64_t src, uint32_t len,
                               bool counted) {
    uint32_t i;
    uint32_t stored = len;
    if (stored > FORTH_WORD_MAX - 1) stored = FORTH_WORD_MAX - 1;
    if (counted) {
        if (!forth_store_byte(session, session->word_addr, (uint8_t)stored))
            return false;
        for (i = 0; i < stored; i++) {
            uint8_t ch = 0;
            if (!forth_fetch_byte(session, src + i, &ch)) return false;
            if (!forth_store_byte(session, session->word_addr + 1 + i, ch))
                return false;
        }
    } else {
        for (i = 0; i < stored; i++) {
            uint8_t ch = 0;
            if (!forth_fetch_byte(session, src + i, &ch)) return false;
            if (!forth_store_byte(session, session->word_addr + i, ch))
                return false;
        }
    }
    return true;
}

static bool forth_env_query(ForthSession *session, uint64_t caddr, uint32_t len) {
    char name[64];
    uint32_t i;
    uint32_t n = len < 63 ? len : 63;
    for (i = 0; i < n; i++) {
        uint8_t ch = 0;
        if (!forth_fetch_byte(session, caddr + i, &ch)) return false;
        name[i] = (char)ch;
    }
    name[n] = '\0';
    if (strcmp(name, "/COUNTED-STRING") == 0) {
        return forth_data_push(session, 255) && forth_data_push(session, -1);
    }
    if (strcmp(name, "/HOLD") == 0) {
        return forth_data_push(session, FORTH_HOLD_MAX)
            && forth_data_push(session, -1);
    }
    if (strcmp(name, "/PAD") == 0) {
        return forth_data_push(session, FORTH_PAD_MAX)
            && forth_data_push(session, -1);
    }
    if (strcmp(name, "RETURN-STACK-CELLS") == 0) {
        return forth_data_push(session, FORTH_RETURN_STACK_CELLS)
            && forth_data_push(session, -1);
    }
    if (strcmp(name, "STACK-CELLS") == 0) {
        return forth_data_push(session, FORTH_STACK_CELLS)
            && forth_data_push(session, -1);
    }
    if (strcmp(name, "ADDRESS-UNIT-BITS") == 0) {
        return forth_data_push(session, 8) && forth_data_push(session, -1);
    }
    if (strcmp(name, "MAX-CHAR") == 0) {
        return forth_data_push(session, 255) && forth_data_push(session, -1);
    }
    if (strcmp(name, "MAX-N") == 0) {
        return forth_data_push(session, INT64_MAX) && forth_data_push(session, -1);
    }
    if (strcmp(name, "MAX-U") == 0) {
        return forth_data_push(session, -1) && forth_data_push(session, -1);
    }
    if (strcmp(name, "FLOORED") == 0) {
        return forth_data_push(session, -1) && forth_data_push(session, -1);
    }
    return forth_data_push(session, 0);
}

static bool forth_um_mod(ForthSession *session) {
    int64_t n = 0;
    int64_t hi = 0;
    int64_t lo = 0;
    unsigned __int128 den;
    unsigned __int128 num;
    unsigned __int128 q;
    unsigned __int128 r;
    if (!forth_data_pop(session, &n) || n == 0) return false;
    if (!forth_data_pop(session, &hi)) return false;
    if (!forth_data_pop(session, &lo)) return false;
    den = (unsigned __int128)(uint64_t)n;
    num = ((unsigned __int128)(uint64_t)hi << 64) | (uint64_t)lo;
    q = num / den;
    r = num % den;
    return forth_data_push(session, (int64_t)(uint64_t)r)
        && forth_data_push(session, (int64_t)(uint64_t)q);
}

static bool forth_sm_rem(ForthSession *session) {
    int64_t n = 0;
    int64_t hi = 0;
    int64_t lo = 0;
    __int128 den;
    __int128 num;
    __int128 q;
    __int128 r;
    if (!forth_data_pop(session, &n) || n == 0) return false;
    if (!forth_data_pop(session, &hi)) return false;
    if (!forth_data_pop(session, &lo)) return false;
    den = (__int128)n;
    num = ((__int128)hi << 64) | (__int128)(uint64_t)lo;
    q = num / den;
    r = num % den;
    return forth_data_push(session, (int64_t)r) && forth_data_push(session, (int64_t)q);
}

static bool forth_fm_mod(ForthSession *session) {
    int64_t n = 0;
    int64_t hi = 0;
    int64_t lo = 0;
    __int128 den;
    __int128 num;
    __int128 q;
    __int128 r;
    if (!forth_data_pop(session, &n) || n == 0) return false;
    if (!forth_data_pop(session, &hi)) return false;
    if (!forth_data_pop(session, &lo)) return false;
    den = (__int128)n;
    num = ((__int128)hi << 64) | (__int128)(uint64_t)lo;
    q = num / den;
    r = num % den;
    if (r != 0 && ((num < 0) != (den < 0))) {
        q -= 1;
        r += den;
    }
    return forth_data_push(session, (int64_t)r) && forth_data_push(session, (int64_t)q);
}

static int64_t forth_base_value(ForthSession *session) {
    int64_t base = 10;
    if (!forth_fetch_cell(session, forth_base_addr(session), &base)) return 10;
    if (base < 2 || base > 36) return 10;
    return base;
}

static bool forth_pict_reset(ForthSession *session) {
    return forth_store_cell(session, session->hld_addr,
                            (int64_t)(session->hold_addr + FORTH_HOLD_MAX));
}

static bool forth_pict_hold(ForthSession *session, uint8_t ch) {
    int64_t hld = 0;
    if (!forth_fetch_cell(session, session->hld_addr, &hld)) return false;
    if (hld <= (int64_t)session->hold_addr) return false;
    hld--;
    if (!forth_store_byte(session, (uint64_t)hld, ch)) return false;
    return forth_store_cell(session, session->hld_addr, hld);
}

static bool forth_pict_hash(ForthSession *session) {
    int64_t hi = 0;
    int64_t lo = 0;
    int64_t base;
    unsigned __int128 num;
    unsigned __int128 den;
    unsigned __int128 q;
    unsigned __int128 r;
    uint8_t digit;
    base = forth_base_value(session);
    if (!forth_data_pop(session, &hi)) return false;
    if (!forth_data_pop(session, &lo)) return false;
    den = (unsigned __int128)(uint64_t)base;
    num = ((unsigned __int128)(uint64_t)hi << 64) | (uint64_t)lo;
    q = num / den;
    r = num % den;
    digit = (uint8_t)r;
    if (digit < 10) digit = (uint8_t)('0' + digit);
    else digit = (uint8_t)('A' + (digit - 10));
    if (!forth_pict_hold(session, digit)) return false;
    return forth_data_push(session, (int64_t)(uint64_t)q)
        && forth_data_push(session, (int64_t)(uint64_t)(q >> 64));
}

static bool forth_pict_end(ForthSession *session) {
    int64_t hi = 0;
    int64_t lo = 0;
    int64_t hld = 0;
    uint64_t addr;
    uint32_t len;
    if (!forth_data_pop(session, &hi)) return false;
    if (!forth_data_pop(session, &lo)) return false;
    (void)hi;
    (void)lo;
    if (!forth_fetch_cell(session, session->hld_addr, &hld)) return false;
    addr = (uint64_t)hld;
    len = (uint32_t)((session->hold_addr + FORTH_HOLD_MAX) - addr);
    return forth_data_push(session, (int64_t)addr)
        && forth_data_push(session, (int64_t)len);
}

static bool forth_host_create(ForthSession *session) {
    uint8_t def[FORTH_NAME_MAX];
    uint32_t deflen = 0;
    ForthNt published = 0;
    ForthHeader *header;
    uint64_t addr;
    int got;
    got = forth_take_word(session, def, &deflen);
    if (got <= 0) return false;
    if (!forth_dict_align(session)) return false;
    if (!forth_colon_begin(session, (const char *)def, deflen)) return false;
    addr = session->bump;
    if (!forth_colon_literal(session, (int64_t)addr)) {
        forth_colon_abort(session);
        return false;
    }
    if (!forth_colon_finish(session, &published)) return false;
    header = header_at(session, published);
    if (!header) return false;
    header->data_addr = addr;
    return true;
}

static bool forth_host_does(ForthSession *session) {
    int64_t does_xt = 0;
    ForthNt nt;
    ForthHeader *header;
    ForthXt child;
    uint8_t code[128];
    uint32_t off = 0;
    NvmModule *mod;
    NvmFunctionEntry *fn;
    NvmVerifyResult verified;

    if (!forth_data_pop(session, &does_xt)) return false;
    nt = forth_latest(session);
    header = header_at(session, nt);
    if (!header) return false;
    child = header->xt;
    mod = session->module;
    if ((uint32_t)does_xt >= mod->function_count) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_PUSH_I64, (int64_t)header->data_addr))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_CALL, session->dpush_fn))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_CALL, (uint32_t)does_xt))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_RET)) return false;
    fn = &mod->functions[child];
    fn->code_offset = nvm_append_code(mod, code, off);
    fn->code_length = off;
    fn->local_count = 0;
    fn->result_tag = TAG_VOID;
    fn->result_count = 0;
    verified = nvm_verify_function(mod, child);
    if (!verified.ok) return false;
    return forth_session_rebuild(session);
}

static bool forth_host_named_colon(ForthSession *session, int64_t literal,
                                  bool is_variable) {
    uint8_t def[FORTH_NAME_MAX];
    uint32_t deflen = 0;
    ForthNt published = 0;
    uint64_t addr;
    int got;

    got = forth_take_word(session, def, &deflen);
    if (got <= 0) return false;
    if (!forth_dict_align(session)) return false;
    if (is_variable) {
        if (!forth_colon_begin(session, (const char *)def, deflen)) return false;
        addr = session->bump;
        if (!forth_dict_allot(session, (int64_t)FORTH_CELL_BYTES)) {
            forth_colon_abort(session);
            return false;
        }
        if (!forth_store_cell(session, addr, 0)) {
            forth_colon_abort(session);
            return false;
        }
        if (!forth_colon_literal(session, (int64_t)addr)) {
            forth_colon_abort(session);
            return false;
        }
        if (!forth_colon_finish(session, &published)) return false;
        {
            ForthHeader *header = header_at(session, published);
            if (header) header->data_addr = addr;
        }
        return true;
    }
    if (!forth_colon_begin(session, (const char *)def, deflen)) return false;
    if (!forth_colon_literal(session, literal)) {
        forth_colon_abort(session);
        return false;
    }
    if (!forth_colon_finish(session, &published)) return false;
    return true;
}

static bool forth_colon_host_runtime(ForthSession *session, uint8_t host) {
    if (!session || !session->colon_open) return false;
    if (!colon_emit(session, OP_PUSH_I64, (int64_t)host)) return false;
    if (!colon_emit(session, OP_CALL_EXTERN, session->runtime_import)) return false;
    return colon_emit(session, OP_POP);
}

static bool forth_host_to_body(ForthSession *session) {
    int64_t cell = 0;
    ForthHeader *header;
    if (!forth_data_pop(session, &cell) || cell < 0) return false;
    header = header_by_xt(session, (ForthXt)cell);
    if (!header) return false;
    return forth_data_push(session, (int64_t)header->data_addr);
}

static bool forth_host_to_number(ForthSession *session) {
    int64_t u = 0;
    int64_t caddr = 0;
    int64_t hi = 0;
    int64_t lo = 0;
    int64_t base = 10;
    uint32_t consumed = 0;
    unsigned __int128 acc;

    if (!forth_data_pop(session, &u) || u < 0) return false;
    if (!forth_data_pop(session, &caddr)) return false;
    if (!forth_data_pop(session, &hi)) return false;
    if (!forth_data_pop(session, &lo)) return false;
    if (!forth_fetch_cell(session, forth_base_addr(session), &base)) return false;
    if (base < 2 || base > 36) return false;
    acc = ((unsigned __int128)(uint64_t)hi << 64) | (uint64_t)lo;
    while (consumed < (uint32_t)u) {
        uint8_t ch = 0;
        int digit;
        if (!forth_fetch_byte(session, (uint64_t)caddr + consumed, &ch)) return false;
        digit = forth_digit_value(ch);
        if (digit < 0 || (int64_t)digit >= base) break;
        acc = acc * (unsigned __int128)(uint64_t)base + (unsigned)digit;
        consumed++;
    }
    lo = (int64_t)(uint64_t)acc;
    hi = (int64_t)(uint64_t)(acc >> 64);
    if (!forth_data_push(session, lo)) return false;
    if (!forth_data_push(session, hi)) return false;
    if (!forth_data_push(session, (int64_t)((uint64_t)caddr + consumed))) return false;
    return forth_data_push(session, (int64_t)((uint32_t)u - consumed));
}

static int forth_host_postpone(ForthSession *session) {
    uint8_t name[FORTH_NAME_MAX];
    uint32_t nlen = 0;
    ForthNt nt = 0;
    ForthXt xt = 0;
    ForthXt comma_xt = 0;
    ForthNt comma_nt = 0;
    bool immediate = false;
    bool comma_imm = false;
    ForthHeader *header;
    int got;

    if (!session->colon_open) return -1;
    got = forth_take_word(session, name, &nlen);
    if (got <= 0) return -1;
    if (!forth_find(session, (const char *)name, nlen, &nt, &xt, &immediate))
        return -1;
    header = header_at(session, nt);
    if (immediate) {
        if (header && header->host_kind != FORTH_HOST_NONE)
            return forth_colon_host_runtime(session, header->host_kind) ? 1 : -1;
        return forth_colon_call(session, xt) ? 1 : -1;
    }
    if (!forth_find(session, "COMPILE,", 8, &comma_nt, &comma_xt, &comma_imm))
        return -1;
    if (!forth_colon_literal(session, (int64_t)xt)) return -1;
    return forth_colon_call(session, comma_xt) ? 1 : -1;
}

static bool forth_host_compile_comma(ForthSession *session) {
    int64_t cell = 0;
    if (!session->colon_open) return false;
    if (!forth_data_pop(session, &cell) || cell < 0) return false;
    return forth_colon_call(session, (ForthXt)cell);
}

static int forth_host_abort_quote(ForthSession *session, int64_t state) {
    uint64_t src = 0;
    uint32_t wlen = 0;
    int64_t flag = 0;
    ForthNt throw_nt = 0;
    ForthXt throw_xt = 0;
    bool imm = false;

    if (!forth_skip_blanks(session)) return -1;
    if (!forth_parse_delimited(session, (uint8_t)'"', false, &src, &wlen))
        return -1;
    if (state != 0) {
        uint64_t dest = session->bump;
        uint32_t i;
        ForthNt type_nt = 0;
        ForthXt type_xt = 0;
        if (!forth_dict_allot(session, (int64_t)wlen)) return -1;
        for (i = 0; i < wlen; i++) {
            uint8_t ch = 0;
            if (!forth_fetch_byte(session, src + i, &ch)) return -1;
            if (!forth_store_byte(session, dest + i, ch)) return -1;
        }
        if (!forth_colon_if(session)) return -1;
        if (!forth_colon_literal(session, (int64_t)dest)) return -1;
        if (!forth_colon_literal(session, (int64_t)wlen)) return -1;
        if (!forth_find(session, "TYPE", 4, &type_nt, &type_xt, &imm)) return -1;
        if (!forth_colon_call(session, type_xt)) return -1;
        if (!forth_colon_literal(session, -2)) return -1;
        if (!forth_find(session, "THROW", 5, &throw_nt, &throw_xt, &imm)) return -1;
        if (!forth_colon_call(session, throw_xt)) return -1;
        return forth_colon_then(session) ? 1 : -1;
    }
    if (!forth_data_pop(session, &flag)) return -1;
    if (flag == 0) return 1;
    if (!forth_type_range(session, src, wlen)) return -1;
    while (forth_data_depth(session) > 0) {
        int64_t drop = 0;
        forth_data_pop(session, &drop);
    }
    forth_store_cell(session, session->throw_code_addr, -2);
    return -1;
}

static int forth_host_key(ForthSession *session) {
    uint64_t caddr = 0;
    uint64_t u = 0;
    int64_t to_in = 0;
    uint8_t ch = 0;

    if (!forth_source(session, &caddr, &u)) return -1;
    if (!forth_fetch_cell(session, session->sysvars, &to_in)) return -1;
    if (to_in < 0 || (uint64_t)to_in >= u) return -1;
    if (!forth_fetch_byte(session, caddr + (uint64_t)to_in, &ch)) return -1;
    to_in++;
    if (!forth_store_cell(session, session->sysvars, to_in)) return -1;
    return forth_data_push(session, (int64_t)ch) ? 1 : -1;
}

static int forth_host_accept(ForthSession *session) {
    int64_t n1 = 0;
    int64_t dest = 0;
    uint64_t caddr = 0;
    uint64_t u = 0;
    int64_t to_in = 0;
    uint32_t n2 = 0;

    if (!forth_data_pop(session, &n1) || n1 < 0) return -1;
    if (!forth_data_pop(session, &dest)) return -1;
    if (!forth_source(session, &caddr, &u)) return -1;
    if (!forth_fetch_cell(session, session->sysvars, &to_in)) return -1;
    if (to_in < 0) return -1;
    while ((uint64_t)to_in < u && n2 < (uint32_t)n1) {
        uint8_t ch = 0;
        if (!forth_fetch_byte(session, caddr + (uint64_t)to_in, &ch)) return -1;
        if (!forth_store_byte(session, (uint64_t)dest + n2, ch)) return -1;
        to_in++;
        n2++;
    }
    if (!forth_store_cell(session, session->sysvars, to_in)) return -1;
    return forth_data_push(session, (int64_t)n2) ? 1 : -1;
}

static int forth_host_quit(ForthSession *session) {
    while (forth_source_id(session) != 0) {
        if (!forth_source_pop(session)) return -1;
    }
    if (!forth_store_cell(session, session->ret_depth_addr, 0)) return -1;
    if (!forth_store_cell(session, forth_state_addr(session), 0)) return -1;
    session->quit_requested = true;
    return 1;
}

static int forth_run_host(ForthSession *session, uint8_t host, int64_t state) {
    uint8_t name[FORTH_NAME_MAX];
    uint32_t nlen = 0;
    ForthNt nt = 0;
    ForthXt xt = 0;
    bool immediate = false;
    int64_t cell = 0;
    ForthNt published = 0;
    int got;

    if (!session) return -1;
    switch (host) {
    case FORTH_HOST_NONE:
        return 0;
    case FORTH_HOST_COLON:
        if (forth_colon_is_open(session)) return -1;
        got = forth_take_word(session, name, &nlen);
        if (got <= 0) return -1;
        if (!forth_colon_begin(session, (const char *)name, nlen)) return -1;
        if (!forth_store_cell(session, forth_state_addr(session), 1)) return -1;
        return 1;
    case FORTH_HOST_SEMI:
        if (!forth_colon_is_open(session)) return -1;
        if (!forth_colon_finish(session, &published)) return -1;
        if (!forth_store_cell(session, forth_state_addr(session), 0)) return -1;
        return 1;
    case FORTH_HOST_IF:
        return forth_colon_if(session) ? 1 : -1;
    case FORTH_HOST_ELSE:
        return forth_colon_else(session) ? 1 : -1;
    case FORTH_HOST_THEN:
        return forth_colon_then(session) ? 1 : -1;
    case FORTH_HOST_BEGIN:
        return forth_colon_cs_begin(session) ? 1 : -1;
    case FORTH_HOST_UNTIL:
        return forth_colon_until(session) ? 1 : -1;
    case FORTH_HOST_AGAIN:
        return forth_colon_again(session) ? 1 : -1;
    case FORTH_HOST_WHILE:
        return forth_colon_while(session) ? 1 : -1;
    case FORTH_HOST_REPEAT:
        return forth_colon_repeat(session) ? 1 : -1;
    case FORTH_HOST_DO:
        return forth_colon_do(session) ? 1 : -1;
    case FORTH_HOST_LOOP:
        return forth_colon_loop(session) ? 1 : -1;
    case FORTH_HOST_PLUS_LOOP:
        return forth_colon_plus_loop(session) ? 1 : -1;
    case FORTH_HOST_RECURSE:
        return forth_colon_recurse(session) ? 1 : -1;
    case FORTH_HOST_LBRACKET:
        return forth_store_cell(session, forth_state_addr(session), 0) ? 1 : -1;
    case FORTH_HOST_RBRACKET:
        if (!forth_colon_is_open(session)) return -1;
        return forth_store_cell(session, forth_state_addr(session), 1) ? 1 : -1;
    case FORTH_HOST_LITERAL:
        if (state == 0) return -1;
        if (!forth_data_pop(session, &cell)) return -1;
        return forth_colon_literal(session, cell) ? 1 : -1;
    case FORTH_HOST_IMMEDIATE:
        if (forth_latest(session) == 0) return -1;
        return forth_mark_immediate(session, forth_latest(session)) ? 1 : -1;
    case FORTH_HOST_TICK:
        got = forth_take_word(session, name, &nlen);
        if (got <= 0) return -1;
        if (!forth_find(session, (const char *)name, nlen, &nt, &xt, &immediate))
            return -1;
        if (state != 0)
            return forth_colon_literal(session, (int64_t)xt) ? 1 : -1;
        return forth_data_push(session, (int64_t)xt) ? 1 : -1;
    case FORTH_HOST_BRACKET_TICK:
        if (state == 0) return -1;
        got = forth_take_word(session, name, &nlen);
        if (got <= 0) return -1;
        if (!forth_find(session, (const char *)name, nlen, &nt, &xt, &immediate))
            return -1;
        return forth_colon_literal(session, (int64_t)xt) ? 1 : -1;
    case FORTH_HOST_CHAR:
        if (state != 0) return 0;
        got = forth_take_word(session, name, &nlen);
        if (got <= 0 || nlen == 0) return -1;
        return forth_data_push(session, (int64_t)name[0]) ? 1 : -1;
    case FORTH_HOST_BRACKET_CHAR:
        if (state == 0) return -1;
        got = forth_take_word(session, name, &nlen);
        if (got <= 0 || nlen == 0) return -1;
        return forth_colon_literal(session, (int64_t)name[0]) ? 1 : -1;
    case FORTH_HOST_CONSTANT:
        if (state != 0) return 0;
        if (!forth_data_pop(session, &cell)) return -1;
        return forth_host_named_colon(session, cell, false) ? 1 : -1;
    case FORTH_HOST_VARIABLE:
        if (state != 0) return 0;
        return forth_host_named_colon(session, 0, true) ? 1 : -1;
    case FORTH_HOST_ALLOT:
        if (state != 0) return 0;
        if (!forth_data_pop(session, &cell)) return -1;
        return forth_dict_allot(session, cell) ? 1 : -1;
    case FORTH_HOST_COMMA:
        if (state != 0) return 0;
        if (!forth_data_pop(session, &cell)) return -1;
        if (!forth_dict_align(session)) return -1;
        {
            uint64_t addr = session->bump;
            if (!forth_dict_allot(session, (int64_t)FORTH_CELL_BYTES)) return -1;
            return forth_store_cell(session, addr, cell) ? 1 : -1;
        }
    case FORTH_HOST_ALIGN:
        if (state != 0) return 0;
        return forth_dict_align(session) ? 1 : -1;
    case FORTH_HOST_EXECUTE:
        if (state != 0) return 0;
        if (!forth_data_pop(session, &cell)) return -1;
        if (cell < 0) return -1;
        xt = (ForthXt)cell;
        {
            ForthHeader *exec_hdr = header_by_xt(session, xt);
            if (exec_hdr && exec_hdr->compile_only) return -1;
        }
        return forth_invoke_nested(session, xt) == VM_OK ? 1 : -1;
    case FORTH_HOST_BACKSLASH:
        return forth_skip_until(session, 0, true) ? 1 : -1;
    case FORTH_HOST_PAREN:
        return forth_skip_until(session, (uint8_t)')', false) ? 1 : -1;
    case FORTH_HOST_QDO:
        return forth_colon_qdo(session) ? 1 : -1;
    case FORTH_HOST_LEAVE:
        return forth_colon_leave(session) ? 1 : -1;
    case FORTH_HOST_EXIT:
        if (state == 0) return -1;
        return forth_colon_exit(session) ? 1 : -1;
    case FORTH_HOST_CREATE:
        if (state != 0) return 0;
        return forth_host_create(session) ? 1 : -1;
    case FORTH_HOST_DOES:
        if (state != 0) {
            session->colon_does_pending = true;
            session->colon_does_off = session->colon_code_len;
            return 1;
        }
        return forth_host_does(session) ? 1 : -1;
    case FORTH_HOST_SOURCE:
        if (state != 0) return 0;
        {
            uint64_t caddr = 0;
            uint64_t u = 0;
            if (!forth_source(session, &caddr, &u)) return -1;
            if (!forth_data_push(session, (int64_t)caddr)) return -1;
            return forth_data_push(session, (int64_t)u) ? 1 : -1;
        }
    case FORTH_HOST_EVALUATE:
        if (state != 0) return 0;
        {
            int64_t u = 0;
            int64_t caddr = 0;
            if (!forth_data_pop(session, &u) || u < 0) return -1;
            if (!forth_data_pop(session, &caddr)) return -1;
            if (!forth_source_push_evaluate(session, (uint64_t)caddr, (uint64_t)u))
                return -1;
            if (!forth_interpret_loop(session)) {
                forth_source_pop(session);
                return -1;
            }
            return forth_source_pop(session) ? 1 : -1;
        }
    case FORTH_HOST_FIND:
        if (state != 0) return 0;
        if (!forth_data_pop(session, &cell)) return -1;
        {
            uint8_t nlen_b = 0;
            uint32_t nlen_w = 0;
            uint8_t wname[FORTH_NAME_MAX];
            uint32_t i;
            ForthXt found_xt = 0;
            bool imm = false;
            if (!forth_fetch_byte(session, (uint64_t)cell, &nlen_b)) return -1;
            nlen_w = nlen_b;
            if (nlen_w > FORTH_NAME_MAX) return -1;
            for (i = 0; i < nlen_w; i++) {
                if (!forth_fetch_byte(session, (uint64_t)cell + 1 + i, &wname[i]))
                    return -1;
            }
            if (forth_find(session, (const char *)wname, nlen_w, &nt, &found_xt,
                           &imm)) {
                if (!forth_data_push(session, (int64_t)found_xt)) return -1;
                return forth_data_push(session, imm ? 1 : -1) ? 1 : -1;
            }
            if (!forth_data_push(session, cell)) return -1;
            return forth_data_push(session, 0) ? 1 : -1;
        }
    case FORTH_HOST_WORD:
        if (state != 0) return 0;
        if (!forth_data_pop(session, &cell)) return -1;
        {
            uint64_t src = 0;
            uint32_t wlen = 0;
            if (!forth_parse_delimited(session, (uint8_t)cell, true, &src, &wlen))
                return -1;
            if (!forth_copy_to_word(session, src, wlen, true)) return -1;
            return forth_data_push(session, (int64_t)session->word_addr) ? 1 : -1;
        }
    case FORTH_HOST_PARSE:
        if (state != 0) return 0;
        if (!forth_data_pop(session, &cell)) return -1;
        {
            uint64_t src = 0;
            uint32_t wlen = 0;
            if (!forth_parse_delimited(session, (uint8_t)cell, false, &src, &wlen))
                return -1;
            if (!forth_data_push(session, (int64_t)src)) return -1;
            return forth_data_push(session, (int64_t)wlen) ? 1 : -1;
        }
    case FORTH_HOST_S_QUOTE:
        {
            uint64_t src = 0;
            uint32_t wlen = 0;
            uint64_t dest;
            uint32_t i;
            if (!forth_skip_blanks(session)) return -1;
            if (!forth_parse_delimited(session, (uint8_t)'"', false, &src, &wlen))
                return -1;
            if (state != 0) {
                dest = session->bump;
                if (!forth_dict_allot(session, (int64_t)wlen)) return -1;
                for (i = 0; i < wlen; i++) {
                    uint8_t ch = 0;
                    if (!forth_fetch_byte(session, src + i, &ch)) return -1;
                    if (!forth_store_byte(session, dest + i, ch)) return -1;
                }
                if (!forth_colon_literal(session, (int64_t)dest)) return -1;
                return forth_colon_literal(session, (int64_t)wlen) ? 1 : -1;
            }
            if (!forth_copy_to_word(session, src, wlen, false)) return -1;
            if (!forth_data_push(session, (int64_t)session->word_addr)) return -1;
            return forth_data_push(session, (int64_t)wlen) ? 1 : -1;
        }
    case FORTH_HOST_DOT_QUOTE:
        {
            uint64_t src = 0;
            uint32_t wlen = 0;
            if (!forth_skip_blanks(session)) return -1;
            if (!forth_parse_delimited(session, (uint8_t)'"', false, &src, &wlen))
                return -1;
            if (state != 0) {
                uint64_t dest = session->bump;
                uint32_t i;
                ForthNt type_nt = 0;
                ForthXt type_xt = 0;
                bool imm = false;
                if (!forth_dict_allot(session, (int64_t)wlen)) return -1;
                for (i = 0; i < wlen; i++) {
                    uint8_t ch = 0;
                    if (!forth_fetch_byte(session, src + i, &ch)) return -1;
                    if (!forth_store_byte(session, dest + i, ch)) return -1;
                }
                if (!forth_colon_literal(session, (int64_t)dest)) return -1;
                if (!forth_colon_literal(session, (int64_t)wlen)) return -1;
                if (!forth_find(session, "TYPE", 4, &type_nt, &type_xt, &imm))
                    return -1;
                return forth_colon_call(session, type_xt) ? 1 : -1;
            }
            return forth_type_range(session, src, wlen) ? 1 : -1;
        }
    case FORTH_HOST_EMIT:
        if (state != 0) return 0;
        if (!forth_data_pop(session, &cell)) return -1;
        return forth_emit_char(session, (uint8_t)cell) ? 1 : -1;
    case FORTH_HOST_TYPE:
        if (state != 0) return 0;
        {
            int64_t u = 0;
            int64_t caddr = 0;
            if (!forth_data_pop(session, &u) || u < 0) return -1;
            if (!forth_data_pop(session, &caddr)) return -1;
            return forth_type_range(session, (uint64_t)caddr, (uint32_t)u) ? 1 : -1;
        }
    case FORTH_HOST_CR:
        if (state != 0) return 0;
        return forth_emit_char(session, (uint8_t)'\n') ? 1 : -1;
    case FORTH_HOST_ENVIRONMENT:
        if (state != 0) return 0;
        {
            int64_t u = 0;
            int64_t caddr = 0;
            if (!forth_data_pop(session, &u) || u < 0) return -1;
            if (!forth_data_pop(session, &caddr)) return -1;
            return forth_env_query(session, (uint64_t)caddr, (uint32_t)u) ? 1 : -1;
        }
    case FORTH_HOST_ABORT:
        if (state != 0) return 0;
        while (forth_data_depth(session) > 0) {
            int64_t drop = 0;
            forth_data_pop(session, &drop);
        }
        forth_store_cell(session, session->throw_code_addr, -1);
        return -1;
    case FORTH_HOST_CATCH:
        if (state != 0) return 0;
        if (!forth_data_pop(session, &cell)) return -1;
        if (cell < 0) return -1;
        {
            ForthXt catch_xt = (ForthXt)cell;
            int64_t code = 0;
            if (!forth_catch(session, catch_xt, &code)) return -1;
            return forth_data_push(session, code) ? 1 : -1;
        }
    case FORTH_HOST_BYE:
        if (state != 0) return 0;
        session->exit_requested = true;
        return 1;
    case FORTH_HOST_UM_MOD:
        if (state != 0) return 0;
        return forth_um_mod(session) ? 1 : -1;
    case FORTH_HOST_SM_REM:
        if (state != 0) return 0;
        return forth_sm_rem(session) ? 1 : -1;
    case FORTH_HOST_FM_MOD:
        if (state != 0) return 0;
        return forth_fm_mod(session) ? 1 : -1;
    case FORTH_HOST_FILL:
        if (state != 0) return 0;
        {
            int64_t ch = 0;
            int64_t u = 0;
            int64_t addr = 0;
            int64_t i;
            if (!forth_data_pop(session, &ch)) return -1;
            if (!forth_data_pop(session, &u) || u < 0) return -1;
            if (!forth_data_pop(session, &addr)) return -1;
            for (i = 0; i < u; i++) {
                if (!forth_store_byte(session, (uint64_t)addr + (uint64_t)i,
                                      (uint8_t)ch))
                    return -1;
            }
            return 1;
        }
    case FORTH_HOST_MOVE:
        if (state != 0) return 0;
        {
            int64_t u = 0;
            int64_t dest = 0;
            int64_t src = 0;
            int64_t i;
            if (!forth_data_pop(session, &u) || u < 0) return -1;
            if (u == 0) return 1;
            if (!forth_data_pop(session, &dest)) return -1;
            if (!forth_data_pop(session, &src)) return -1;
            if ((uint64_t)dest < (uint64_t)src) {
                for (i = 0; i < u; i++) {
                    uint8_t ch = 0;
                    if (!forth_fetch_byte(session, (uint64_t)src + (uint64_t)i, &ch))
                        return -1;
                    if (!forth_store_byte(session, (uint64_t)dest + (uint64_t)i, ch))
                        return -1;
                }
            } else {
                for (i = u - 1; i >= 0; i--) {
                    uint8_t ch = 0;
                    if (!forth_fetch_byte(session, (uint64_t)src + (uint64_t)i, &ch))
                        return -1;
                    if (!forth_store_byte(session, (uint64_t)dest + (uint64_t)i, ch))
                        return -1;
                }
            }
            return 1;
        }
    case FORTH_HOST_C_COMMA:
        if (state != 0) return 0;
        if (!forth_data_pop(session, &cell)) return -1;
        {
            uint64_t addr = session->bump;
            if (!forth_dict_allot(session, 1)) return -1;
            return forth_store_byte(session, addr, (uint8_t)cell) ? 1 : -1;
        }
    case FORTH_HOST_PICK:
        if (state != 0) return 0;
        if (!forth_data_pop(session, &cell) || cell < 0) return -1;
        {
            uint32_t depth = forth_data_depth(session);
            int64_t picked = 0;
            uint64_t addr;
            if ((uint64_t)cell >= depth) return -1;
            addr = session->data_stack_addr
                + (uint64_t)(depth - 1 - (uint32_t)cell) * FORTH_CELL_BYTES;
            if (!forth_fetch_cell(session, addr, &picked)) return -1;
            return forth_data_push(session, picked) ? 1 : -1;
        }
    case FORTH_HOST_ROLL:
        if (state != 0) return 0;
        if (!forth_data_pop(session, &cell) || cell < 0) return -1;
        {
            uint32_t depth = forth_data_depth(session);
            uint32_t n = (uint32_t)cell;
            uint32_t i;
            int64_t picked = 0;
            if (n == 0) return 1;
            if (n >= depth) return -1;
            {
                uint64_t addr = session->data_stack_addr
                    + (uint64_t)(depth - 1 - n) * FORTH_CELL_BYTES;
                if (!forth_fetch_cell(session, addr, &picked)) return -1;
                for (i = depth - 1 - n; i + 1 < depth; i++) {
                    int64_t v = 0;
                    uint64_t from = session->data_stack_addr
                        + (uint64_t)(i + 1) * FORTH_CELL_BYTES;
                    uint64_t to = session->data_stack_addr
                        + (uint64_t)i * FORTH_CELL_BYTES;
                    if (!forth_fetch_cell(session, from, &v)) return -1;
                    if (!forth_store_cell(session, to, v)) return -1;
                }
                {
                    uint64_t tos = session->data_stack_addr
                        + (uint64_t)(depth - 1) * FORTH_CELL_BYTES;
                    if (!forth_store_cell(session, tos, picked)) return -1;
                }
            }
            return 1;
        }
    case FORTH_HOST_LESS_NUM:
        if (state != 0) return 0;
        return forth_pict_reset(session) ? 1 : -1;
    case FORTH_HOST_HOLD:
        if (state != 0) return 0;
        if (!forth_data_pop(session, &cell)) return -1;
        return forth_pict_hold(session, (uint8_t)cell) ? 1 : -1;
    case FORTH_HOST_SIGN:
        if (state != 0) return 0;
        if (!forth_data_pop(session, &cell)) return -1;
        if (cell < 0) return forth_pict_hold(session, (uint8_t)'-') ? 1 : -1;
        return 1;
    case FORTH_HOST_HASH:
        if (state != 0) return 0;
        return forth_pict_hash(session) ? 1 : -1;
    case FORTH_HOST_HASH_S:
        if (state != 0) return 0;
        for (;;) {
            int64_t hi = 0;
            int64_t lo = 0;
            uint32_t depth = forth_data_depth(session);
            if (depth < 2) return -1;
            if (!forth_pict_hash(session)) return -1;
            if (!forth_data_pop(session, &hi)) return -1;
            if (!forth_data_pop(session, &lo)) return -1;
            if (hi == 0 && lo == 0) {
                if (!forth_data_push(session, 0)) return -1;
                return forth_data_push(session, 0) ? 1 : -1;
            }
            if (!forth_data_push(session, lo)) return -1;
            if (!forth_data_push(session, hi)) return -1;
        }
    case FORTH_HOST_NUM_END:
        if (state != 0) return 0;
        return forth_pict_end(session) ? 1 : -1;
    case FORTH_HOST_DOT:
        if (state != 0) return 0;
        if (!forth_data_pop(session, &cell)) return -1;
        {
            int64_t mag = cell < 0 ? -cell : cell;
            if (cell < 0 && !forth_emit_char(session, (uint8_t)'-')) return -1;
            if (!forth_data_push(session, mag)) return -1;
            if (!forth_data_push(session, 0)) return -1;
            if (!forth_pict_reset(session)) return -1;
            for (;;) {
                int64_t hi = 0;
                int64_t lo = 0;
                if (!forth_pict_hash(session)) return -1;
                if (!forth_data_pop(session, &hi)) return -1;
                if (!forth_data_pop(session, &lo)) return -1;
                if (hi == 0 && lo == 0) break;
                if (!forth_data_push(session, lo) || !forth_data_push(session, hi))
                    return -1;
            }
            if (!forth_data_push(session, 0) || !forth_data_push(session, 0))
                return -1;
            if (!forth_pict_end(session)) return -1;
            {
                int64_t u = 0;
                int64_t caddr = 0;
                if (!forth_data_pop(session, &u) || !forth_data_pop(session, &caddr))
                    return -1;
                if (!forth_type_range(session, (uint64_t)caddr, (uint32_t)u))
                    return -1;
            }
            return forth_emit_char(session, (uint8_t)' ') ? 1 : -1;
        }
    case FORTH_HOST_UDOT:
        if (state != 0) return 0;
        if (!forth_data_pop(session, &cell)) return -1;
        if (!forth_data_push(session, cell) || !forth_data_push(session, 0))
            return -1;
        if (!forth_pict_reset(session)) return -1;
        for (;;) {
            int64_t hi = 0;
            int64_t lo = 0;
            if (!forth_pict_hash(session)) return -1;
            if (!forth_data_pop(session, &hi)) return -1;
            if (!forth_data_pop(session, &lo)) return -1;
            if (hi == 0 && lo == 0) break;
            if (!forth_data_push(session, lo) || !forth_data_push(session, hi))
                return -1;
        }
        if (!forth_data_push(session, 0) || !forth_data_push(session, 0))
            return -1;
        if (!forth_pict_end(session)) return -1;
        {
            int64_t u = 0;
            int64_t caddr = 0;
            if (!forth_data_pop(session, &u) || !forth_data_pop(session, &caddr))
                return -1;
            if (!forth_type_range(session, (uint64_t)caddr, (uint32_t)u))
                return -1;
        }
        return forth_emit_char(session, (uint8_t)' ') ? 1 : -1;
    case FORTH_HOST_TO_BODY:
        if (state != 0) return 0;
        return forth_host_to_body(session) ? 1 : -1;
    case FORTH_HOST_TO_NUMBER:
        if (state != 0) return 0;
        return forth_host_to_number(session) ? 1 : -1;
    case FORTH_HOST_POSTPONE:
        if (state == 0) return -1;
        return forth_host_postpone(session);
    case FORTH_HOST_COMPILE_COMMA:
        if (state != 0) return 0;
        return forth_host_compile_comma(session) ? 1 : -1;
    case FORTH_HOST_ABORT_QUOTE:
        return forth_host_abort_quote(session, state);
    case FORTH_HOST_ACCEPT:
        if (state != 0) return 0;
        return forth_host_accept(session);
    case FORTH_HOST_KEY:
        if (state != 0) return 0;
        return forth_host_key(session);
    case FORTH_HOST_QUIT:
        if (state != 0) return 0;
        return forth_host_quit(session);
    default:
        return -1;
    }
}

static bool forth_throw_pending(ForthSession *session) {
    int64_t thrown = 0;
    if (!session) return false;
    if (!forth_fetch_cell(session, session->throw_code_addr, &thrown)) return true;
    return thrown != 0;
}

static bool forth_interpret_loop(ForthSession *session) {
    uint8_t name[FORTH_NAME_MAX];
    uint32_t nlen = 0;

    for (;;) {
        ForthNt nt = 0;
        ForthXt xt = 0;
        bool immediate = false;
        int64_t number = 0;
        int64_t state = 0;
        int got;
        int host_rc;
        ForthHeader *header;
        VmResult ran;

        got = forth_take_word(session, name, &nlen);
        if (got < 0) return false;
        if (got == 0) return true;
        if (!forth_fetch_cell(session, forth_state_addr(session), &state))
            return false;
        if (forth_find(session, (const char *)name, nlen, &nt, &xt, &immediate)) {
            header = header_at(session, nt);
            host_rc = forth_run_host(session,
                                     header ? header->host_kind : FORTH_HOST_NONE,
                                     state);
            if (host_rc < 0) return false;
            if (session->exit_requested || session->quit_requested) return true;
            if (host_rc == 0 && header && header->host_kind != FORTH_HOST_NONE
                    && state == 0)
                return false;
            if (host_rc > 0) continue;
            if (state != 0 && !immediate) {
                if (!forth_colon_call(session, xt)) return false;
            } else {
                if (header && header->compile_only && state == 0) return false;
                ran = forth_session_invoke(session, xt, NULL, 0, NULL);
                if (ran != VM_OK) return false;
                if (session->exit_requested || session->quit_requested) return true;
                if (forth_throw_pending(session)) return false;
            }
            continue;
        }
        if (!forth_parse_number(session, name, nlen, &number)) return false;
        if (state != 0) {
            if (!forth_colon_literal(session, number)) return false;
        } else if (!forth_data_push(session, number)) {
            return false;
        }
    }
}

bool forth_exit_requested(const ForthSession *session) {
    return session != NULL && session->exit_requested;
}

bool forth_interpret(ForthSession *session, const uint8_t *text, uint32_t len) {
    ForthSession *prev;
    bool ok;
    if (!session || (len > 0 && text == NULL)) return false;
    if (!forth_store_cell(session, session->throw_code_addr, 0)) return false;
    session->quit_requested = false;
    if (!forth_source_load_terminal(session, text, len)) return false;
    prev = g_forth;
    g_forth = session;
    ok = forth_interpret_loop(session);
    g_forth = prev;
    return ok;
}

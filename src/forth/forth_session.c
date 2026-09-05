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
#include <errno.h>
#include <float.h>
#include <math.h>
#include <sys/stat.h>
#include <unistd.h>

#define FORTH_ADDR_INITIAL 65536u
#define FORTH_ADDR_MAX (16u * 1024u * 1024u)
#define FORTH_HEAP_BASE (8u * 1024u * 1024u)
#define FORTH_FILE_SLOTS 32
#define FORTH_PATH_MAX 1024
#define FORTH_REQUIRED_MAX 64
#define FORTH_FAM_RO 1
#define FORTH_FAM_WO 2
#define FORTH_FAM_RW 3
#define FORTH_FAM_BIN 8
#define FORTH_REGION_INITIAL 16
#define FORTH_SUBST_MAX 32
#define FORTH_SUBST_NAME_MAX 64
#define FORTH_LOCAL_MAX 16

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
    char path[FORTH_PATH_MAX];
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
    uint16_t host_kind;
    uint8_t body_cells;
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
    FORTH_HOST_QUIT,
    FORTH_HOST_DOES_ENTER,
    FORTH_HOST_DOT_PAREN,
    FORTH_HOST_PLUSLOOP,
    FORTH_HOST_NONAME,
    FORTH_HOST_SOURCE_ID,
    FORTH_HOST_REFILL,
    FORTH_HOST_PARSE_NAME,
    FORTH_HOST_VALUE,
    FORTH_HOST_TO,
    FORTH_HOST_MARKER,
    FORTH_HOST_MARKER_RUN,
    FORTH_HOST_CASE,
    FORTH_HOST_OF,
    FORTH_HOST_ENDOF,
    FORTH_HOST_ENDCASE,
    FORTH_HOST_C_QUOTE,
    FORTH_HOST_S_BACKSLASH,
    FORTH_HOST_DOT_R,
    FORTH_HOST_UDOT_R,
    FORTH_HOST_HOLDS,
    FORTH_HOST_UNUSED,
    FORTH_HOST_SAVE_INPUT,
    FORTH_HOST_RESTORE_INPUT,
    FORTH_HOST_IS,
    FORTH_HOST_ACTION_OF,
    FORTH_HOST_TWO_VALUE,
    FORTH_HOST_DPLUS,
    FORTH_HOST_DMINUS,
    FORTH_HOST_DNEGATE,
    FORTH_HOST_DTWO_STAR,
    FORTH_HOST_DTWO_SLASH,
    FORTH_HOST_DLESS,
    FORTH_HOST_DEQUAL,
    FORTH_HOST_DABS,
    FORTH_HOST_DMAX,
    FORTH_HOST_DMIN,
    FORTH_HOST_MPLUS,
    FORTH_HOST_DULESS,
    FORTH_HOST_M_STAR_SLASH,
    FORTH_HOST_TWO_LITERAL,
    FORTH_HOST_TRAILING,
    FORTH_HOST_CMOVE,
    FORTH_HOST_CMOVE_UP,
    FORTH_HOST_COMPARE,
    FORTH_HOST_SEARCH,
    FORTH_HOST_SLITERAL,
    FORTH_HOST_UNESCAPE,
    FORTH_HOST_REPLACES,
    FORTH_HOST_SUBSTITUTE,
    FORTH_HOST_WORDLIST,
    FORTH_HOST_GET_ORDER,
    FORTH_HOST_SET_ORDER,
    FORTH_HOST_GET_CURRENT,
    FORTH_HOST_SET_CURRENT,
    FORTH_HOST_FORTH_WORDLIST,
    FORTH_HOST_ALSO,
    FORTH_HOST_PREVIOUS,
    FORTH_HOST_ONLY,
    FORTH_HOST_FORTH,
    FORTH_HOST_DEFINITIONS,
    FORTH_HOST_SEARCH_WORDLIST,
    FORTH_HOST_ORDER,
    FORTH_HOST_BIN,
    FORTH_HOST_OPEN_FILE,
    FORTH_HOST_CREATE_FILE,
    FORTH_HOST_CLOSE_FILE,
    FORTH_HOST_DELETE_FILE,
    FORTH_HOST_READ_FILE,
    FORTH_HOST_READ_LINE,
    FORTH_HOST_WRITE_FILE,
    FORTH_HOST_WRITE_LINE,
    FORTH_HOST_FILE_POSITION,
    FORTH_HOST_FILE_SIZE,
    FORTH_HOST_REPOSITION_FILE,
    FORTH_HOST_RESIZE_FILE,
    FORTH_HOST_FLUSH_FILE,
    FORTH_HOST_RENAME_FILE,
    FORTH_HOST_FILE_STATUS,
    FORTH_HOST_INCLUDED,
    FORTH_HOST_INCLUDE,
    FORTH_HOST_INCLUDE_FILE,
    FORTH_HOST_REQUIRED,
    FORTH_HOST_REQUIRE,
    FORTH_HOST_ALLOCATE,
    FORTH_HOST_MEM_FREE,
    FORTH_HOST_MEM_RESIZE,
    FORTH_HOST_LOCALS_BRACE,
    FORTH_HOST_LOCAL,
    FORTH_HOST_DOT_S,
    FORTH_HOST_AHEAD,
    FORTH_HOST_BRACKET_IF,
    FORTH_HOST_BRACKET_ELSE,
    FORTH_HOST_BRACKET_THEN,
    FORTH_HOST_CS_PICK,
    FORTH_HOST_CS_ROLL,
    FORTH_HOST_DEFINED,
    FORTH_HOST_UNDEFINED,
    FORTH_HOST_N_TO_R,
    FORTH_HOST_NR_FROM,
    FORTH_HOST_SYNONYM,
    FORTH_HOST_TRAVERSE_WORDLIST,
    FORTH_HOST_NAME_TO_COMPILE,
    FORTH_HOST_NAME_TO_INTERPRET,
    FORTH_HOST_NAME_TO_STRING,
    FORTH_HOST_D_TO_F,
    FORTH_HOST_F_TO_D,
    FORTH_HOST_FDEPTH,
    FORTH_HOST_FDROP,
    FORTH_HOST_FDUP,
    FORTH_HOST_FSWAP,
    FORTH_HOST_FOVER,
    FORTH_HOST_FROT,
    FORTH_HOST_FPLUS,
    FORTH_HOST_FMINUS,
    FORTH_HOST_FSTAR,
    FORTH_HOST_FSLASH,
    FORTH_HOST_FNEGATE,
    FORTH_HOST_FZERO_LESS,
    FORTH_HOST_FZERO_EQUAL,
    FORTH_HOST_FLESS,
    FORTH_HOST_FABS,
    FORTH_HOST_FMAX,
    FORTH_HOST_FMIN,
    FORTH_HOST_FTILDE,
    FORTH_HOST_FFETCH,
    FORTH_HOST_FSTORE,
    FORTH_HOST_SFFETCH,
    FORTH_HOST_SFSTORE,
    FORTH_HOST_DFFETCH,
    FORTH_HOST_DFSTORE,
    FORTH_HOST_FLITERAL,
    FORTH_HOST_F_LIT_BITS,
    FORTH_HOST_FLOATS,
    FORTH_HOST_SFLOATS,
    FORTH_HOST_DFLOATS,
    FORTH_HOST_TO_FLOAT,
    FORTH_HOST_FLOOR,
    FORTH_HOST_FROUND,
    FORTH_HOST_FSQRT,
    FORTH_HOST_FSIN,
    FORTH_HOST_FCOS,
    FORTH_HOST_FTAN,
    FORTH_HOST_FASIN,
    FORTH_HOST_FACOS,
    FORTH_HOST_FATAN,
    FORTH_HOST_FATAN2,
    FORTH_HOST_FSINCOS,
    FORTH_HOST_FEXP,
    FORTH_HOST_FEXPM1,
    FORTH_HOST_FLN,
    FORTH_HOST_FLOG,
    FORTH_HOST_FLNP1,
    FORTH_HOST_FSTAR_STAR,
    FORTH_HOST_FALOG,
    FORTH_HOST_FSINH,
    FORTH_HOST_FCOSH,
    FORTH_HOST_FTANH,
    FORTH_HOST_FASINH,
    FORTH_HOST_FACOSH,
    FORTH_HOST_FATANH,
    FORTH_HOST_REPRESENT,
    FORTH_HOST_PRECISION,
    FORTH_HOST_SET_PRECISION,
    FORTH_HOST_FS_DOT,
    FORTH_HOST_FE_DOT,
    FORTH_HOST_F_DOT,
    FORTH_HOST_XCHAR_PLUS,
    FORTH_HOST_XCHAR_MINUS,
    FORTH_HOST_XC_FETCH_PLUS,
    FORTH_HOST_XC_STORE_PLUS,
    FORTH_HOST_XC_STORE_PLUS_Q,
    FORTH_HOST_XC_SIZE,
    FORTH_HOST_X_SIZE,
    FORTH_HOST_XC_COMMA,
    FORTH_HOST_XEMIT,
    FORTH_HOST_XKEY,
    FORTH_HOST_XKEY_Q,
    FORTH_HOST_PLUS_XSTRING,
    FORTH_HOST_X_STRING_MINUS,
    FORTH_HOST_TRAILING_GARBAGE,
    FORTH_HOST_X_WIDTH,
    FORTH_HOST_XC_WIDTH,
    FORTH_HOST_XHOLD,
    FORTH_HOST_EKEY_TO_XCHAR,
    FORTH_HOST_BLOCK,
    FORTH_HOST_BUFFER,
    FORTH_HOST_UPDATE,
    FORTH_HOST_FLUSH,
    FORTH_HOST_SAVE_BUFFERS,
    FORTH_HOST_EMPTY_BUFFERS,
    FORTH_HOST_LOAD,
    FORTH_HOST_LIST,
    FORTH_HOST_THRU
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
    int64_t file_pos;
} ForthSourceFrame;

typedef struct {
    uint8_t name[FORTH_SUBST_NAME_MAX];
    uint32_t nlen;
    uint64_t text_addr;
    uint32_t tlen;
    bool used;
} ForthSubst;

typedef struct {
    uint8_t name[FORTH_NAME_MAX];
    uint32_t nlen;
    bool from_stack;
    bool inited;
} ForthCompileLocal;

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
    uint32_t fprecision;
    ForthCtrlItem control[FORTH_CONTROL_STACK_CELLS];
    uint32_t control_depth;
    ForthRegion *regions;
    uint32_t region_count;
    uint32_t region_cap;
    uint64_t bump;
    uint64_t heap_next;
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
    uint64_t s_quote_addr[2];
    uint32_t s_quote_sel;
    uint64_t hold_addr;
    uint64_t hld_addr;
    char *out_buf;
    uint32_t out_len;
    uint32_t out_cap;
    bool echo_output;
    uint64_t tib_addr;
    uint64_t file_tib_addr;
    uint64_t blocks_addr;
    uint64_t block_cache_addr;
    uint64_t xchar_enc_addr;
    bool block_assigned[FORTH_BLOCK_COUNT];
    bool block_dirty[FORTH_BLOCK_COUNT];
    int32_t block_current;
    ForthSourceFrame sources[FORTH_SOURCE_NEST];
    uint32_t source_depth;
    bool colon_open;
    bool colon_noname;
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
    uint32_t colon_does_chain_off;
    ForthCompileLocal colon_locals[FORTH_LOCAL_MAX];
    uint32_t colon_local_count;
    bool colon_locals_closed;
    uint32_t colon_parent_local_count;
    uint32_t colon_does_a_local_count;
    ForthNt does_child_nt;
    bool does_rebuild_pending;
    uint32_t vm_exec_depth;
    bool exit_requested;
    bool quit_requested;
    ForthSubst subst[FORTH_SUBST_MAX];
    char required[FORTH_REQUIRED_MAX][FORTH_PATH_MAX];
    uint32_t required_count;
};

static bool forth_allocate_ex(ForthSession *session, uint64_t bytes, uint64_t *addr,
                              bool pinned);
static bool forth_install_dpush(ForthSession *session);
static bool forth_install_kernel(ForthSession *session);
static bool forth_install_runtime_import(ForthSession *session);
static void wrap_patch_rel(uint8_t *code, uint32_t instr_off, uint32_t target_off);
static bool wrap_emit(uint8_t *code, uint32_t *off, uint32_t cap, NanoOpcode op, ...);
bool forth_interpret_loop(ForthSession *session);
static int forth_run_host(ForthSession *session, uint16_t host, int64_t state);
static bool forth_throw_pending(ForthSession *session);
static bool forth_throw_now(ForthSession *session, int64_t code);
static bool forth_dict_allot(ForthSession *session, int64_t n);
static bool refill_file_line(ForthSession *session, ForthSourceFrame *frame);
static int forth_host_parse_name(ForthSession *session);
static bool forth_block_assign(ForthSession *session, uint32_t blk, bool load);
static bool forth_block_in_range(uint32_t blk);
static uint64_t forth_block_cache(const ForthSession *session, uint32_t blk);

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
    uint32_t slot;
    FILE *fp;

    if (!frame) return false;
    if (frame->kind == FORTH_SRC_FILE) {
        if (!decode_fileid(session, frame->fileid, &slot)) return false;
        fp = session->files[slot].fp;
        if (!fp || fseek(fp, (long)frame->file_pos, SEEK_SET) != 0) return false;
        if (!refill_file_line(session, frame)) return false;
    }
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
    if (!forth_allocate_ex(session,
                           (uint64_t)FORTH_BLOCK_SIZE * (uint64_t)FORTH_BLOCK_COUNT,
                           &session->block_cache_addr, true))
        return false;
    if (!forth_allocate_ex(session, 8, &session->xchar_enc_addr, true))
        return false;
    {
        const char *enc = "UTF-8";
        uint32_t i;
        for (i = 0; enc[i] != 0; i++) {
            if (!forth_store_byte(session, session->xchar_enc_addr + i,
                                  (uint8_t)enc[i]))
                return false;
        }
    }
    session->block_current = -1;

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
    if (!forth_allocate_ex(session, FORTH_WORD_MAX, &session->s_quote_addr[0],
                           true))
        return false;
    if (!forth_allocate_ex(session, FORTH_WORD_MAX, &session->s_quote_addr[1],
                           true))
        return false;
    session->s_quote_sel = 0;
    session->heap_next = FORTH_HEAP_BASE;
    session->fprecision = 5;
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
    if (!forth_install_runtime_import(session)) {
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

static bool forth_flush_does_rebuild(ForthSession *session) {
    if (!session || !session->does_rebuild_pending) return true;
    if (session->vm_exec_depth != 0) return true;
    session->does_rebuild_pending = false;
    return forth_session_rebuild(session);
}

VmResult forth_session_invoke(ForthSession *session, uint32_t fn_idx,
                              const NanoValue *args, uint16_t arg_count,
                              NanoValue *out_result) {
    ForthSession *prev;
    VmResult ran;
    if (!session) return VM_ERR_UNDEFINED_FUNCTION;
    prev = g_forth;
    g_forth = session;
    session->vm_exec_depth++;
    ran = vm_invoke(&session->vm, fn_idx, args, arg_count, out_result);
    session->vm_exec_depth--;
    g_forth = prev;
    if (ran == VM_OK && !forth_flush_does_rebuild(session))
        return VM_ERR_DECODE;
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
    session->vm_exec_depth++;
    ran = vm_invoke(vm, xt, NULL, 0, NULL);
    session->vm_exec_depth--;
    g_forth = prev;
    if (saved_frames > 0)
        memcpy(vm->frames, copy, (size_t)saved_frames * sizeof(VmCallFrame));
    vm->frame_count = saved_frames;
    vm->ip = saved_ip;
    vm->current_fn = saved_fn;
    if (ran == VM_OK && !forth_flush_does_rebuild(session))
        return VM_ERR_DECODE;
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

/* 1 reused, 0 none found, -1 error. Dictionary and heap never share holes. */
static int forth_try_reuse_region(ForthSession *session, uint64_t size,
                                  bool pinned, bool heap, uint64_t *addr) {
    uint32_t i;

    for (i = 0; i < session->region_count; i++) {
        ForthRegion *region = &session->regions[i];
        bool region_heap;
        if (region->used || region->size < size) continue;
        region_heap = region->addr >= FORTH_HEAP_BASE;
        if (region_heap != heap) continue;
        if (region->size - size >= FORTH_CELL_BYTES) {
            if (!regions_reserve(session, 1)) return -1;
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
        return 1;
    }
    return 0;
}

static bool forth_allocate_ex(ForthSession *session, uint64_t bytes, uint64_t *addr,
                              bool pinned) {
    uint64_t size;
    int reused;

    if (!session || !addr || bytes == 0) return false;
    size = align_cells(bytes);
    if (size == 0 || size == UINT64_MAX) return false;

    reused = forth_try_reuse_region(session, size, pinned, false, addr);
    if (reused < 0) return false;
    if (reused > 0) return true;

    if (session->bump > UINT64_MAX - size) return false;
    if (session->bump + size > FORTH_HEAP_BASE) return false;
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

static bool forth_heap_allocate(ForthSession *session, uint64_t bytes, uint64_t *addr) {
    uint64_t size;
    int reused;

    if (!session || !addr || bytes == 0) return false;
    size = align_cells(bytes);
    if (size == 0 || size == UINT64_MAX) return false;

    reused = forth_try_reuse_region(session, size, false, true, addr);
    if (reused < 0) return false;
    if (reused > 0) return true;

    if (session->heap_next < FORTH_HEAP_BASE) session->heap_next = FORTH_HEAP_BASE;
    if (session->heap_next > UINT64_MAX - size) return false;
    if (!ensure_memory(session, session->heap_next + size)) return false;
    if (!regions_reserve(session, 1)) return false;
    session->regions[session->region_count].addr = session->heap_next;
    session->regions[session->region_count].size = size;
    session->regions[session->region_count].used = true;
    session->regions[session->region_count].pinned = false;
    session->region_count++;
    *addr = session->heap_next;
    session->heap_next += size;
    return true;
}

static bool forth_resize(ForthSession *session, uint64_t addr, uint64_t new_bytes,
                         uint64_t *new_addr) {
    int idx;
    ForthRegion *region;
    uint64_t new_size;
    uint64_t old_size;
    uint64_t dst = 0;
    uint64_t ncopy;
    uint64_t i;

    if (!session || !new_addr || addr == 0 || new_bytes == 0) return false;
    if (addr < FORTH_HEAP_BASE) return false;
    idx = find_region_at(session, addr);
    if (idx < 0) return false;
    region = &session->regions[idx];
    if (region->pinned) return false;
    new_size = align_cells(new_bytes);
    if (new_size == 0 || new_size == UINT64_MAX) return false;
    old_size = region->size;
    if (new_size == old_size) {
        *new_addr = addr;
        return true;
    }
    if (new_size < old_size) {
        if (addr + old_size == session->heap_next) {
            session->heap_next = addr + new_size;
            region->size = new_size;
            *new_addr = addr;
            return true;
        }
        if (old_size - new_size >= FORTH_CELL_BYTES) {
            if (!regions_reserve(session, 1)) return false;
            region = &session->regions[idx];
            session->regions[session->region_count].addr = addr + new_size;
            session->regions[session->region_count].size = old_size - new_size;
            session->regions[session->region_count].used = false;
            session->regions[session->region_count].pinned = false;
            session->region_count++;
        }
        region->size = new_size;
        *new_addr = addr;
        return true;
    }
    if (addr + old_size == session->heap_next) {
        uint64_t extra = new_size - old_size;
        if (session->heap_next > UINT64_MAX - extra) return false;
        if (!ensure_memory(session, session->heap_next + extra)) return false;
        session->heap_next += extra;
        region->size = new_size;
        *new_addr = addr;
        return true;
    }
    if (!forth_heap_allocate(session, new_bytes, &dst)) return false;
    ncopy = old_size < new_size ? old_size : new_size;
    if (dst + ncopy > session->vm.memory_size
            || addr + ncopy > session->vm.memory_size)
        return false;
    for (i = 0; i < ncopy; i++) {
        session->vm.memory[dst + i] = session->vm.memory[addr + i];
    }
    if (!forth_free(session, addr)) return false;
    *new_addr = dst;
    return true;
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
    strncpy(session->files[slot].path, path, FORTH_PATH_MAX - 1);
    session->files[slot].path[FORTH_PATH_MAX - 1] = '\0';
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

static bool forth_find_in_wid(const ForthSession *session, ForthWid wid,
                              const char *name, uint32_t name_len,
                              ForthNt *nt, ForthXt *xt, bool *immediate) {
    uint32_t i;

    if (!session || !name || !valid_wid(session, wid)) return false;
    if (name_len == 0 || name_len > FORTH_NAME_MAX) return false;
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
    return false;
}

bool forth_find(const ForthSession *session, const char *name, uint32_t name_len,
                ForthNt *nt, ForthXt *xt, bool *immediate) {
    uint32_t o;

    if (!session || !name) return false;
    if (name_len == 0 || name_len > FORTH_NAME_MAX) return false;

    for (o = 0; o < session->order_count; o++) {
        if (forth_find_in_wid(session, session->order[o], name, name_len,
                              nt, xt, immediate))
            return true;
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
    child->caddr = session->block_cache_addr + (uint64_t)blk * FORTH_BLOCK_SIZE;
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
    {
        long pos = ftell(fp);
        frame->file_pos = (pos < 0) ? 0 : (int64_t)pos;
    }

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
    if (frame->kind == FORTH_SRC_BLOCK) {
        int64_t next = frame->blk + 1;
        if (next < 1 || next >= (int64_t)FORTH_BLOCK_COUNT) return false;
        if (!forth_block_assign(session, (uint32_t)next, true)) return false;
        frame->blk = next;
        frame->caddr = forth_block_cache(session, (uint32_t)next);
        frame->u = FORTH_BLOCK_SIZE;
        if (!forth_store_cell(session, session->sysvars, 0)) return false;
        if (!forth_store_cell(session, session->sysvars + FORTH_CELL_BYTES, next))
            return false;
        return true;
    }
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
    n = forth_emit_into(code + off, OP_PUSH_I64, (int64_t)FORTH_HOST_PLUSLOOP);
    if (!n) return false;
    off += n;
    n = forth_emit_into(code + off, OP_CALL_EXTERN, session->runtime_import);
    if (!n) return false;
    off += n;
    n = forth_emit_into(code + off, OP_POP); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_CALL, session->dpop_fn); if (!n) return false; off += n;
    n = forth_emit_into(code + off, OP_RET); if (!n) return false; off += n;
    fn.name_idx = nvm_add_string(mod, "nl_forth_plusloop_step", 22);
    fn.arity = 0;
    fn.code_offset = nvm_append_code(mod, code, off);
    fn.code_length = off;
    fn.local_count = 0;
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

static int forth_local_slot(const ForthSession *session, const uint8_t *name,
                            uint32_t nlen) {
    uint32_t i;
    if (!session || !session->colon_open || !name || nlen == 0) return -1;
    for (i = session->colon_local_count; i > 0; i--) {
        const ForthCompileLocal *loc = &session->colon_locals[i - 1];
        if (names_equal(loc->name, loc->nlen, name, nlen))
            return (int)(i - 1);
    }
    return -1;
}

static bool forth_colon_local_fetch(ForthSession *session, int slot) {
    if (!colon_emit(session, OP_LOAD_LOCAL, slot)) return false;
    return colon_emit(session, OP_CALL, session->dpush_fn);
}

static bool forth_colon_local_store(ForthSession *session, int slot) {
    if (!colon_emit(session, OP_CALL, session->dpop_fn)) return false;
    return colon_emit(session, OP_STORE_LOCAL, slot);
}

static void forth_reset_colon_locals(ForthSession *session) {
    if (!session) return;
    session->colon_local_count = 0;
    session->colon_locals_closed = false;
}

static bool forth_add_colon_local(ForthSession *session, const uint8_t *name,
                                  uint32_t nlen, bool from_stack, bool eager) {
    ForthCompileLocal *loc;
    uint32_t slot;
    if (!session || !session->colon_open || !name || nlen == 0) return false;
    if (session->colon_locals_closed) return false;
    if (nlen > FORTH_NAME_MAX) return false;
    if (session->colon_local_count >= FORTH_LOCAL_MAX) return false;
    if (forth_local_slot(session, name, nlen) >= 0) return false;
    slot = session->colon_local_count;
    loc = &session->colon_locals[slot];
    memcpy(loc->name, name, nlen);
    loc->nlen = nlen;
    loc->from_stack = from_stack;
    loc->inited = false;
    session->colon_local_count++;
    if (eager && from_stack) {
        if (!forth_colon_local_store(session, (int)slot)) return false;
        loc->inited = true;
    }
    return true;
}

static bool forth_locals_close(ForthSession *session) {
    uint32_t i;
    if (!session || !session->colon_open) return false;
    if (session->colon_locals_closed) return true;
    for (i = session->colon_local_count; i > 0; i--) {
        ForthCompileLocal *loc = &session->colon_locals[i - 1];
        if (!loc->from_stack || loc->inited) continue;
        if (!forth_colon_local_store(session, (int)(i - 1))) return false;
        loc->inited = true;
    }
    session->colon_locals_closed = true;
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
    session->colon_noname = false;
    session->colon_code_len = 0;
    session->colon_fn_idx = 0;
    session->colon_nt = 0;
    session->colon_local_count = 0;
    session->colon_locals_closed = false;
    session->colon_parent_local_count = 0;
    session->colon_does_a_local_count = 0;
    if (!forth_store_cell(session, forth_state_addr(session), 0)) return false;
    return true;
}

static bool colon_begin_common(ForthSession *session, const char *name,
                               uint32_t name_len, bool named) {
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
    session->colon_noname = !named;
    session->colon_code_len = 0;
    session->colon_does_pending = false;
    session->colon_does_off = 0;
    session->colon_does_chain_off = 0;
    session->colon_local_count = 0;
    session->colon_locals_closed = false;
    session->colon_parent_local_count = 0;
    session->colon_does_a_local_count = 0;
    if (!named) {
        session->colon_nt = 0;
        return true;
    }
    if (!forth_define(session, name, name_len, (ForthXt)session->colon_fn_idx,
                      false, true, &nt)) {
        colon_rollback(session);
        return false;
    }
    session->colon_nt = nt;
    return true;
}

bool forth_colon_begin(ForthSession *session, const char *name, uint32_t name_len) {
    return colon_begin_common(session, name, name_len, true);
}

static bool forth_colon_begin_noname(ForthSession *session) {
    return colon_begin_common(session, ":NONAME", 7, false);
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

static bool colon_call_named(ForthSession *session, const char *name) {
    ForthNt nt = 0;
    ForthXt xt = 0;
    bool imm = false;
    if (!name) return false;
    if (!forth_find(session, name, (uint32_t)strlen(name), &nt, &xt, &imm))
        return false;
    return forth_colon_call(session, xt);
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

static bool forth_colon_ahead(ForthSession *session) {
    uint32_t instr_off;
    if (!session || !session->colon_open) return false;
    instr_off = session->colon_code_len;
    if (!colon_emit(session, OP_JMP, (int32_t)0)) return false;
    return forth_control_push(session, FORTH_CTRL_ORIG, instr_off);
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
    uint32_t dest = 0;
    uint32_t instr_off;
    if (!session || !session->colon_open) return false;
    if (!colon_pop_ctrl(session, FORTH_CTRL_DEST, &dest)) return false;
    if (!colon_emit_dpop(session)) return false;
    instr_off = session->colon_code_len;
    if (!colon_emit(session, OP_JMP_FALSE, (int32_t)0)) return false;
    if (!forth_control_push(session, FORTH_CTRL_ORIG, instr_off)) return false;
    return forth_control_push(session, FORTH_CTRL_DEST, dest);
}

bool forth_colon_repeat(ForthSession *session) {
    uint32_t orig = 0;
    uint32_t dest = 0;
    uint32_t instr_off;
    int32_t rel;
    if (!session || !session->colon_open) return false;
    if (!colon_pop_ctrl(session, FORTH_CTRL_DEST, &dest)) return false;
    if (!colon_pop_ctrl(session, FORTH_CTRL_ORIG, &orig)) return false;
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

static bool forth_publish_does_code(ForthSession *session, uint8_t *code,
                                    uint32_t len, uint16_t locals,
                                    uint32_t *xt_out) {
    NvmFunctionEntry dfn;
    NvmVerifyResult dver;
    NvmModule *mod;
    uint32_t xt;

    if (!session || !code || !xt_out || !session->module) return false;
    mod = session->module;
    memset(&dfn, 0, sizeof(dfn));
    dfn.name_idx = nvm_add_string(mod, "nl_forth_does", 13);
    dfn.arity = 0;
    dfn.code_offset = nvm_append_code(mod, code, len);
    dfn.code_length = len;
    dfn.local_count = locals;
    dfn.result_tag = TAG_VOID;
    dfn.result_count = 0;
    xt = nvm_add_function(mod, &dfn);
    if (xt >= mod->function_count) return false;
    dver = nvm_verify_function(mod, xt);
    if (!dver.ok) return false;
    *xt_out = xt;
    return true;
}

static bool forth_emit_does_attach(ForthSession *session, uint8_t *code,
                                   uint32_t *off, uint32_t cap, uint32_t does_xt) {
    if (!wrap_emit(code, off, cap, OP_PUSH_I64, (int64_t)does_xt)) return false;
    if (!wrap_emit(code, off, cap, OP_CALL, session->dpush_fn)) return false;
    if (!wrap_emit(code, off, cap, OP_PUSH_I64, (int64_t)FORTH_HOST_DOES))
        return false;
    if (!wrap_emit(code, off, cap, OP_CALL_EXTERN, session->runtime_import))
        return false;
    return wrap_emit(code, off, cap, OP_POP);
}

bool forth_colon_finish(ForthSession *session, ForthNt *nt) {
    NvmModule *mod;
    NvmFunctionEntry *fn;
    NvmVerifyResult verified;
    uint32_t code_off;
    uint16_t parent_locals;
    uint16_t does_locals;
    uint16_t does_a_locals;

    if (!session || !nt || !session->colon_open || !session->module) return false;
    if (session->control_depth != session->colon_saved_control_depth) {
        colon_rollback(session);
        return false;
    }
    if (!forth_locals_close(session)) {
        colon_rollback(session);
        return false;
    }
    parent_locals = session->colon_does_pending
        ? (uint16_t)session->colon_parent_local_count
        : (uint16_t)session->colon_local_count;
    does_locals = (uint16_t)session->colon_local_count;
    does_a_locals = (uint16_t)session->colon_does_a_local_count;
    if (session->colon_does_pending) {
        uint8_t does_code[FORTH_COLON_CODE_MAX];
        uint32_t does_off = 0;
        uint32_t does_xt = 0;
        uint32_t start = session->colon_does_off;
        uint32_t chain = session->colon_does_chain_off;
        uint32_t end = session->colon_code_len;

        if (start > end) {
            colon_rollback(session);
            return false;
        }
        if (chain != 0 && chain >= start && chain <= end) {
            uint32_t does_xt_b = 0;
            uint32_t b_len = end - chain;
            uint8_t b_code[FORTH_COLON_CODE_MAX];
            uint32_t b_off = 0;

            memcpy(b_code, session->colon_code + chain, b_len);
            b_off = b_len;
            if (!wrap_emit(b_code, &b_off, sizeof(b_code), OP_RET)) {
                colon_rollback(session);
                return false;
            }
            if (!forth_publish_does_code(session, b_code, b_off, does_locals,
                                         &does_xt_b)) {
                colon_rollback(session);
                return false;
            }
            memcpy(does_code, session->colon_code + start, chain - start);
            does_off = chain - start;
            if (!forth_emit_does_attach(session, does_code, &does_off,
                                        sizeof(does_code), does_xt_b)) {
                colon_rollback(session);
                return false;
            }
            if (!wrap_emit(does_code, &does_off, sizeof(does_code), OP_RET)) {
                colon_rollback(session);
                return false;
            }
            if (!forth_publish_does_code(session, does_code, does_off,
                                         does_a_locals, &does_xt)) {
                colon_rollback(session);
                return false;
            }
        } else {
            uint32_t does_len = end - start;
            memcpy(does_code, session->colon_code + start, does_len);
            does_off = does_len;
            if (!wrap_emit(does_code, &does_off, sizeof(does_code), OP_RET)) {
                colon_rollback(session);
                return false;
            }
            if (!forth_publish_does_code(session, does_code, does_off,
                                         does_locals, &does_xt)) {
                colon_rollback(session);
                return false;
            }
        }
        session->colon_code_len = start;
        if (!forth_colon_literal(session, (int64_t)does_xt)
                || !colon_emit(session, OP_PUSH_I64, (int64_t)FORTH_HOST_DOES)
                || !colon_emit(session, OP_CALL_EXTERN, session->runtime_import)
                || !colon_emit(session, OP_POP)) {
            colon_rollback(session);
            return false;
        }
        session->colon_does_pending = false;
        session->colon_does_chain_off = 0;
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
    fn->local_count = parent_locals;
    verified = nvm_verify_function(mod, session->colon_fn_idx);
    if (!verified.ok) {
        colon_rollback(session);
        return false;
    }
    {
        ForthXt published_xt = session->colon_fn_idx;
        bool noname = session->colon_noname;
        if (!noname) {
            if (!forth_reveal(session, session->colon_nt)) {
                colon_rollback(session);
                return false;
            }
        }
        *nt = session->colon_nt;
        session->colon_open = false;
        session->colon_noname = false;
        session->colon_code_len = 0;
        forth_reset_colon_locals(session);
        session->colon_parent_local_count = 0;
        session->colon_does_a_local_count = 0;
        if (session->vm_exec_depth != 0) {
            session->does_rebuild_pending = true;
            if (!vm_sync_new_functions(&session->vm, session->module)) {
                session->colon_open = true;
                colon_rollback(session);
                return false;
            }
        } else if (!forth_session_rebuild(session)) {
            session->colon_open = true;
            colon_rollback(session);
            return false;
        }
        if (noname && !forth_data_push(session, (int64_t)published_xt))
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
                              uint16_t locals, bool immediate, uint16_t host) {
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
                                     uint16_t host) {
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
                               bool immediate, uint16_t host) {
    return forth_install_host_flags(session, name, immediate, false, host);
}

static bool forth_install_compile_only(ForthSession *session, const char *name,
                                       uint16_t host) {
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
                                       uint16_t host) {
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

static bool forth_install_abort(ForthSession *session) {
    uint8_t code[64];
    uint32_t off = 0;
    if (!wrap_emit(code, &off, sizeof(code), OP_PUSH_I64, (int64_t)-1))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_CALL, session->throw_fn))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_RET)) return false;
    return forth_publish_prim(session, "ABORT", code, off, 0, false,
                              FORTH_HOST_NONE);
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
    int64_t thrown = 0;
    if (!session) return 0;
    rc = forth_run_host(session, (uint16_t)kind, 0);
    if (rc < 0) {
        if (!forth_fetch_cell(session, session->throw_code_addr, &thrown)
                || thrown == 0) {
            forth_store_cell(session, session->throw_code_addr, -1);
        }
        vm_request_halt(&session->vm);
    } else if (forth_throw_pending(session)) {
        vm_request_halt(&session->vm);
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
        ": ERASE 0 FILL ;",
        ": 2>R SWAP >R >R ;",
        ": 2R> R> R> SWAP ;",
        ": 2R@ R> R> 2DUP >R >R SWAP ;",
        ": BUFFER: CREATE ALLOT ;",
        ": DEFER CREATE ['] ABORT , DOES> @ EXECUTE ;",
        ": DEFER@ >BODY @ ;",
        ": DEFER! >BODY ! ;",
        ": 2CONSTANT CREATE , , DOES> 2@ ;",
        ": 2VARIABLE CREATE 0 , 0 , ;",
        ": D0= OR 0= ;",
        ": D0< NIP 0< ;",
        ": D>S DROP ;",
        ": 2ROT 2>R 2SWAP 2R> 2SWAP ;",
        ": D. DUP >R DABS <# #S R> SIGN #> TYPE SPACE ;",
        ": D.R >R DUP >R DABS <# #S R> SIGN #> R> OVER - 0 MAX SPACES TYPE ;",
        ": /STRING DUP >R - SWAP R> + SWAP ;",
        ": BLANK BL FILL ;",
        ": R/O 1 ;",
        ": W/O 2 ;",
        ": R/W 3 ;",
        ": BEGIN-STRUCTURE CREATE HERE 0 0 , DOES> @ ;",
        ": END-STRUCTURE SWAP ! ;",
        ": +FIELD CREATE OVER , + DOES> @ + ;",
        ": FIELD: ALIGNED 1 CELLS +FIELD ;",
        ": CFIELD: 1 CHARS +FIELD ;",
        ": F, HERE 8 ALLOT F! ;",
        ": FCONSTANT CREATE F, DOES> F@ ;",
        ": FVARIABLE CREATE 8 ALLOT ;",
        "VARIABLE SCR",
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

    if (!forth_install_runtime_host(session, ":", FORTH_HOST_COLON)) return false;
    {
        ForthNt colon_nt = 0;
        ForthXt colon_xt = 0;
        bool colon_imm = false;
        ForthHeader *colon_hdr;
        if (!forth_find(session, ":", 1, &colon_nt, &colon_xt, &colon_imm))
            return false;
        colon_hdr = header_at(session, colon_nt);
        if (!colon_hdr) return false;
        colon_hdr->compile_only = true;
    }
    if (!forth_install_runtime_host(session, ":NONAME", FORTH_HOST_NONAME))
        return false;
    if (!forth_install_compile_only(session, ";", FORTH_HOST_SEMI)) return false;
    if (!forth_install_compile_only(session, "IF", FORTH_HOST_IF)) return false;
    if (!forth_install_compile_only(session, "ELSE", FORTH_HOST_ELSE)) return false;
    if (!forth_install_compile_only(session, "THEN", FORTH_HOST_THEN)) return false;
    if (!forth_install_compile_only(session, "AHEAD", FORTH_HOST_AHEAD)) return false;
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
    if (!forth_install_compile_only(session, "2LITERAL", FORTH_HOST_TWO_LITERAL))
        return false;
    if (!forth_install_runtime_host(session, "IMMEDIATE", FORTH_HOST_IMMEDIATE))
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
    if (!forth_install_host(session, ".(", true, FORTH_HOST_DOT_PAREN)) return false;
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
    if (!forth_install_abort(session)) return false;
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
    if (!forth_install_runtime_host(session, "SOURCE-ID", FORTH_HOST_SOURCE_ID))
        return false;
    if (!forth_install_runtime_host(session, "REFILL", FORTH_HOST_REFILL))
        return false;
    if (!forth_install_runtime_host(session, "PARSE-NAME", FORTH_HOST_PARSE_NAME))
        return false;
    if (!forth_install_runtime_host(session, "VALUE", FORTH_HOST_VALUE))
        return false;
    if (!forth_install_host(session, "TO", true, FORTH_HOST_TO)) return false;
    if (!forth_install_host(session, "IS", true, FORTH_HOST_IS)) return false;
    if (!forth_install_host(session, "ACTION-OF", true, FORTH_HOST_ACTION_OF))
        return false;
    if (!forth_install_runtime_host(session, "MARKER", FORTH_HOST_MARKER))
        return false;
    if (!forth_install_compile_only(session, "CASE", FORTH_HOST_CASE)) return false;
    if (!forth_install_compile_only(session, "OF", FORTH_HOST_OF)) return false;
    if (!forth_install_compile_only(session, "ENDOF", FORTH_HOST_ENDOF))
        return false;
    if (!forth_install_compile_only(session, "ENDCASE", FORTH_HOST_ENDCASE))
        return false;
    if (!forth_install_compile_only(session, "C\"", FORTH_HOST_C_QUOTE))
        return false;
    if (!forth_install_host(session, "S\\\"", true, FORTH_HOST_S_BACKSLASH))
        return false;
    if (!forth_install_runtime_host(session, ".R", FORTH_HOST_DOT_R)) return false;
    if (!forth_install_runtime_host(session, "U.R", FORTH_HOST_UDOT_R))
        return false;
    if (!forth_install_runtime_host(session, "HOLDS", FORTH_HOST_HOLDS))
        return false;
    if (!forth_install_runtime_host(session, "UNUSED", FORTH_HOST_UNUSED))
        return false;
    if (!forth_install_runtime_host(session, "SAVE-INPUT", FORTH_HOST_SAVE_INPUT))
        return false;
    if (!forth_install_runtime_host(session, "RESTORE-INPUT",
                                    FORTH_HOST_RESTORE_INPUT))
        return false;
    if (!forth_install_runtime_host(session, "2VALUE", FORTH_HOST_TWO_VALUE))
        return false;
    if (!forth_install_runtime_host(session, "D+", FORTH_HOST_DPLUS)) return false;
    if (!forth_install_runtime_host(session, "D-", FORTH_HOST_DMINUS)) return false;
    if (!forth_install_runtime_host(session, "DNEGATE", FORTH_HOST_DNEGATE))
        return false;
    if (!forth_install_runtime_host(session, "D2*", FORTH_HOST_DTWO_STAR))
        return false;
    if (!forth_install_runtime_host(session, "D2/", FORTH_HOST_DTWO_SLASH))
        return false;
    if (!forth_install_runtime_host(session, "D<", FORTH_HOST_DLESS)) return false;
    if (!forth_install_runtime_host(session, "D=", FORTH_HOST_DEQUAL)) return false;
    if (!forth_install_runtime_host(session, "DABS", FORTH_HOST_DABS)) return false;
    if (!forth_install_runtime_host(session, "DMAX", FORTH_HOST_DMAX)) return false;
    if (!forth_install_runtime_host(session, "DMIN", FORTH_HOST_DMIN)) return false;
    if (!forth_install_runtime_host(session, "M+", FORTH_HOST_MPLUS)) return false;
    if (!forth_install_runtime_host(session, "DU<", FORTH_HOST_DULESS)) return false;
    if (!forth_install_runtime_host(session, "M*/", FORTH_HOST_M_STAR_SLASH))
        return false;
    if (!forth_install_runtime_host(session, "-TRAILING", FORTH_HOST_TRAILING))
        return false;
    if (!forth_install_runtime_host(session, "CMOVE", FORTH_HOST_CMOVE))
        return false;
    if (!forth_install_runtime_host(session, "CMOVE>", FORTH_HOST_CMOVE_UP))
        return false;
    if (!forth_install_runtime_host(session, "COMPARE", FORTH_HOST_COMPARE))
        return false;
    if (!forth_install_runtime_host(session, "SEARCH", FORTH_HOST_SEARCH))
        return false;
    if (!forth_install_compile_only(session, "SLITERAL", FORTH_HOST_SLITERAL))
        return false;
    if (!forth_install_runtime_host(session, "UNESCAPE", FORTH_HOST_UNESCAPE))
        return false;
    if (!forth_install_runtime_host(session, "REPLACES", FORTH_HOST_REPLACES))
        return false;
    if (!forth_install_runtime_host(session, "SUBSTITUTE", FORTH_HOST_SUBSTITUTE))
        return false;
    if (!forth_install_runtime_host(session, "WORDLIST", FORTH_HOST_WORDLIST))
        return false;
    if (!forth_install_runtime_host(session, "GET-ORDER", FORTH_HOST_GET_ORDER))
        return false;
    if (!forth_install_runtime_host(session, "SET-ORDER", FORTH_HOST_SET_ORDER))
        return false;
    if (!forth_install_runtime_host(session, "GET-CURRENT",
                                    FORTH_HOST_GET_CURRENT))
        return false;
    if (!forth_install_runtime_host(session, "SET-CURRENT",
                                    FORTH_HOST_SET_CURRENT))
        return false;
    if (!forth_install_runtime_host(session, "FORTH-WORDLIST",
                                    FORTH_HOST_FORTH_WORDLIST))
        return false;
    if (!forth_install_runtime_host(session, "ALSO", FORTH_HOST_ALSO))
        return false;
    if (!forth_install_runtime_host(session, "PREVIOUS", FORTH_HOST_PREVIOUS))
        return false;
    if (!forth_install_runtime_host(session, "ONLY", FORTH_HOST_ONLY))
        return false;
    if (!forth_install_runtime_host(session, "FORTH", FORTH_HOST_FORTH))
        return false;
    if (!forth_install_runtime_host(session, "DEFINITIONS",
                                    FORTH_HOST_DEFINITIONS))
        return false;
    if (!forth_install_runtime_host(session, "SEARCH-WORDLIST",
                                    FORTH_HOST_SEARCH_WORDLIST))
        return false;
    if (!forth_install_runtime_host(session, "ORDER", FORTH_HOST_ORDER))
        return false;
    if (!forth_install_runtime_host(session, "BIN", FORTH_HOST_BIN))
        return false;
    if (!forth_install_runtime_host(session, "OPEN-FILE", FORTH_HOST_OPEN_FILE))
        return false;
    if (!forth_install_runtime_host(session, "CREATE-FILE",
                                    FORTH_HOST_CREATE_FILE))
        return false;
    if (!forth_install_runtime_host(session, "CLOSE-FILE", FORTH_HOST_CLOSE_FILE))
        return false;
    if (!forth_install_runtime_host(session, "DELETE-FILE",
                                    FORTH_HOST_DELETE_FILE))
        return false;
    if (!forth_install_runtime_host(session, "READ-FILE", FORTH_HOST_READ_FILE))
        return false;
    if (!forth_install_runtime_host(session, "READ-LINE", FORTH_HOST_READ_LINE))
        return false;
    if (!forth_install_runtime_host(session, "WRITE-FILE", FORTH_HOST_WRITE_FILE))
        return false;
    if (!forth_install_runtime_host(session, "WRITE-LINE", FORTH_HOST_WRITE_LINE))
        return false;
    if (!forth_install_runtime_host(session, "FILE-POSITION",
                                    FORTH_HOST_FILE_POSITION))
        return false;
    if (!forth_install_runtime_host(session, "FILE-SIZE", FORTH_HOST_FILE_SIZE))
        return false;
    if (!forth_install_runtime_host(session, "REPOSITION-FILE",
                                    FORTH_HOST_REPOSITION_FILE))
        return false;
    if (!forth_install_runtime_host(session, "RESIZE-FILE",
                                    FORTH_HOST_RESIZE_FILE))
        return false;
    if (!forth_install_runtime_host(session, "FLUSH-FILE", FORTH_HOST_FLUSH_FILE))
        return false;
    if (!forth_install_runtime_host(session, "RENAME-FILE",
                                    FORTH_HOST_RENAME_FILE))
        return false;
    if (!forth_install_runtime_host(session, "FILE-STATUS",
                                    FORTH_HOST_FILE_STATUS))
        return false;
    if (!forth_install_runtime_host(session, "INCLUDED", FORTH_HOST_INCLUDED))
        return false;
    if (!forth_install_runtime_host(session, "INCLUDE", FORTH_HOST_INCLUDE))
        return false;
    if (!forth_install_runtime_host(session, "INCLUDE-FILE",
                                    FORTH_HOST_INCLUDE_FILE))
        return false;
    if (!forth_install_runtime_host(session, "REQUIRED", FORTH_HOST_REQUIRED))
        return false;
    if (!forth_install_runtime_host(session, "REQUIRE", FORTH_HOST_REQUIRE))
        return false;
    if (!forth_install_runtime_host(session, "ALLOCATE", FORTH_HOST_ALLOCATE))
        return false;
    if (!forth_install_runtime_host(session, "FREE", FORTH_HOST_MEM_FREE))
        return false;
    if (!forth_install_runtime_host(session, "RESIZE", FORTH_HOST_MEM_RESIZE))
        return false;
    if (!forth_install_compile_only(session, "{:", FORTH_HOST_LOCALS_BRACE))
        return false;
    if (!forth_install_runtime_host(session, "(LOCAL)", FORTH_HOST_LOCAL))
        return false;
    {
        ForthNt local_nt = 0;
        ForthXt local_xt = 0;
        bool local_imm = false;
        ForthHeader *local_hdr;
        if (!forth_find(session, "(LOCAL)", 7, &local_nt, &local_xt, &local_imm))
            return false;
        local_hdr = header_at(session, local_nt);
        if (!local_hdr) return false;
        local_hdr->compile_only = true;
    }
    if (!forth_install_runtime_host(session, ".S", FORTH_HOST_DOT_S))
        return false;
    if (!forth_install_host(session, "[IF]", true, FORTH_HOST_BRACKET_IF))
        return false;
    if (!forth_install_host(session, "[ELSE]", true, FORTH_HOST_BRACKET_ELSE))
        return false;
    if (!forth_install_host(session, "[THEN]", true, FORTH_HOST_BRACKET_THEN))
        return false;
    if (!forth_install_runtime_host(session, "CS-PICK", FORTH_HOST_CS_PICK))
        return false;
    if (!forth_install_runtime_host(session, "CS-ROLL", FORTH_HOST_CS_ROLL))
        return false;
    {
        ForthNt cs_nt = 0;
        ForthXt cs_xt = 0;
        bool cs_imm = false;
        ForthHeader *cs_hdr;
        if (!forth_find(session, "CS-PICK", 7, &cs_nt, &cs_xt, &cs_imm))
            return false;
        cs_hdr = header_at(session, cs_nt);
        if (!cs_hdr) return false;
        cs_hdr->compile_only = true;
        if (!forth_find(session, "CS-ROLL", 7, &cs_nt, &cs_xt, &cs_imm))
            return false;
        cs_hdr = header_at(session, cs_nt);
        if (!cs_hdr) return false;
        cs_hdr->compile_only = true;
    }
    if (!forth_install_host(session, "[DEFINED]", true, FORTH_HOST_DEFINED))
        return false;
    if (!forth_install_host(session, "[UNDEFINED]", true, FORTH_HOST_UNDEFINED))
        return false;
    if (!forth_install_runtime_host(session, "N>R", FORTH_HOST_N_TO_R))
        return false;
    if (!forth_install_runtime_host(session, "NR>", FORTH_HOST_NR_FROM))
        return false;
    if (!forth_install_runtime_host(session, "SYNONYM", FORTH_HOST_SYNONYM))
        return false;
    if (!forth_install_runtime_host(session, "TRAVERSE-WORDLIST",
                                    FORTH_HOST_TRAVERSE_WORDLIST))
        return false;
    if (!forth_install_runtime_host(session, "NAME>COMPILE",
                                    FORTH_HOST_NAME_TO_COMPILE))
        return false;
    if (!forth_install_runtime_host(session, "NAME>INTERPRET",
                                    FORTH_HOST_NAME_TO_INTERPRET))
        return false;
    if (!forth_install_runtime_host(session, "NAME>STRING",
                                    FORTH_HOST_NAME_TO_STRING))
        return false;
    if (!forth_install_runtime_host(session, "D>F", FORTH_HOST_D_TO_F))
        return false;
    if (!forth_install_runtime_host(session, "F>D", FORTH_HOST_F_TO_D))
        return false;
    if (!forth_install_runtime_host(session, "FDEPTH", FORTH_HOST_FDEPTH))
        return false;
    if (!forth_install_runtime_host(session, "FDROP", FORTH_HOST_FDROP))
        return false;
    if (!forth_install_runtime_host(session, "FDUP", FORTH_HOST_FDUP))
        return false;
    if (!forth_install_runtime_host(session, "FSWAP", FORTH_HOST_FSWAP))
        return false;
    if (!forth_install_runtime_host(session, "FOVER", FORTH_HOST_FOVER))
        return false;
    if (!forth_install_runtime_host(session, "FROT", FORTH_HOST_FROT))
        return false;
    if (!forth_install_runtime_host(session, "F+", FORTH_HOST_FPLUS))
        return false;
    if (!forth_install_runtime_host(session, "F-", FORTH_HOST_FMINUS))
        return false;
    if (!forth_install_runtime_host(session, "F*", FORTH_HOST_FSTAR))
        return false;
    if (!forth_install_runtime_host(session, "F/", FORTH_HOST_FSLASH))
        return false;
    if (!forth_install_runtime_host(session, "FNEGATE", FORTH_HOST_FNEGATE))
        return false;
    if (!forth_install_runtime_host(session, "F0<", FORTH_HOST_FZERO_LESS))
        return false;
    if (!forth_install_runtime_host(session, "F0=", FORTH_HOST_FZERO_EQUAL))
        return false;
    if (!forth_install_runtime_host(session, "F<", FORTH_HOST_FLESS))
        return false;
    if (!forth_install_runtime_host(session, "FABS", FORTH_HOST_FABS))
        return false;
    if (!forth_install_runtime_host(session, "FMAX", FORTH_HOST_FMAX))
        return false;
    if (!forth_install_runtime_host(session, "FMIN", FORTH_HOST_FMIN))
        return false;
    if (!forth_install_runtime_host(session, "F~", FORTH_HOST_FTILDE))
        return false;
    if (!forth_install_runtime_host(session, "F@", FORTH_HOST_FFETCH))
        return false;
    if (!forth_install_runtime_host(session, "F!", FORTH_HOST_FSTORE))
        return false;
    if (!forth_install_runtime_host(session, "SF@", FORTH_HOST_SFFETCH))
        return false;
    if (!forth_install_runtime_host(session, "SF!", FORTH_HOST_SFSTORE))
        return false;
    if (!forth_install_runtime_host(session, "DF@", FORTH_HOST_DFFETCH))
        return false;
    if (!forth_install_runtime_host(session, "DF!", FORTH_HOST_DFSTORE))
        return false;
    if (!forth_install_compile_only(session, "FLITERAL", FORTH_HOST_FLITERAL))
        return false;
    if (!forth_install_runtime_host(session, "FLOATS", FORTH_HOST_FLOATS))
        return false;
    if (!forth_install_runtime_host(session, "SFLOATS", FORTH_HOST_SFLOATS))
        return false;
    if (!forth_install_runtime_host(session, "DFLOATS", FORTH_HOST_DFLOATS))
        return false;
    if (!forth_install_runtime_host(session, ">FLOAT", FORTH_HOST_TO_FLOAT))
        return false;
    if (!forth_install_runtime_host(session, "FLOOR", FORTH_HOST_FLOOR))
        return false;
    if (!forth_install_runtime_host(session, "FROUND", FORTH_HOST_FROUND))
        return false;
    if (!forth_install_runtime_host(session, "FSQRT", FORTH_HOST_FSQRT))
        return false;
    if (!forth_install_runtime_host(session, "FSIN", FORTH_HOST_FSIN))
        return false;
    if (!forth_install_runtime_host(session, "FCOS", FORTH_HOST_FCOS))
        return false;
    if (!forth_install_runtime_host(session, "FTAN", FORTH_HOST_FTAN))
        return false;
    if (!forth_install_runtime_host(session, "FASIN", FORTH_HOST_FASIN))
        return false;
    if (!forth_install_runtime_host(session, "FACOS", FORTH_HOST_FACOS))
        return false;
    if (!forth_install_runtime_host(session, "FATAN", FORTH_HOST_FATAN))
        return false;
    if (!forth_install_runtime_host(session, "FATAN2", FORTH_HOST_FATAN2))
        return false;
    if (!forth_install_runtime_host(session, "FSINCOS", FORTH_HOST_FSINCOS))
        return false;
    if (!forth_install_runtime_host(session, "FEXP", FORTH_HOST_FEXP))
        return false;
    if (!forth_install_runtime_host(session, "FEXPM1", FORTH_HOST_FEXPM1))
        return false;
    if (!forth_install_runtime_host(session, "FLN", FORTH_HOST_FLN))
        return false;
    if (!forth_install_runtime_host(session, "FLOG", FORTH_HOST_FLOG))
        return false;
    if (!forth_install_runtime_host(session, "FLNP1", FORTH_HOST_FLNP1))
        return false;
    if (!forth_install_runtime_host(session, "F**", FORTH_HOST_FSTAR_STAR))
        return false;
    if (!forth_install_runtime_host(session, "FALOG", FORTH_HOST_FALOG))
        return false;
    if (!forth_install_runtime_host(session, "FSINH", FORTH_HOST_FSINH))
        return false;
    if (!forth_install_runtime_host(session, "FCOSH", FORTH_HOST_FCOSH))
        return false;
    if (!forth_install_runtime_host(session, "FTANH", FORTH_HOST_FTANH))
        return false;
    if (!forth_install_runtime_host(session, "FASINH", FORTH_HOST_FASINH))
        return false;
    if (!forth_install_runtime_host(session, "FACOSH", FORTH_HOST_FACOSH))
        return false;
    if (!forth_install_runtime_host(session, "FATANH", FORTH_HOST_FATANH))
        return false;
    if (!forth_install_runtime_host(session, "REPRESENT", FORTH_HOST_REPRESENT))
        return false;
    if (!forth_install_runtime_host(session, "PRECISION", FORTH_HOST_PRECISION))
        return false;
    if (!forth_install_runtime_host(session, "SET-PRECISION",
                                    FORTH_HOST_SET_PRECISION))
        return false;
    if (!forth_install_runtime_host(session, "FS.", FORTH_HOST_FS_DOT))
        return false;
    if (!forth_install_runtime_host(session, "FE.", FORTH_HOST_FE_DOT))
        return false;
    if (!forth_install_runtime_host(session, "F.", FORTH_HOST_F_DOT))
        return false;
    if (!forth_install_runtime_host(session, "XCHAR+", FORTH_HOST_XCHAR_PLUS))
        return false;
    if (!forth_install_runtime_host(session, "XCHAR-", FORTH_HOST_XCHAR_MINUS))
        return false;
    if (!forth_install_runtime_host(session, "XC@+", FORTH_HOST_XC_FETCH_PLUS))
        return false;
    if (!forth_install_runtime_host(session, "XC!+", FORTH_HOST_XC_STORE_PLUS))
        return false;
    if (!forth_install_runtime_host(session, "XC!+?", FORTH_HOST_XC_STORE_PLUS_Q))
        return false;
    if (!forth_install_runtime_host(session, "XC-SIZE", FORTH_HOST_XC_SIZE))
        return false;
    if (!forth_install_runtime_host(session, "X-SIZE", FORTH_HOST_X_SIZE))
        return false;
    if (!forth_install_runtime_host(session, "XC,", FORTH_HOST_XC_COMMA))
        return false;
    if (!forth_install_runtime_host(session, "XEMIT", FORTH_HOST_XEMIT))
        return false;
    if (!forth_install_runtime_host(session, "XKEY", FORTH_HOST_XKEY))
        return false;
    if (!forth_install_runtime_host(session, "XKEY?", FORTH_HOST_XKEY_Q))
        return false;
    if (!forth_install_runtime_host(session, "+X/STRING", FORTH_HOST_PLUS_XSTRING))
        return false;
    if (!forth_install_runtime_host(session, "X\\STRING-",
                                    FORTH_HOST_X_STRING_MINUS))
        return false;
    if (!forth_install_runtime_host(session, "-TRAILING-GARBAGE",
                                    FORTH_HOST_TRAILING_GARBAGE))
        return false;
    if (!forth_install_runtime_host(session, "X-WIDTH", FORTH_HOST_X_WIDTH))
        return false;
    if (!forth_install_runtime_host(session, "XC-WIDTH", FORTH_HOST_XC_WIDTH))
        return false;
    if (!forth_install_runtime_host(session, "XHOLD", FORTH_HOST_XHOLD))
        return false;
    if (!forth_install_runtime_host(session, "EKEY>XCHAR",
                                    FORTH_HOST_EKEY_TO_XCHAR))
        return false;
    if (!forth_install_runtime_host(session, "BLOCK", FORTH_HOST_BLOCK))
        return false;
    if (!forth_install_runtime_host(session, "BUFFER", FORTH_HOST_BUFFER))
        return false;
    if (!forth_install_runtime_host(session, "UPDATE", FORTH_HOST_UPDATE))
        return false;
    if (!forth_install_runtime_host(session, "FLUSH", FORTH_HOST_FLUSH))
        return false;
    if (!forth_install_runtime_host(session, "SAVE-BUFFERS",
                                    FORTH_HOST_SAVE_BUFFERS))
        return false;
    if (!forth_install_runtime_host(session, "EMPTY-BUFFERS",
                                    FORTH_HOST_EMPTY_BUFFERS))
        return false;
    if (!forth_install_runtime_host(session, "LOAD", FORTH_HOST_LOAD))
        return false;
    if (!forth_install_runtime_host(session, "LIST", FORTH_HOST_LIST))
        return false;
    if (!forth_install_runtime_host(session, "THRU", FORTH_HOST_THRU))
        return false;

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

typedef struct {
    int64_t lo;
    int64_t hi;
    bool is_double;
    bool is_float;
    double fvalue;
} ForthParsedNumber;

static bool forth_parse_number(ForthSession *session, const uint8_t *name,
                               uint32_t len, ForthParsedNumber *out) {
    uint32_t i = 0;
    int sign = 1;
    int64_t base = 10;
    bool is_double = false;

    if (!session || !name || !out || len == 0) return false;
    memset(out, 0, sizeof(*out));
    if (len == 3 && name[0] == (uint8_t)'\'' && name[2] == (uint8_t)'\'') {
        out->lo = (int64_t)name[1];
        return true;
    }
    if (name[len - 1] == (uint8_t)'.') {
        is_double = true;
        len--;
        if (len == 0) return false;
    }
    if (!forth_fetch_cell(session, forth_base_addr(session), &base)) return false;
    if (base < 2 || base > 36) return false;
    if (name[0] == (uint8_t)'#') {
        base = 10;
        i = 1;
    } else if (name[0] == (uint8_t)'$') {
        base = 16;
        i = 1;
    } else if (name[0] == (uint8_t)'%') {
        base = 2;
        i = 1;
    }
    if (i >= len) return false;
    if (name[i] == (uint8_t)'-' && i + 1 < len) {
        sign = -1;
        i++;
    }
    if (i >= len) return false;
    if (is_double) {
        unsigned __int128 acc = 0;

        for (; i < len; i++) {
            int digit = forth_digit_value(name[i]);
            if (digit < 0 || (int64_t)digit >= base) return false;
            if (acc > (((unsigned __int128)-1) - (unsigned)digit)
                    / (unsigned __int128)(uint64_t)base)
                return false;
            acc = acc * (unsigned __int128)(uint64_t)base + (unsigned)digit;
        }
        if (sign < 0) acc = ~acc + 1;
        out->lo = (int64_t)(uint64_t)acc;
        out->hi = (int64_t)(uint64_t)(acc >> 64);
        out->is_double = true;
        return true;
    }
    {
        uint64_t uacc = 0;

        for (; i < len; i++) {
            int digit = forth_digit_value(name[i]);
            if (digit < 0 || (int64_t)digit >= base) return false;
            if (uacc > (UINT64_MAX - (uint64_t)digit) / (uint64_t)base)
                return false;
            uacc = uacc * (uint64_t)base + (uint64_t)digit;
        }
        if (sign < 0) uacc = 0u - uacc;
        out->lo = (int64_t)uacc;
        return true;
    }
}

static bool forth_parse_to_float(const uint8_t *s, uint32_t len, bool allow_blank,
                                 double *out) {
    uint32_t i = 0;
    int sign = 1;
    int exp_sign = 1;
    int expv = 0;
    bool have_digit = false;
    bool in_frac = false;
    double acc = 0.0;
    double place = 0.1;

    if (!s || !out) return false;
    while (i < len && forth_is_blank(s[i])) i++;
    if (i >= len) {
        if (allow_blank) {
            *out = 0.0;
            return true;
        }
        return false;
    }
    if (s[i] == (uint8_t)'+' || s[i] == (uint8_t)'-') {
        if (s[i] == (uint8_t)'-') sign = -1;
        i++;
    }
    while (i < len) {
        uint8_t c = s[i];
        if (c >= (uint8_t)'0' && c <= (uint8_t)'9') {
            have_digit = true;
            if (!in_frac) {
                acc = acc * 10.0 + (double)(c - (uint8_t)'0');
            } else {
                acc += place * (double)(c - (uint8_t)'0');
                place *= 0.1;
            }
            i++;
        } else if (c == (uint8_t)'.' && !in_frac) {
            in_frac = true;
            i++;
        } else {
            break;
        }
    }
    if (!have_digit) return false;
    if (i >= len) return false;
    {
        uint8_t mark = s[i];
        if (mark != (uint8_t)'E' && mark != (uint8_t)'e'
                && mark != (uint8_t)'D' && mark != (uint8_t)'d')
            return false;
    }
    i++;
    if (i < len && (s[i] == (uint8_t)'+' || s[i] == (uint8_t)'-')) {
        if (s[i] == (uint8_t)'-') exp_sign = -1;
        i++;
    }
    while (i < len && s[i] >= (uint8_t)'0' && s[i] <= (uint8_t)'9') {
        if (expv > 10000) return false;
        expv = expv * 10 + (int)(s[i] - (uint8_t)'0');
        i++;
    }
    if (i != len) return false;
    acc *= pow(10.0, (double)(exp_sign * expv));
    if (sign < 0) acc = -acc;
    *out = acc;
    return true;
}

static bool forth_dpop(ForthSession *session, int64_t *lo, int64_t *hi) {
    if (!forth_data_pop(session, hi)) return false;
    return forth_data_pop(session, lo);
}

static bool forth_dpush(ForthSession *session, int64_t lo, int64_t hi) {
    return forth_data_push(session, lo) && forth_data_push(session, hi);
}

static __int128 forth_pack_d(int64_t lo, int64_t hi) {
    return ((__int128)hi << 64) | (__int128)(uint64_t)lo;
}

static void forth_unpack_d(__int128 d, int64_t *lo, int64_t *hi) {
    *lo = (int64_t)(uint64_t)d;
    *hi = (int64_t)(uint64_t)((unsigned __int128)d >> 64);
}

static uint64_t forth_i64_absu(int64_t n, int *sign) {
    if (n < 0) {
        *sign = -1;
        return ~(uint64_t)n + 1;
    }
    *sign = 1;
    return (uint64_t)n;
}

static bool forth_udiv192(uint64_t hi64, unsigned __int128 lo128, uint64_t den,
                          unsigned __int128 *q, uint64_t *r) {
    unsigned __int128 n1;
    unsigned __int128 q1;
    unsigned __int128 r1;
    unsigned __int128 n0;

    if (den == 0 || !q || !r) return false;
    n1 = ((unsigned __int128)hi64 << 64) | (uint64_t)(lo128 >> 64);
    q1 = n1 / den;
    r1 = n1 % den;
    if ((q1 >> 64) != 0) return false;
    n0 = (r1 << 64) | (uint64_t)lo128;
    *q = (q1 << 64) | (n0 / den);
    *r = (uint64_t)(n0 % den);
    return true;
}

static bool forth_m_star_slash(ForthSession *session) {
    int64_t n2 = 0;
    int64_t n1 = 0;
    int64_t dlo = 0;
    int64_t dhi = 0;
    int dsign = 1;
    int n1sign = 1;
    int n2sign = 1;
    int rsign;
    uint64_t n1u;
    uint64_t n2u;
    unsigned __int128 mag;
    unsigned __int128 p0;
    unsigned __int128 p1;
    unsigned __int128 mid;
    unsigned __int128 prod_lo;
    uint64_t prod_hi;
    unsigned __int128 q;
    uint64_t rem = 0;
    int64_t qlo = 0;
    int64_t qhi = 0;

    if (!forth_data_pop(session, &n2) || n2 == 0)
        return forth_throw_now(session, -10);
    if (!forth_data_pop(session, &n1)) return false;
    if (!forth_dpop(session, &dlo, &dhi)) return false;
    mag = ((unsigned __int128)(uint64_t)dhi << 64) | (uint64_t)dlo;
    if (dhi < 0) {
        dsign = -1;
        mag = ~mag + 1;
    }
    n1u = forth_i64_absu(n1, &n1sign);
    n2u = forth_i64_absu(n2, &n2sign);
    rsign = dsign * n1sign * n2sign;
    p0 = (unsigned __int128)(uint64_t)mag * n1u;
    p1 = (mag >> 64) * (unsigned __int128)n1u;
    mid = (p0 >> 64) + (uint64_t)p1;
    prod_lo = ((unsigned __int128)(uint64_t)mid << 64) | (uint64_t)p0;
    prod_hi = (uint64_t)(p1 >> 64) + (uint64_t)(mid >> 64);
    if (!forth_udiv192(prod_hi, prod_lo, n2u, &q, &rem)) return false;
    if (rsign < 0) {
        if (rem != 0) q += 1;
        q = ~q + 1;
    }
    qlo = (int64_t)(uint64_t)q;
    qhi = (int64_t)(uint64_t)(q >> 64);
    return forth_dpush(session, qlo, qhi);
}

static int forth_host_dmath(ForthSession *session, uint8_t kind) {
    int64_t alo = 0;
    int64_t ahi = 0;
    int64_t blo = 0;
    int64_t bhi = 0;
    int64_t n = 0;
    __int128 a;
    __int128 b;
    unsigned __int128 ua;
    unsigned __int128 ub;

    switch (kind) {
    case FORTH_HOST_DPLUS:
        if (!forth_dpop(session, &blo, &bhi) || !forth_dpop(session, &alo, &ahi))
            return -1;
        forth_unpack_d(forth_pack_d(alo, ahi) + forth_pack_d(blo, bhi), &alo, &ahi);
        return forth_dpush(session, alo, ahi) ? 1 : -1;
    case FORTH_HOST_DMINUS:
        if (!forth_dpop(session, &blo, &bhi) || !forth_dpop(session, &alo, &ahi))
            return -1;
        forth_unpack_d(forth_pack_d(alo, ahi) - forth_pack_d(blo, bhi), &alo, &ahi);
        return forth_dpush(session, alo, ahi) ? 1 : -1;
    case FORTH_HOST_DNEGATE:
        if (!forth_dpop(session, &alo, &ahi)) return -1;
        forth_unpack_d(-forth_pack_d(alo, ahi), &alo, &ahi);
        return forth_dpush(session, alo, ahi) ? 1 : -1;
    case FORTH_HOST_DTWO_STAR:
        if (!forth_dpop(session, &alo, &ahi)) return -1;
        forth_unpack_d(forth_pack_d(alo, ahi) << 1, &alo, &ahi);
        return forth_dpush(session, alo, ahi) ? 1 : -1;
    case FORTH_HOST_DTWO_SLASH:
        if (!forth_dpop(session, &alo, &ahi)) return -1;
        forth_unpack_d(forth_pack_d(alo, ahi) >> 1, &alo, &ahi);
        return forth_dpush(session, alo, ahi) ? 1 : -1;
    case FORTH_HOST_DLESS:
        if (!forth_dpop(session, &blo, &bhi) || !forth_dpop(session, &alo, &ahi))
            return -1;
        a = forth_pack_d(alo, ahi);
        b = forth_pack_d(blo, bhi);
        return forth_data_push(session, a < b ? -1 : 0) ? 1 : -1;
    case FORTH_HOST_DEQUAL:
        if (!forth_dpop(session, &blo, &bhi) || !forth_dpop(session, &alo, &ahi))
            return -1;
        return forth_data_push(session, (alo == blo && ahi == bhi) ? -1 : 0) ? 1
                                                                            : -1;
    case FORTH_HOST_DABS:
        if (!forth_dpop(session, &alo, &ahi)) return -1;
        a = forth_pack_d(alo, ahi);
        if (a < 0) a = -a;
        forth_unpack_d(a, &alo, &ahi);
        return forth_dpush(session, alo, ahi) ? 1 : -1;
    case FORTH_HOST_DMAX:
        if (!forth_dpop(session, &blo, &bhi) || !forth_dpop(session, &alo, &ahi))
            return -1;
        a = forth_pack_d(alo, ahi);
        b = forth_pack_d(blo, bhi);
        if (a < b) {
            alo = blo;
            ahi = bhi;
        }
        return forth_dpush(session, alo, ahi) ? 1 : -1;
    case FORTH_HOST_DMIN:
        if (!forth_dpop(session, &blo, &bhi) || !forth_dpop(session, &alo, &ahi))
            return -1;
        a = forth_pack_d(alo, ahi);
        b = forth_pack_d(blo, bhi);
        if (a > b) {
            alo = blo;
            ahi = bhi;
        }
        return forth_dpush(session, alo, ahi) ? 1 : -1;
    case FORTH_HOST_MPLUS:
        if (!forth_data_pop(session, &n) || !forth_dpop(session, &alo, &ahi))
            return -1;
        forth_unpack_d(forth_pack_d(alo, ahi) + (__int128)n, &alo, &ahi);
        return forth_dpush(session, alo, ahi) ? 1 : -1;
    case FORTH_HOST_DULESS:
        if (!forth_dpop(session, &blo, &bhi) || !forth_dpop(session, &alo, &ahi))
            return -1;
        ua = ((unsigned __int128)(uint64_t)ahi << 64) | (uint64_t)alo;
        ub = ((unsigned __int128)(uint64_t)bhi << 64) | (uint64_t)blo;
        return forth_data_push(session, ua < ub ? -1 : 0) ? 1 : -1;
    case FORTH_HOST_M_STAR_SLASH:
        return forth_m_star_slash(session) ? 1 : -1;
    default:
        return -1;
    }
}

static bool forth_copy_from_vm(ForthSession *session, uint64_t addr, uint32_t n,
                               uint8_t *dst) {
    uint32_t i;
    if (n > 0 && !dst) return false;
    for (i = 0; i < n; i++) {
        if (!forth_fetch_byte(session, addr + i, &dst[i])) return false;
    }
    return true;
}

static bool forth_copy_to_vm(ForthSession *session, uint64_t addr, uint32_t n,
                             const uint8_t *src) {
    uint32_t i;
    if (n > 0 && !src) return false;
    for (i = 0; i < n; i++) {
        if (!forth_store_byte(session, addr + i, src[i])) return false;
    }
    return true;
}

static ForthSubst *forth_subst_find(ForthSession *session, const uint8_t *name,
                                    uint32_t nlen) {
    uint32_t i;
    for (i = 0; i < FORTH_SUBST_MAX; i++) {
        if (session->subst[i].used
                && names_equal(session->subst[i].name, session->subst[i].nlen,
                               name, nlen))
            return &session->subst[i];
    }
    return NULL;
}

static int forth_host_cmove(ForthSession *session, bool backward) {
    int64_t u = 0;
    int64_t dest = 0;
    int64_t src = 0;
    int64_t i;

    if (!forth_data_pop(session, &u) || u < 0) return -1;
    if (!forth_data_pop(session, &dest) || !forth_data_pop(session, &src))
        return -1;
    if (u == 0) return 1;
    if (!backward) {
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

static int forth_host_trailing(ForthSession *session) {
    int64_t u = 0;
    int64_t addr = 0;
    uint8_t ch = 0;

    if (!forth_data_pop(session, &u) || u < 0) return -1;
    if (!forth_data_pop(session, &addr)) return -1;
    while (u > 0) {
        if (!forth_fetch_byte(session, (uint64_t)addr + (uint64_t)u - 1, &ch))
            return -1;
        if (ch != (uint8_t)' ') break;
        u--;
    }
    if (!forth_data_push(session, addr)) return -1;
    return forth_data_push(session, u) ? 1 : -1;
}

static int forth_host_compare(ForthSession *session) {
    int64_t u2 = 0;
    int64_t a2 = 0;
    int64_t u1 = 0;
    int64_t a1 = 0;
    int64_t n;
    int64_t i;

    if (!forth_data_pop(session, &u2) || u2 < 0) return -1;
    if (!forth_data_pop(session, &a2) || !forth_data_pop(session, &u1) || u1 < 0)
        return -1;
    if (!forth_data_pop(session, &a1)) return -1;
    n = u1 < u2 ? u1 : u2;
    for (i = 0; i < n; i++) {
        uint8_t c1 = 0;
        uint8_t c2 = 0;
        if (!forth_fetch_byte(session, (uint64_t)a1 + (uint64_t)i, &c1)) return -1;
        if (!forth_fetch_byte(session, (uint64_t)a2 + (uint64_t)i, &c2)) return -1;
        if (c1 != c2)
            return forth_data_push(session, c1 < c2 ? -1 : 1) ? 1 : -1;
    }
    if (u1 == u2) return forth_data_push(session, 0) ? 1 : -1;
    return forth_data_push(session, u1 < u2 ? -1 : 1) ? 1 : -1;
}

static int forth_host_search(ForthSession *session) {
    int64_t u2 = 0;
    int64_t a2 = 0;
    int64_t u1 = 0;
    int64_t a1 = 0;
    int64_t i;
    int64_t j;

    if (!forth_data_pop(session, &u2) || u2 < 0) return -1;
    if (!forth_data_pop(session, &a2) || !forth_data_pop(session, &u1) || u1 < 0)
        return -1;
    if (!forth_data_pop(session, &a1)) return -1;
    if (u2 == 0) {
        if (!forth_data_push(session, a1) || !forth_data_push(session, u1))
            return -1;
        return forth_data_push(session, -1) ? 1 : -1;
    }
    if (u2 <= u1) {
        for (i = 0; i <= u1 - u2; i++) {
            int match = 1;
            for (j = 0; j < u2; j++) {
                uint8_t c1 = 0;
                uint8_t c2 = 0;
                if (!forth_fetch_byte(session, (uint64_t)a1 + (uint64_t)i
                                      + (uint64_t)j, &c1))
                    return -1;
                if (!forth_fetch_byte(session, (uint64_t)a2 + (uint64_t)j, &c2))
                    return -1;
                if (c1 != c2) {
                    match = 0;
                    break;
                }
            }
            if (match) {
                if (!forth_data_push(session, a1 + i)
                        || !forth_data_push(session, u1 - i))
                    return -1;
                return forth_data_push(session, -1) ? 1 : -1;
            }
        }
    }
    if (!forth_data_push(session, a1) || !forth_data_push(session, u1))
        return -1;
    return forth_data_push(session, 0) ? 1 : -1;
}

static int forth_host_sliteral(ForthSession *session) {
    int64_t u = 0;
    int64_t src = 0;
    uint64_t dest;
    uint32_t i;

    if (!forth_colon_is_open(session)) return -1;
    if (!forth_data_pop(session, &u) || u < 0) return -1;
    if (!forth_data_pop(session, &src)) return -1;
    dest = session->bump;
    if (!forth_dict_allot(session, u)) return -1;
    for (i = 0; i < (uint32_t)u; i++) {
        uint8_t ch = 0;
        if (!forth_fetch_byte(session, (uint64_t)src + i, &ch)) return -1;
        if (!forth_store_byte(session, dest + i, ch)) return -1;
    }
    if (!forth_colon_literal(session, (int64_t)dest)) return -1;
    return forth_colon_literal(session, u) ? 1 : -1;
}

static int forth_host_unescape(ForthSession *session) {
    int64_t dest = 0;
    int64_t u1 = 0;
    int64_t src = 0;
    int64_t i;
    int64_t o = 0;

    if (!forth_data_pop(session, &dest) || !forth_data_pop(session, &u1) || u1 < 0)
        return -1;
    if (!forth_data_pop(session, &src)) return -1;
    for (i = 0; i < u1; i++) {
        uint8_t ch = 0;
        if (!forth_fetch_byte(session, (uint64_t)src + (uint64_t)i, &ch))
            return -1;
        if (!forth_store_byte(session, (uint64_t)dest + (uint64_t)o, ch))
            return -1;
        o++;
        if (ch == (uint8_t)'%') {
            if (!forth_store_byte(session, (uint64_t)dest + (uint64_t)o, ch))
                return -1;
            o++;
        }
    }
    if (!forth_data_push(session, dest)) return -1;
    return forth_data_push(session, o) ? 1 : -1;
}

static int forth_host_replaces(ForthSession *session) {
    int64_t nlen = 0;
    int64_t naddr = 0;
    int64_t tlen = 0;
    int64_t taddr = 0;
    ForthSubst *slot;
    uint64_t copy;
    uint32_t i;

    if (!forth_data_pop(session, &nlen) || nlen < 0) return -1;
    if (!forth_data_pop(session, &naddr) || !forth_data_pop(session, &tlen)
            || tlen < 0)
        return -1;
    if (!forth_data_pop(session, &taddr)) return -1;
    if ((uint32_t)nlen > FORTH_SUBST_NAME_MAX) return -1;
    {
        uint8_t name[FORTH_SUBST_NAME_MAX];
        if ((uint32_t)nlen > 0
                && !forth_copy_from_vm(session, (uint64_t)naddr, (uint32_t)nlen,
                                       name))
            return -1;
        slot = forth_subst_find(session, name, (uint32_t)nlen);
        if (!slot) {
            for (i = 0; i < FORTH_SUBST_MAX; i++) {
                if (!session->subst[i].used) {
                    slot = &session->subst[i];
                    break;
                }
            }
        }
        if (!slot) return -1;
        memcpy(slot->name, name, (size_t)nlen);
        slot->nlen = (uint32_t)nlen;
        slot->used = true;
    }
    copy = session->bump;
    if (!forth_dict_allot(session, tlen)) return -1;
    for (i = 0; i < (uint32_t)tlen; i++) {
        uint8_t ch = 0;
        if (!forth_fetch_byte(session, (uint64_t)taddr + i, &ch)) return -1;
        if (!forth_store_byte(session, copy + i, ch)) return -1;
    }
    slot->text_addr = copy;
    slot->tlen = (uint32_t)tlen;
    return 1;
}

static int forth_host_substitute(ForthSession *session) {
    int64_t u2 = 0;
    int64_t dest = 0;
    int64_t u1 = 0;
    int64_t src = 0;
    uint8_t *inbuf = NULL;
    uint8_t *outbuf = NULL;
    uint32_t i = 0;
    uint32_t o = 0;
    int64_t nsub = 0;
    int rc = -1;

    if (!forth_data_pop(session, &u2) || u2 < 0) return -1;
    if (!forth_data_pop(session, &dest) || !forth_data_pop(session, &u1) || u1 < 0)
        return -1;
    if (!forth_data_pop(session, &src)) return -1;
    if (u1 > 0) {
        inbuf = malloc((size_t)u1);
        if (!inbuf) return -1;
        if (!forth_copy_from_vm(session, (uint64_t)src, (uint32_t)u1, inbuf))
            goto done;
    }
    if (u2 > 0) {
        outbuf = malloc((size_t)u2);
        if (!outbuf) goto done;
    }
    while (i < (uint32_t)u1) {
        if (inbuf[i] != (uint8_t)'%') {
            if (o >= (uint32_t)u2) {
                nsub = -1;
                goto overflow;
            }
            if (u2 > 0) outbuf[o] = inbuf[i];
            o++;
            i++;
            continue;
        }
        if (i + 1 < (uint32_t)u1 && inbuf[i + 1] == (uint8_t)'%') {
            if (o >= (uint32_t)u2) {
                nsub = -1;
                goto overflow;
            }
            if (u2 > 0) outbuf[o] = (uint8_t)'%';
            o++;
            i += 2;
            continue;
        }
        {
            uint32_t j = i + 1;
            ForthSubst *slot;
            while (j < (uint32_t)u1 && inbuf[j] != (uint8_t)'%') j++;
            if (j >= (uint32_t)u1) {
                while (i < (uint32_t)u1) {
                    if (o >= (uint32_t)u2) {
                        nsub = -1;
                        goto overflow;
                    }
                    if (u2 > 0) outbuf[o] = inbuf[i];
                    o++;
                    i++;
                }
                break;
            }
            slot = forth_subst_find(session, inbuf + i + 1, j - (i + 1));
            if (slot) {
                uint32_t k;
                if (o + slot->tlen > (uint32_t)u2) {
                    nsub = -1;
                    goto overflow;
                }
                for (k = 0; k < slot->tlen; k++) {
                    uint8_t ch = 0;
                    if (!forth_fetch_byte(session, slot->text_addr + k, &ch))
                        goto done;
                    if (u2 > 0) outbuf[o] = ch;
                    o++;
                }
                nsub++;
                i = j + 1;
            } else {
                uint32_t k;
                for (k = i; k <= j; k++) {
                    if (o >= (uint32_t)u2) {
                        nsub = -1;
                        goto overflow;
                    }
                    if (u2 > 0) outbuf[o] = inbuf[k];
                    o++;
                }
                i = j + 1;
            }
        }
    }
    if (o > 0 && !forth_copy_to_vm(session, (uint64_t)dest, o, outbuf)) goto done;
    if (!forth_data_push(session, dest) || !forth_data_push(session, (int64_t)o)
            || !forth_data_push(session, nsub))
        goto done;
    rc = 1;
    goto done;
overflow:
    if (!forth_data_push(session, dest) || !forth_data_push(session, 0)
            || !forth_data_push(session, -1))
        goto done;
    rc = 1;
done:
    free(inbuf);
    free(outbuf);
    return rc;
}

static int forth_host_string(ForthSession *session, uint8_t kind) {
    switch (kind) {
    case FORTH_HOST_TRAILING:
        return forth_host_trailing(session);
    case FORTH_HOST_CMOVE:
        return forth_host_cmove(session, false);
    case FORTH_HOST_CMOVE_UP:
        return forth_host_cmove(session, true);
    case FORTH_HOST_COMPARE:
        return forth_host_compare(session);
    case FORTH_HOST_SEARCH:
        return forth_host_search(session);
    case FORTH_HOST_SLITERAL:
        return forth_host_sliteral(session);
    case FORTH_HOST_UNESCAPE:
        return forth_host_unescape(session);
    case FORTH_HOST_REPLACES:
        return forth_host_replaces(session);
    case FORTH_HOST_SUBSTITUTE:
        return forth_host_substitute(session);
    default:
        return -1;
    }
}

static int forth_minimum_order(ForthSession *session) {
    ForthWid forth = forth_forth_wordlist(session);
    return forth_set_order(session, &forth, 1) ? 1 : -1;
}

static int forth_host_get_order(ForthSession *session) {
    ForthWid wids[FORTH_ORDER_MAX];
    uint32_t count = 0;
    uint32_t i;

    if (!forth_get_order(session, wids, FORTH_ORDER_MAX, &count)) return -1;
    for (i = count; i > 0; i--) {
        if (!forth_data_push(session, (int64_t)wids[i - 1])) return -1;
    }
    return forth_data_push(session, (int64_t)count) ? 1 : -1;
}

static int forth_host_set_order(ForthSession *session) {
    int64_t n = 0;
    ForthWid wids[FORTH_ORDER_MAX];
    uint32_t i;

    if (!forth_data_pop(session, &n)) return -1;
    if (n == -1) return forth_minimum_order(session);
    if (n < 0 || n > (int64_t)FORTH_ORDER_MAX) return -1;
    for (i = 0; i < (uint32_t)n; i++) {
        int64_t wid = 0;
        if (!forth_data_pop(session, &wid) || wid < 0
                || (uint64_t)wid > (uint64_t)UINT32_MAX)
            return -1;
        wids[i] = (ForthWid)wid;
    }
    return forth_set_order(session, wids, (uint32_t)n) ? 1 : -1;
}

static int forth_host_also(ForthSession *session) {
    ForthWid wids[FORTH_ORDER_MAX];
    uint32_t count = 0;
    uint32_t i;

    if (!forth_get_order(session, wids, FORTH_ORDER_MAX, &count)) return -1;
    if (count == 0 || count >= FORTH_ORDER_MAX) return -1;
    for (i = count; i > 0; i--) wids[i] = wids[i - 1];
    count++;
    return forth_set_order(session, wids, count) ? 1 : -1;
}

static int forth_host_previous(ForthSession *session) {
    ForthWid wids[FORTH_ORDER_MAX];
    uint32_t count = 0;

    if (!forth_get_order(session, wids, FORTH_ORDER_MAX, &count)) return -1;
    if (count == 0) return -1;
    return forth_set_order(session, wids + 1, count - 1) ? 1 : -1;
}

static int forth_host_forth(ForthSession *session) {
    ForthWid wids[FORTH_ORDER_MAX];
    uint32_t count = 0;
    ForthWid forth = forth_forth_wordlist(session);

    if (!forth_get_order(session, wids, FORTH_ORDER_MAX, &count)) return -1;
    if (count == 0) return forth_set_order(session, &forth, 1) ? 1 : -1;
    wids[0] = forth;
    return forth_set_order(session, wids, count) ? 1 : -1;
}

static int forth_host_definitions(ForthSession *session) {
    if (!session || session->order_count == 0) return -1;
    return forth_set_current(session, session->order[0]) ? 1 : -1;
}

static int forth_host_search_wordlist(ForthSession *session) {
    int64_t wid = 0;
    int64_t u = 0;
    int64_t caddr = 0;
    uint8_t name[FORTH_NAME_MAX];
    ForthXt xt = 0;
    bool imm = false;

    if (!forth_data_pop(session, &wid) || !forth_data_pop(session, &u) || u < 0
            || !forth_data_pop(session, &caddr))
        return -1;
    if (u > (int64_t)FORTH_NAME_MAX) return -1;
    if (u > 0
            && !forth_copy_from_vm(session, (uint64_t)caddr, (uint32_t)u, name))
        return -1;
    if (wid < 0 || (uint64_t)wid > (uint64_t)UINT32_MAX) return -1;
    if (forth_find_in_wid(session, (ForthWid)wid, (const char *)name,
                          (uint32_t)u, NULL, &xt, &imm)) {
        if (!forth_data_push(session, (int64_t)xt)) return -1;
        return forth_data_push(session, imm ? 1 : -1) ? 1 : -1;
    }
    return forth_data_push(session, 0) ? 1 : -1;
}

static int forth_host_order(ForthSession *session) {
    ForthWid wids[FORTH_ORDER_MAX];
    uint32_t count = 0;
    uint32_t i;
    uint32_t k;
    char buf[48];
    int n;

    if (!forth_get_order(session, wids, FORTH_ORDER_MAX, &count)) return -1;
    for (i = 0; i < count; i++) {
        n = snprintf(buf, sizeof(buf), "%u ", (unsigned)wids[i]);
        if (n < 0) return -1;
        for (k = 0; k < (uint32_t)n; k++) {
            if (!forth_emit_char(session, (uint8_t)buf[k])) return -1;
        }
    }
    n = snprintf(buf, sizeof(buf), "CURRENT %u\n",
                 (unsigned)forth_get_current(session));
    if (n < 0) return -1;
    for (k = 0; k < (uint32_t)n; k++) {
        if (!forth_emit_char(session, (uint8_t)buf[k])) return -1;
    }
    return 1;
}

static int forth_host_search_order(ForthSession *session, uint8_t kind) {
    ForthWid wid = 0;
    int64_t cell = 0;

    switch (kind) {
    case FORTH_HOST_WORDLIST:
        if (!forth_wordlist_create(session, &wid)) return -1;
        return forth_data_push(session, (int64_t)wid) ? 1 : -1;
    case FORTH_HOST_GET_ORDER:
        return forth_host_get_order(session);
    case FORTH_HOST_SET_ORDER:
        return forth_host_set_order(session);
    case FORTH_HOST_GET_CURRENT:
        return forth_data_push(session, (int64_t)forth_get_current(session))
            ? 1 : -1;
    case FORTH_HOST_SET_CURRENT:
        if (!forth_data_pop(session, &cell) || cell < 0
                || (uint64_t)cell > (uint64_t)UINT32_MAX)
            return -1;
        return forth_set_current(session, (ForthWid)cell) ? 1 : -1;
    case FORTH_HOST_FORTH_WORDLIST:
        return forth_data_push(session,
                               (int64_t)forth_forth_wordlist(session)) ? 1 : -1;
    case FORTH_HOST_ALSO:
        return forth_host_also(session);
    case FORTH_HOST_PREVIOUS:
        return forth_host_previous(session);
    case FORTH_HOST_ONLY:
        return forth_minimum_order(session);
    case FORTH_HOST_FORTH:
        return forth_host_forth(session);
    case FORTH_HOST_DEFINITIONS:
        return forth_host_definitions(session);
    case FORTH_HOST_SEARCH_WORDLIST:
        return forth_host_search_wordlist(session);
    case FORTH_HOST_ORDER:
        return forth_host_order(session);
    default:
        return -1;
    }
}

static int64_t forth_saved_errno(void) {
    int err = errno;
    return err == 0 ? (int64_t)-1 : (int64_t)err;
}

static FILE *forth_fp_from_id(ForthSession *session, uint32_t fileid,
                              uint32_t *slot_out) {
    uint32_t slot;
    if (!decode_fileid(session, fileid, &slot)) return NULL;
    if (slot_out) *slot_out = slot;
    return session->files[slot].fp;
}

static bool forth_pop_path(ForthSession *session, char *out, size_t cap) {
    int64_t u = 0;
    int64_t caddr = 0;
    uint32_t i;

    if (!out || cap < 2) return false;
    if (!forth_data_pop(session, &u) || u < 0) return false;
    if (!forth_data_pop(session, &caddr)) return false;
    if ((uint64_t)u >= cap) return false;
    for (i = 0; i < (uint32_t)u; i++) {
        uint8_t ch = 0;
        if (!forth_fetch_byte(session, (uint64_t)caddr + i, &ch)) return false;
        out[i] = (char)ch;
    }
    out[u] = '\0';
    return true;
}

static bool forth_already_required(const ForthSession *session, const char *path) {
    uint32_t i;
    if (!session || !path) return false;
    for (i = 0; i < session->required_count; i++) {
        if (strcmp(session->required[i], path) == 0) return true;
    }
    return false;
}

static bool forth_mark_required(ForthSession *session, const char *path) {
    if (!session || !path || path[0] == '\0') return false;
    if (forth_already_required(session, path)) return true;
    if (session->required_count >= FORTH_REQUIRED_MAX) return false;
    strncpy(session->required[session->required_count], path, FORTH_PATH_MAX - 1);
    session->required[session->required_count][FORTH_PATH_MAX - 1] = '\0';
    session->required_count++;
    return true;
}

static bool forth_resolve_include_path(ForthSession *session, const char *name,
                                       char *out, size_t cap) {
    struct stat st;
    uint32_t d;

    if (!session || !name || name[0] == '\0' || !out || cap < 2) return false;
    if (name[0] == '/') {
        strncpy(out, name, cap - 1);
        out[cap - 1] = '\0';
        return true;
    }
    if (stat(name, &st) == 0) {
        strncpy(out, name, cap - 1);
        out[cap - 1] = '\0';
        return true;
    }
    for (d = session->source_depth; d > 0; d--) {
        const ForthSourceFrame *frame = &session->sources[d - 1];
        uint32_t slot;
        const char *base;
        const char *slash;
        size_t dirlen;
        size_t nlen;

        if (frame->kind != FORTH_SRC_FILE) continue;
        if (!decode_fileid(session, frame->fileid, &slot)) continue;
        base = session->files[slot].path;
        slash = strrchr(base, '/');
        if (!slash) continue;
        dirlen = (size_t)(slash - base);
        nlen = strlen(name);
        if (dirlen + 1 + nlen + 1 > cap) continue;
        memcpy(out, base, dirlen);
        out[dirlen] = '/';
        memcpy(out + dirlen + 1, name, nlen + 1);
        if (stat(out, &st) == 0) return true;
    }
    strncpy(out, name, cap - 1);
    out[cap - 1] = '\0';
    return true;
}

static const char *forth_fam_mode(int64_t fam, bool create) {
    int64_t acc = fam & 3;

    if (create) {
        if (acc == FORTH_FAM_WO) return "wb";
        return "w+b";
    }
    if (acc == FORTH_FAM_RO) return "rb";
    return "r+b";
}

static bool forth_pop_fileid(ForthSession *session, uint32_t *fileid) {
    int64_t cell = 0;
    if (!fileid || !forth_data_pop(session, &cell)) return false;
    if (cell < 0 || cell > (int64_t)UINT32_MAX) return false;
    *fileid = (uint32_t)cell;
    return true;
}

static int forth_push_ud_ior(ForthSession *session, uint64_t value, int64_t ior) {
    if (!forth_data_push(session, (int64_t)value)) return -1;
    if (!forth_data_push(session, 0)) return -1;
    return forth_data_push(session, ior) ? 1 : -1;
}

static int forth_include_fileid(ForthSession *session, uint32_t fileid) {
    ForthSession *prev;
    bool ok = true;

    if (!forth_source_push_file(session, fileid)) return -1;
    prev = g_forth;
    g_forth = session;
    while (forth_refill(session)) {
        if (!forth_interpret_loop(session)) {
            ok = false;
            break;
        }
        if (session->exit_requested) break;
        if (session->quit_requested) {
            session->quit_requested = false;
            break;
        }
    }
    g_forth = prev;
    if (ok && forth_colon_is_open(session)) ok = false;
    if (!forth_source_pop(session)) ok = false;
    return ok ? 1 : -1;
}

static int forth_included_path(ForthSession *session, const char *path) {
    uint32_t fileid = 0;
    int rc;

    if (!path || path[0] == '\0') return -1;
    if (!forth_file_open(session, path, "rb", &fileid)
            && !forth_file_open(session, path, "r", &fileid))
        return -1;
    rc = forth_include_fileid(session, fileid);
    forth_file_close(session, fileid);
    if (rc > 0 && !forth_mark_required(session, path)) return -1;
    return rc;
}

static int forth_host_included(ForthSession *session) {
    char name[FORTH_PATH_MAX];
    char resolved[FORTH_PATH_MAX];

    if (!forth_pop_path(session, name, sizeof(name))) return -1;
    if (!forth_resolve_include_path(session, name, resolved, sizeof(resolved)))
        return -1;
    return forth_included_path(session, resolved);
}

static int forth_host_required(ForthSession *session) {
    char name[FORTH_PATH_MAX];
    char resolved[FORTH_PATH_MAX];

    if (!forth_pop_path(session, name, sizeof(name))) return -1;
    if (!forth_resolve_include_path(session, name, resolved, sizeof(resolved)))
        return -1;
    if (forth_already_required(session, resolved)) return 1;
    return forth_included_path(session, resolved);
}

static int forth_host_include_word(ForthSession *session, bool required_only) {
    int pr = forth_host_parse_name(session);
    if (pr < 0) return -1;
    return required_only ? forth_host_required(session)
                         : forth_host_included(session);
}

static int forth_host_open_create(ForthSession *session, bool create) {
    int64_t fam = 0;
    char path[FORTH_PATH_MAX];
    uint32_t fileid = 0;
    const char *mode;

    if (!forth_data_pop(session, &fam)) return -1;
    if (!forth_pop_path(session, path, sizeof(path))) return -1;
    mode = forth_fam_mode(fam, create);
    errno = 0;
    if (!forth_file_open(session, path, mode, &fileid)) {
        if (!forth_data_push(session, 0)) return -1;
        return forth_data_push(session, forth_saved_errno()) ? 1 : -1;
    }
    if (!forth_data_push(session, (int64_t)fileid)) return -1;
    return forth_data_push(session, 0) ? 1 : -1;
}

static int forth_host_read_line(ForthSession *session) {
    int64_t u1 = 0;
    int64_t caddr = 0;
    uint32_t fileid = 0;
    FILE *fp;
    int c;
    uint32_t n = 0;
    bool got_term = false;
    bool any = false;

    if (!forth_pop_fileid(session, &fileid)) return -1;
    if (!forth_data_pop(session, &u1) || u1 < 0) return -1;
    if (!forth_data_pop(session, &caddr)) return -1;
    fp = forth_fp_from_id(session, fileid, NULL);
    if (!fp) {
        if (!forth_data_push(session, 0) || !forth_data_push(session, 0))
            return -1;
        return forth_data_push(session, (int64_t)-1) ? 1 : -1;
    }
    if (u1 == 0) {
        c = fgetc(fp);
        if (c == EOF) {
            if (!forth_data_push(session, 0) || !forth_data_push(session, 0))
                return -1;
            return forth_data_push(session, 0) ? 1 : -1;
        }
        ungetc(c, fp);
        if (!forth_data_push(session, 0) || !forth_data_push(session, (int64_t)-1))
            return -1;
        return forth_data_push(session, 0) ? 1 : -1;
    }
    for (;;) {
        c = fgetc(fp);
        if (c == EOF) break;
        any = true;
        if (c == '\n') {
            got_term = true;
            break;
        }
        if (c == '\r') {
            int next = fgetc(fp);
            if (next != '\n' && next != EOF) ungetc(next, fp);
            got_term = true;
            break;
        }
        if ((uint64_t)n >= (uint64_t)u1) {
            ungetc(c, fp);
            got_term = true;
            break;
        }
        if (!forth_store_byte(session, (uint64_t)caddr + n, (uint8_t)c))
            return -1;
        n++;
    }
    if (!any && !got_term) {
        if (!forth_data_push(session, 0) || !forth_data_push(session, 0))
            return -1;
        return forth_data_push(session, 0) ? 1 : -1;
    }
    if (!got_term && n > 0) got_term = true;
    if (!forth_data_push(session, (int64_t)n)) return -1;
    if (!forth_data_push(session, got_term ? (int64_t)-1 : 0)) return -1;
    return forth_data_push(session, 0) ? 1 : -1;
}

static int forth_host_file(ForthSession *session, uint8_t kind) {
    int64_t fam = 0;
    int64_t u = 0;
    int64_t caddr = 0;
    int64_t lo = 0;
    int64_t hi = 0;
    uint32_t fileid = 0;
    FILE *fp;
    char path[FORTH_PATH_MAX];
    char dest[FORTH_PATH_MAX];
    struct stat st;
    uint32_t i;
    int c;
    uint32_t n;

    switch (kind) {
    case FORTH_HOST_BIN:
        if (!forth_data_pop(session, &fam)) return -1;
        return forth_data_push(session, fam | FORTH_FAM_BIN) ? 1 : -1;
    case FORTH_HOST_OPEN_FILE:
        return forth_host_open_create(session, false);
    case FORTH_HOST_CREATE_FILE:
        return forth_host_open_create(session, true);
    case FORTH_HOST_CLOSE_FILE:
        if (!forth_pop_fileid(session, &fileid))
            return forth_data_push(session, (int64_t)-1) ? 1 : -1;
        errno = 0;
        if (!forth_file_close(session, fileid))
            return forth_data_push(session, forth_saved_errno()) ? 1 : -1;
        return forth_data_push(session, 0) ? 1 : -1;
    case FORTH_HOST_DELETE_FILE:
        if (!forth_pop_path(session, path, sizeof(path))) return -1;
        errno = 0;
        if (unlink(path) != 0)
            return forth_data_push(session, forth_saved_errno()) ? 1 : -1;
        return forth_data_push(session, 0) ? 1 : -1;
    case FORTH_HOST_READ_FILE:
        if (!forth_pop_fileid(session, &fileid)) return -1;
        if (!forth_data_pop(session, &u) || u < 0) return -1;
        if (!forth_data_pop(session, &caddr)) return -1;
        fp = forth_fp_from_id(session, fileid, NULL);
        if (!fp) {
            if (!forth_data_push(session, 0)) return -1;
            return forth_data_push(session, (int64_t)-1) ? 1 : -1;
        }
        n = 0;
        while ((uint64_t)n < (uint64_t)u) {
            c = fgetc(fp);
            if (c == EOF) break;
            if (!forth_store_byte(session, (uint64_t)caddr + n, (uint8_t)c))
                return -1;
            n++;
        }
        if (!forth_data_push(session, (int64_t)n)) return -1;
        return forth_data_push(session, 0) ? 1 : -1;
    case FORTH_HOST_READ_LINE:
        return forth_host_read_line(session);
    case FORTH_HOST_WRITE_FILE:
    case FORTH_HOST_WRITE_LINE:
        if (!forth_pop_fileid(session, &fileid)) return -1;
        if (!forth_data_pop(session, &u) || u < 0) return -1;
        if (!forth_data_pop(session, &caddr)) return -1;
        fp = forth_fp_from_id(session, fileid, NULL);
        if (!fp) return forth_data_push(session, (int64_t)-1) ? 1 : -1;
        for (i = 0; i < (uint32_t)u; i++) {
            uint8_t ch = 0;
            if (!forth_fetch_byte(session, (uint64_t)caddr + i, &ch)) return -1;
            if (fputc((int)ch, fp) == EOF)
                return forth_data_push(session, forth_saved_errno()) ? 1 : -1;
        }
        if (kind == FORTH_HOST_WRITE_LINE && fputc('\n', fp) == EOF)
            return forth_data_push(session, forth_saved_errno()) ? 1 : -1;
        return forth_data_push(session, 0) ? 1 : -1;
    case FORTH_HOST_FILE_POSITION:
        if (!forth_pop_fileid(session, &fileid)) return -1;
        fp = forth_fp_from_id(session, fileid, NULL);
        if (!fp) return forth_push_ud_ior(session, 0, (int64_t)-1);
        {
            long pos = ftell(fp);
            if (pos < 0) return forth_push_ud_ior(session, 0, forth_saved_errno());
            return forth_push_ud_ior(session, (uint64_t)pos, 0);
        }
    case FORTH_HOST_FILE_SIZE:
        if (!forth_pop_fileid(session, &fileid)) return -1;
        fp = forth_fp_from_id(session, fileid, NULL);
        if (!fp) return forth_push_ud_ior(session, 0, (int64_t)-1);
        if (fstat(fileno(fp), &st) != 0)
            return forth_push_ud_ior(session, 0, forth_saved_errno());
        return forth_push_ud_ior(session, (uint64_t)st.st_size, 0);
    case FORTH_HOST_REPOSITION_FILE:
        if (!forth_pop_fileid(session, &fileid)) return -1;
        if (!forth_data_pop(session, &hi) || !forth_data_pop(session, &lo))
            return -1;
        fp = forth_fp_from_id(session, fileid, NULL);
        if (!fp || hi != 0 || lo < 0)
            return forth_data_push(session, (int64_t)-1) ? 1 : -1;
        if (fseek(fp, (long)lo, SEEK_SET) != 0)
            return forth_data_push(session, forth_saved_errno()) ? 1 : -1;
        return forth_data_push(session, 0) ? 1 : -1;
    case FORTH_HOST_RESIZE_FILE:
        if (!forth_pop_fileid(session, &fileid)) return -1;
        if (!forth_data_pop(session, &hi) || !forth_data_pop(session, &lo))
            return -1;
        fp = forth_fp_from_id(session, fileid, NULL);
        if (!fp || hi != 0 || lo < 0)
            return forth_data_push(session, (int64_t)-1) ? 1 : -1;
        if (fflush(fp) != 0 || ftruncate(fileno(fp), (off_t)lo) != 0)
            return forth_data_push(session, forth_saved_errno()) ? 1 : -1;
        return forth_data_push(session, 0) ? 1 : -1;
    case FORTH_HOST_FLUSH_FILE:
        if (!forth_pop_fileid(session, &fileid))
            return forth_data_push(session, (int64_t)-1) ? 1 : -1;
        fp = forth_fp_from_id(session, fileid, NULL);
        if (!fp) return forth_data_push(session, (int64_t)-1) ? 1 : -1;
        if (fflush(fp) != 0)
            return forth_data_push(session, forth_saved_errno()) ? 1 : -1;
        return forth_data_push(session, 0) ? 1 : -1;
    case FORTH_HOST_RENAME_FILE:
        if (!forth_pop_path(session, dest, sizeof(dest))) return -1;
        if (!forth_pop_path(session, path, sizeof(path))) return -1;
        errno = 0;
        if (rename(path, dest) != 0)
            return forth_data_push(session, forth_saved_errno()) ? 1 : -1;
        return forth_data_push(session, 0) ? 1 : -1;
    case FORTH_HOST_FILE_STATUS:
        if (!forth_pop_path(session, path, sizeof(path))) return -1;
        errno = 0;
        if (stat(path, &st) != 0) {
            if (!forth_data_push(session, 0)) return -1;
            return forth_data_push(session, forth_saved_errno()) ? 1 : -1;
        }
        if (!forth_data_push(session, 0)) return -1;
        return forth_data_push(session, 0) ? 1 : -1;
    case FORTH_HOST_INCLUDED:
        return forth_host_included(session);
    case FORTH_HOST_INCLUDE:
        return forth_host_include_word(session, false);
    case FORTH_HOST_INCLUDE_FILE:
        if (!forth_pop_fileid(session, &fileid)) return -1;
        return forth_include_fileid(session, fileid);
    case FORTH_HOST_REQUIRED:
        return forth_host_required(session);
    case FORTH_HOST_REQUIRE:
        return forth_host_include_word(session, true);
    default:
        return -1;
    }
}

static int forth_host_mem(ForthSession *session, uint8_t kind) {
    int64_t n = 0;
    int64_t addr = 0;
    uint64_t got = 0;

    switch (kind) {
    case FORTH_HOST_ALLOCATE:
        if (!forth_data_pop(session, &n)) return -1;
        if (n <= 0 || !forth_heap_allocate(session, (uint64_t)n, &got)) {
            if (!forth_data_push(session, 0)) return -1;
            return forth_data_push(session, (int64_t)-1) ? 1 : -1;
        }
        if (!forth_data_push(session, (int64_t)got)) return -1;
        return forth_data_push(session, 0) ? 1 : -1;
    case FORTH_HOST_MEM_FREE:
        if (!forth_data_pop(session, &addr)) return -1;
        if (addr < (int64_t)FORTH_HEAP_BASE
                || !forth_free(session, (uint64_t)addr))
            return forth_data_push(session, (int64_t)-1) ? 1 : -1;
        return forth_data_push(session, 0) ? 1 : -1;
    case FORTH_HOST_MEM_RESIZE:
        if (!forth_data_pop(session, &n) || !forth_data_pop(session, &addr))
            return -1;
        if (addr <= 0 || n <= 0
                || !forth_resize(session, (uint64_t)addr, (uint64_t)n, &got)) {
            if (!forth_data_push(session, addr)) return -1;
            return forth_data_push(session, (int64_t)-1) ? 1 : -1;
        }
        if (!forth_data_push(session, (int64_t)got)) return -1;
        return forth_data_push(session, 0) ? 1 : -1;
    default:
        return -1;
    }
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
    if (!session) return false;
    if (n == 0) return true;
    if (n < 0) {
        if (n == INT64_MIN) return false;
        size = (uint64_t)(-n);
        if (session->bump < size) return false;
        session->bump -= size;
        return forth_store_cell(session, session->here_cell_addr,
                                (int64_t)session->bump);
    }
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

static bool forth_store_s_quote(ForthSession *session, uint64_t src, uint32_t len,
                                const uint8_t *bytes, uint64_t *addr_out) {
    uint64_t dest;
    uint32_t i;
    uint32_t stored = len;

    if (!session || !addr_out) return false;
    if (stored > FORTH_WORD_MAX) stored = FORTH_WORD_MAX;
    dest = session->s_quote_addr[session->s_quote_sel & 1u];
    session->s_quote_sel ^= 1u;
    for (i = 0; i < stored; i++) {
        uint8_t ch = 0;
        if (bytes) ch = bytes[i];
        else if (!forth_fetch_byte(session, src + i, &ch)) return false;
        if (!forth_store_byte(session, dest + i, ch)) return false;
    }
    *addr_out = dest;
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
    if (strcmp(name, "DOUBLE") == 0 || strcmp(name, "DOUBLE-EXT") == 0
            || strcmp(name, "STRING") == 0 || strcmp(name, "STRING-EXT") == 0
            || strcmp(name, "SEARCH-ORDER") == 0
            || strcmp(name, "SEARCH-ORDER-EXT") == 0
            || strcmp(name, "FILE") == 0 || strcmp(name, "FILE-EXT") == 0
            || strcmp(name, "MEMORY-ALLOC") == 0
            || strcmp(name, "LOCALS") == 0
            || strcmp(name, "LOCALS-EXT") == 0
            || strcmp(name, "FACILITY") == 0
            || strcmp(name, "FACILITY-EXT") == 0
            || strcmp(name, "TOOLS") == 0
            || strcmp(name, "TOOLS-EXT") == 0
            || strcmp(name, "FLOATING") == 0
            || strcmp(name, "FLOATING-EXT") == 0
            || strcmp(name, "XCHAR") == 0
            || strcmp(name, "XCHAR-EXT") == 0
            || strcmp(name, "BLOCK") == 0
            || strcmp(name, "BLOCK-EXT") == 0) {
        return forth_data_push(session, -1) && forth_data_push(session, -1);
    }
    if (strcmp(name, "XCHAR-ENCODING") == 0) {
        return forth_data_push(session, (int64_t)session->xchar_enc_addr)
            && forth_data_push(session, 5)
            && forth_data_push(session, -1);
    }
    if (strcmp(name, "MAX-XCHAR") == 0) {
        return forth_data_push(session, 0x10FFFF)
            && forth_data_push(session, -1);
    }
    if (strcmp(name, "XCHAR-MAXMEM") == 0) {
        return forth_data_push(session, 4) && forth_data_push(session, -1);
    }
    if (strcmp(name, "FLOATING-STACK") == 0) {
        return forth_data_push(session, FORTH_FLOAT_STACK_CELLS)
            && forth_data_push(session, -1);
    }
    if (strcmp(name, "MAX-FLOAT") == 0) {
        return forth_float_push(session, DBL_MAX)
            && forth_data_push(session, -1);
    }
    if (strcmp(name, "#LOCALS") == 0) {
        return forth_data_push(session, FORTH_LOCAL_MAX)
            && forth_data_push(session, -1);
    }
    if (strcmp(name, "WORDLISTS") == 0) {
        return forth_data_push(session, FORTH_WORDLIST_MAX)
            && forth_data_push(session, -1);
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
    uint8_t code[256];
    uint32_t off = 0;
    NvmModule *mod;
    NvmFunctionEntry fn;
    NvmVerifyResult verified;
    uint32_t before;

    if (!forth_data_pop(session, &does_xt)) return false;
    nt = session->does_child_nt ? session->does_child_nt : forth_latest(session);
    header = header_at(session, nt);
    if (!header) return false;
    mod = session->module;
    if ((uint32_t)does_xt >= mod->function_count) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_PUSH_I64, (int64_t)nt))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_CALL, session->dpush_fn))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_PUSH_I64,
                   (int64_t)FORTH_HOST_DOES_ENTER))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_CALL_EXTERN, session->runtime_import))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_POP)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_PUSH_I64, (int64_t)header->data_addr))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_CALL, session->dpush_fn))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_CALL, (uint32_t)does_xt))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_PUSH_I64, (int64_t)0))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_CALL, session->dpush_fn))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_PUSH_I64,
                   (int64_t)FORTH_HOST_DOES_ENTER))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_CALL_EXTERN, session->runtime_import))
        return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_POP)) return false;
    if (!wrap_emit(code, &off, sizeof(code), OP_RET)) return false;
    memset(&fn, 0, sizeof(fn));
    fn.name_idx = nvm_add_string(mod, "nl_forth_does_run", 17);
    fn.arity = 0;
    fn.code_offset = nvm_append_code(mod, code, off);
    fn.code_length = off;
    fn.local_count = 0;
    fn.result_tag = TAG_VOID;
    fn.result_count = 0;
    before = mod->function_count;
    child = nvm_add_function(mod, &fn);
    if (mod->function_count != before + 1) return false;
    verified = nvm_verify_function(mod, child);
    if (!verified.ok) return false;
    header->xt = child;
    if (session->vm_exec_depth != 0) {
        session->does_rebuild_pending = true;
        return vm_sync_new_functions(&session->vm, session->module);
    }
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

static bool forth_colon_host_runtime(ForthSession *session, uint16_t host) {
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
    return forth_throw_now(session, -2) ? 1 : -1;
}

static int forth_host_key(ForthSession *session) {
    uint64_t caddr = 0;
    uint64_t u = 0;
    int64_t to_in = 0;
    uint8_t ch = 0;
    int got;

    if (forth_source_id(session) != 0) {
        if (!isatty(STDIN_FILENO)) return -1;
        got = getchar();
        if (got == EOF) return -1;
        return forth_data_push(session, (int64_t)(uint8_t)got) ? 1 : -1;
    }
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
    if (forth_source_id(session) != 0) {
        if (!isatty(STDIN_FILENO))
            return forth_data_push(session, 0) ? 1 : -1;
        while (n2 < (uint32_t)n1) {
            int got = getchar();
            if (got == EOF || got == '\n' || got == '\r') break;
            if (!forth_store_byte(session, (uint64_t)dest + n2, (uint8_t)got))
                return -1;
            n2++;
        }
        return forth_data_push(session, (int64_t)n2) ? 1 : -1;
    }
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

static bool forth_plusloop_step(ForthSession *session) {
    int64_t n = 0;
    int64_t index = 0;
    int64_t limit = 0;
    uint64_t uold;
    uint64_t unew;
    uint64_t ulimit;
    bool done;

    if (!forth_data_pop(session, &n)) return false;
    if (!forth_return_pop(session, &index)) return false;
    if (!forth_return_pop(session, &limit)) return false;
    uold = (uint64_t)index;
    unew = uold + (uint64_t)n;
    ulimit = (uint64_t)limit;
    if (n == 0) {
        done = false;
    } else if (n > 0) {
        if (unew > uold)
            done = (ulimit > uold && ulimit <= unew);
        else
            done = (ulimit > uold || ulimit <= unew);
    } else if (unew < uold) {
        done = (ulimit <= uold && ulimit > unew);
    } else {
        done = (ulimit <= uold || ulimit > unew);
    }
    if (!done) {
        if (!forth_return_push(session, limit)) return false;
        if (!forth_return_push(session, (int64_t)unew)) return false;
        return forth_data_push(session, (int64_t)-1);
    }
    return forth_data_push(session, (int64_t)0);
}

static bool source_next_char(ForthSession *session, uint8_t *ch, bool *have) {
    uint64_t caddr = 0;
    uint64_t u = 0;
    int64_t to_in = 0;

    if (!ch || !have) return false;
    if (!forth_source(session, &caddr, &u)) return false;
    if (!forth_fetch_cell(session, session->sysvars, &to_in)) return false;
    if (to_in < 0 || (uint64_t)to_in >= u) {
        *have = false;
        return true;
    }
    if (!forth_fetch_byte(session, caddr + (uint64_t)to_in, ch)) return false;
    to_in++;
    if (!forth_store_cell(session, session->sysvars, to_in)) return false;
    *have = true;
    return true;
}

static int forth_hex_nibble(uint8_t ch) {
    int digit = forth_digit_value(ch);
    if (digit < 0 || digit > 15) return -1;
    return digit;
}

static int forth_host_parse_name(ForthSession *session) {
    uint64_t src = 0;
    uint32_t wlen = 0;
    if (!forth_parse_delimited(session, (uint8_t)' ', true, &src, &wlen))
        return -1;
    if (!forth_data_push(session, (int64_t)src)) return -1;
    return forth_data_push(session, (int64_t)wlen) ? 1 : -1;
}

static int forth_host_value(ForthSession *session) {
    uint8_t def[FORTH_NAME_MAX];
    uint32_t deflen = 0;
    ForthNt published = 0;
    ForthHeader *header;
    uint64_t addr;
    int64_t init = 0;
    int got;

    if (!forth_data_pop(session, &init)) return -1;
    got = forth_take_word(session, def, &deflen);
    if (got <= 0) return -1;
    if (!forth_dict_align(session)) return -1;
    addr = session->bump;
    if (!forth_dict_allot(session, (int64_t)FORTH_CELL_BYTES)) return -1;
    if (!forth_store_cell(session, addr, init)) return -1;
    if (!forth_colon_begin(session, (const char *)def, deflen)) return -1;
    if (!forth_colon_literal(session, (int64_t)addr)) {
        forth_colon_abort(session);
        return -1;
    }
    if (!colon_call_named(session, "@")) {
        forth_colon_abort(session);
        return -1;
    }
    if (!forth_colon_finish(session, &published)) return -1;
    header = header_at(session, published);
    if (!header) return -1;
    header->data_addr = addr;
    header->body_cells = 1;
    return 1;
}

static int forth_host_two_value(ForthSession *session) {
    uint8_t def[FORTH_NAME_MAX];
    uint32_t deflen = 0;
    ForthNt published = 0;
    ForthHeader *header;
    uint64_t addr;
    int64_t x1 = 0;
    int64_t x2 = 0;
    int got;

    if (!forth_data_pop(session, &x2) || !forth_data_pop(session, &x1)) return -1;
    got = forth_take_word(session, def, &deflen);
    if (got <= 0) return -1;
    if (!forth_dict_align(session)) return -1;
    addr = session->bump;
    if (!forth_dict_allot(session, (int64_t)(FORTH_CELL_BYTES * 2))) return -1;
    if (!forth_store_cell(session, addr, x2)) return -1;
    if (!forth_store_cell(session, addr + FORTH_CELL_BYTES, x1)) return -1;
    if (!forth_colon_begin(session, (const char *)def, deflen)) return -1;
    if (!forth_colon_literal(session, (int64_t)addr)) {
        forth_colon_abort(session);
        return -1;
    }
    if (!colon_call_named(session, "2@")) {
        forth_colon_abort(session);
        return -1;
    }
    if (!forth_colon_finish(session, &published)) return -1;
    header = header_at(session, published);
    if (!header) return -1;
    header->data_addr = addr;
    header->body_cells = 2;
    return 1;
}

static int forth_host_body_op(ForthSession *session, int64_t state, bool fetch) {
    uint8_t name[FORTH_NAME_MAX];
    uint32_t nlen = 0;
    ForthNt nt = 0;
    ForthXt xt = 0;
    bool immediate = false;
    ForthHeader *header;
    int got;
    int64_t cell = 0;
    int slot;

    got = forth_take_word(session, name, &nlen);
    if (got <= 0) return -1;
    slot = forth_local_slot(session, name, nlen);
    if (slot >= 0) {
        if (fetch) return -1;
        if (state == 0) return -1;
        return forth_colon_local_store(session, slot) ? 1 : -1;
    }
    if (!forth_find(session, (const char *)name, nlen, &nt, &xt, &immediate))
        return -1;
    header = header_at(session, nt);
    if (!header || header->data_addr == 0) return -1;
    if (header->body_cells == 2) {
        if (fetch) {
            if (state != 0) {
                if (!forth_colon_literal(session, (int64_t)header->data_addr))
                    return -1;
                return colon_call_named(session, "2@") ? 1 : -1;
            }
            if (!forth_fetch_cell(session, header->data_addr + FORTH_CELL_BYTES,
                                  &cell))
                return -1;
            if (!forth_data_push(session, cell)) return -1;
            if (!forth_fetch_cell(session, header->data_addr, &cell)) return -1;
            return forth_data_push(session, cell) ? 1 : -1;
        }
        if (state != 0) {
            if (!forth_colon_literal(session, (int64_t)header->data_addr)) return -1;
            return colon_call_named(session, "2!") ? 1 : -1;
        }
        if (!forth_data_pop(session, &cell)) return -1;
        if (!forth_store_cell(session, header->data_addr, cell)) return -1;
        if (!forth_data_pop(session, &cell)) return -1;
        return forth_store_cell(session, header->data_addr + FORTH_CELL_BYTES,
                                cell) ? 1 : -1;
    }
    if (fetch) {
        if (state != 0) {
            if (!forth_colon_literal(session, (int64_t)header->data_addr))
                return -1;
            return colon_call_named(session, "@") ? 1 : -1;
        }
        if (!forth_fetch_cell(session, header->data_addr, &cell)) return -1;
        return forth_data_push(session, cell) ? 1 : -1;
    }
    if (state != 0) {
        if (!forth_colon_literal(session, (int64_t)header->data_addr)) return -1;
        return colon_call_named(session, "!") ? 1 : -1;
    }
    if (!forth_data_pop(session, &cell)) return -1;
    return forth_store_cell(session, header->data_addr, cell) ? 1 : -1;
}

static int forth_host_marker(ForthSession *session) {
    uint8_t def[FORTH_NAME_MAX];
    uint32_t deflen = 0;
    ForthNt published = 0;
    int64_t save_count;
    int64_t save_bump;
    int64_t save_latest;
    int got;

    got = forth_take_word(session, def, &deflen);
    if (got <= 0) return -1;
    save_count = (int64_t)session->header_count;
    save_bump = (int64_t)session->bump;
    save_latest = (int64_t)session->latest;
    if (!forth_colon_begin(session, (const char *)def, deflen)) return -1;
    if (!forth_colon_literal(session, save_count)) {
        forth_colon_abort(session);
        return -1;
    }
    if (!forth_colon_literal(session, save_bump)) {
        forth_colon_abort(session);
        return -1;
    }
    if (!forth_colon_literal(session, save_latest)) {
        forth_colon_abort(session);
        return -1;
    }
    if (!forth_colon_host_runtime(session, FORTH_HOST_MARKER_RUN)) {
        forth_colon_abort(session);
        return -1;
    }
    return forth_colon_finish(session, &published) ? 1 : -1;
}

static int forth_host_marker_run(ForthSession *session) {
    int64_t save_latest = 0;
    int64_t save_bump = 0;
    int64_t save_count = 0;
    uint32_t i;

    if (!forth_data_pop(session, &save_latest)) return -1;
    if (!forth_data_pop(session, &save_bump)) return -1;
    if (!forth_data_pop(session, &save_count)) return -1;
    if (save_count < 0 || save_latest < 0 || save_bump < 0) return -1;
    if ((uint32_t)save_count > session->header_count) return -1;
    for (i = (uint32_t)save_count; i < session->header_count; i++)
        session->headers[i].used = false;
    session->header_count = (uint32_t)save_count;
    session->latest = (ForthNt)save_latest;
    session->bump = (uint64_t)save_bump;
    return forth_store_cell(session, session->here_cell_addr, save_bump) ? 1 : -1;
}

static int forth_host_case(ForthSession *session) {
    if (!session->colon_open) return -1;
    return forth_control_push(session, FORTH_CTRL_CASE, 0) ? 1 : -1;
}

static int forth_host_of(ForthSession *session) {
    if (!session->colon_open) return -1;
    if (!colon_call_named(session, "OVER")) return -1;
    if (!colon_call_named(session, "=")) return -1;
    if (!forth_colon_if(session)) return -1;
    return colon_call_named(session, "DROP") ? 1 : -1;
}

static int forth_host_endof(ForthSession *session) {
    return forth_colon_else(session) ? 1 : -1;
}

static int forth_host_endcase(ForthSession *session) {
    ForthCtrlKind kind = FORTH_CTRL_ORIG;
    uint32_t value = 0;

    if (!session->colon_open) return -1;
    if (!colon_call_named(session, "DROP")) return -1;
    for (;;) {
        if (!forth_control_pop(session, &kind, &value)) return -1;
        if (kind == FORTH_CTRL_ORIG) {
            if (!colon_patch_jump(session, value, session->colon_code_len))
                return -1;
            continue;
        }
        if (kind == FORTH_CTRL_CASE) return 1;
        return -1;
    }
}

static int forth_host_c_quote(ForthSession *session, int64_t state) {
    uint64_t src = 0;
    uint32_t wlen = 0;
    uint64_t dest;
    uint32_t i;
    uint8_t ch = 0;

    if (state == 0) return -1;
    if (!forth_skip_blanks(session)) return -1;
    if (!forth_parse_delimited(session, (uint8_t)'"', false, &src, &wlen))
        return -1;
    if (wlen > 255) return -1;
    dest = session->bump;
    if (!forth_dict_allot(session, (int64_t)(wlen + 1))) return -1;
    if (!forth_store_byte(session, dest, (uint8_t)wlen)) return -1;
    for (i = 0; i < wlen; i++) {
        if (!forth_fetch_byte(session, src + i, &ch)) return -1;
        if (!forth_store_byte(session, dest + 1 + i, ch)) return -1;
    }
    return forth_colon_literal(session, (int64_t)dest) ? 1 : -1;
}

static int forth_host_s_backslash(ForthSession *session, int64_t state) {
    uint8_t buf[FORTH_TIB_SIZE];
    uint32_t n = 0;
    uint64_t dest;
    uint32_t i;
    bool have = false;
    uint8_t ch = 0;

    if (!forth_skip_blanks(session)) return -1;
    for (;;) {
        if (!source_next_char(session, &ch, &have)) return -1;
        if (!have) return -1;
        if (ch == (uint8_t)'"') break;
        if (ch != (uint8_t)'\\') {
            if (n >= FORTH_TIB_SIZE) return -1;
            buf[n++] = ch;
            continue;
        }
        if (!source_next_char(session, &ch, &have) || !have) return -1;
        {
            uint8_t out[2];
            uint32_t outn = 1;
            int hi;
            int lo;
            switch (ch) {
            case 'a': out[0] = 7; break;
            case 'b': out[0] = 8; break;
            case 'e': out[0] = 27; break;
            case 'f': out[0] = 12; break;
            case 'l': out[0] = 10; break;
            case 'm':
                out[0] = 13;
                out[1] = 10;
                outn = 2;
                break;
            case 'n': out[0] = 10; break;
            case 'q': out[0] = 34; break;
            case 'r': out[0] = 13; break;
            case 't': out[0] = 9; break;
            case 'v': out[0] = 11; break;
            case 'z': out[0] = 0; break;
            case '"': out[0] = 34; break;
            case '\\': out[0] = 92; break;
            case 'x':
                if (!source_next_char(session, &ch, &have) || !have) return -1;
                hi = forth_hex_nibble(ch);
                if (hi < 0) {
                    out[0] = 0;
                    if (n >= FORTH_TIB_SIZE) return -1;
                    buf[n++] = 0;
                    if (n >= FORTH_TIB_SIZE) return -1;
                    buf[n++] = ch;
                    continue;
                }
                lo = -1;
                {
                    uint64_t caddr = 0;
                    uint64_t u = 0;
                    int64_t to_in = 0;
                    uint8_t peek = 0;
                    if (!forth_source(session, &caddr, &u)) return -1;
                    if (!forth_fetch_cell(session, session->sysvars, &to_in))
                        return -1;
                    if (to_in >= 0 && (uint64_t)to_in < u) {
                        if (!forth_fetch_byte(session, caddr + (uint64_t)to_in,
                                              &peek))
                            return -1;
                        lo = forth_hex_nibble(peek);
                        if (lo >= 0) {
                            to_in++;
                            if (!forth_store_cell(session, session->sysvars, to_in))
                                return -1;
                        }
                    }
                }
                if (lo >= 0) out[0] = (uint8_t)((hi << 4) | lo);
                else out[0] = (uint8_t)hi;
                break;
            default:
                out[0] = ch;
                break;
            }
            if (n + outn > FORTH_TIB_SIZE) return -1;
            for (i = 0; i < outn; i++) buf[n++] = out[i];
        }
    }
    if (state == 0) {
        uint64_t addr = 0;
        if (!forth_store_s_quote(session, 0, n, buf, &addr)) return -1;
        if (!forth_data_push(session, (int64_t)addr)) return -1;
        return forth_data_push(session, (int64_t)n) ? 1 : -1;
    }
    dest = session->bump;
    if (!forth_dict_allot(session, (int64_t)n)) return -1;
    for (i = 0; i < n; i++) {
        if (!forth_store_byte(session, dest + i, buf[i])) return -1;
    }
    if (!forth_colon_literal(session, (int64_t)dest)) return -1;
    return forth_colon_literal(session, (int64_t)n) ? 1 : -1;
}

static int forth_print_aligned(ForthSession *session, int64_t cell, bool is_unsigned,
                               int64_t width) {
    uint64_t mag;
    bool sign = false;
    int64_t u = 0;
    int64_t caddr = 0;
    int64_t pad;
    int64_t i;

    if (!is_unsigned && cell < 0) {
        sign = true;
        mag = (uint64_t)(-(cell + 1)) + 1u;
    } else {
        mag = (uint64_t)cell;
    }
    if (!forth_data_push(session, (int64_t)mag)) return -1;
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
    if (!forth_data_push(session, 0) || !forth_data_push(session, 0)) return -1;
    if (!forth_pict_end(session)) return -1;
    if (!forth_data_pop(session, &u) || !forth_data_pop(session, &caddr))
        return -1;
    pad = width - u - (sign ? 1 : 0);
    for (i = 0; i < pad; i++) {
        if (!forth_emit_char(session, (uint8_t)' ')) return -1;
    }
    if (sign && !forth_emit_char(session, (uint8_t)'-')) return -1;
    return forth_type_range(session, (uint64_t)caddr, (uint32_t)u) ? 1 : -1;
}

static int forth_host_holds(ForthSession *session) {
    int64_t u = 0;
    int64_t caddr = 0;
    int64_t i;
    uint8_t ch = 0;

    if (!forth_data_pop(session, &u) || u < 0) return -1;
    if (!forth_data_pop(session, &caddr)) return -1;
    for (i = u; i > 0; i--) {
        if (!forth_fetch_byte(session, (uint64_t)caddr + (uint64_t)(i - 1), &ch))
            return -1;
        if (!forth_pict_hold(session, ch)) return -1;
    }
    return 1;
}

static int forth_host_save_input(ForthSession *session) {
    const ForthSourceFrame *frame = source_top_const(session);
    int64_t to_in = 0;
    if (!frame) return -1;
    if (!forth_fetch_cell(session, session->sysvars, &to_in)) return -1;
    if (frame->kind == FORTH_SRC_FILE) {
        if (!forth_data_push(session, to_in)) return -1;
        if (!forth_data_push(session, frame->source_id)) return -1;
        if (!forth_data_push(session, frame->file_pos)) return -1;
        return forth_data_push(session, 3) ? 1 : -1;
    }
    if (frame->kind == FORTH_SRC_BLOCK) {
        if (!forth_data_push(session, to_in)) return -1;
        if (!forth_data_push(session, frame->blk)) return -1;
        return forth_data_push(session, 2) ? 1 : -1;
    }
    if (!forth_data_push(session, to_in)) return -1;
    if (!forth_data_push(session, frame->source_id)) return -1;
    if (!forth_data_push(session, (int64_t)frame->caddr)) return -1;
    if (!forth_data_push(session, (int64_t)frame->u)) return -1;
    return forth_data_push(session, 4) ? 1 : -1;
}

static int forth_host_restore_input(ForthSession *session) {
    int64_t n = 0;
    int64_t u = 0;
    int64_t caddr = 0;
    int64_t sid = 0;
    int64_t to_in = 0;
    int64_t file_pos = 0;
    int64_t drop;
    ForthSourceFrame *frame;
    int64_t i;

    if (!forth_data_pop(session, &n)) return -1;
    if (n == 2) {
        int64_t blk = 0;
        if (!forth_data_pop(session, &blk) || !forth_data_pop(session, &to_in))
            return -1;
        frame = source_top(session);
        if (!frame || frame->kind != FORTH_SRC_BLOCK || blk < 0
                || !forth_block_in_range((uint32_t)blk) || to_in < 0)
            return forth_data_push(session, (int64_t)-1) ? 1 : -1;
        if (!forth_block_assign(session, (uint32_t)blk, true)) return -1;
        frame->blk = blk;
        frame->caddr = forth_block_cache(session, (uint32_t)blk);
        frame->u = FORTH_BLOCK_SIZE;
        if ((uint64_t)to_in > frame->u)
            return forth_data_push(session, (int64_t)-1) ? 1 : -1;
        if (!forth_store_cell(session, session->sysvars, to_in)) return -1;
        if (!forth_store_cell(session, session->sysvars + FORTH_CELL_BYTES, blk))
            return -1;
        return forth_data_push(session, 0) ? 1 : -1;
    }
    if (n == 3) {
        uint32_t slot;
        FILE *fp;
        if (!forth_data_pop(session, &file_pos) || !forth_data_pop(session, &sid)
                || !forth_data_pop(session, &to_in))
            return -1;
        frame = source_top(session);
        if (!frame || frame->kind != FORTH_SRC_FILE || frame->source_id != sid)
            return forth_data_push(session, (int64_t)-1) ? 1 : -1;
        if (!decode_fileid(session, frame->fileid, &slot))
            return forth_data_push(session, (int64_t)-1) ? 1 : -1;
        fp = session->files[slot].fp;
        if (!fp || fseek(fp, (long)file_pos, SEEK_SET) != 0)
            return forth_data_push(session, (int64_t)-1) ? 1 : -1;
        if (!refill_file_line(session, frame))
            return forth_data_push(session, (int64_t)-1) ? 1 : -1;
        if (to_in < 0 || (uint64_t)to_in > frame->u)
            return forth_data_push(session, (int64_t)-1) ? 1 : -1;
        if (!forth_store_cell(session, session->sysvars, to_in)) return -1;
        return forth_data_push(session, 0) ? 1 : -1;
    }
    if (n != 4) {
        for (i = 0; i < n && i < 16; i++) {
            if (!forth_data_pop(session, &drop)) break;
        }
        return forth_data_push(session, (int64_t)-1) ? 1 : -1;
    }
    if (!forth_data_pop(session, &u)) return -1;
    if (!forth_data_pop(session, &caddr)) return -1;
    if (!forth_data_pop(session, &sid)) return -1;
    if (!forth_data_pop(session, &to_in)) return -1;
    frame = source_top(session);
    if (!frame || frame->source_id != sid || (int64_t)frame->caddr != caddr
            || (int64_t)frame->u != u || to_in < 0
            || (uint64_t)to_in > frame->u) {
        return forth_data_push(session, (int64_t)-1) ? 1 : -1;
    }
    if (!forth_store_cell(session, session->sysvars, to_in)) return -1;
    return forth_data_push(session, 0) ? 1 : -1;
}

static int forth_host_refill_word(ForthSession *session) {
    const ForthSourceFrame *frame = source_top_const(session);
    bool ok;
    if (!frame) return -1;
    if (frame->kind != FORTH_SRC_FILE && frame->kind != FORTH_SRC_BLOCK)
        return forth_data_push(session, 0) ? 1 : -1;
    ok = forth_refill(session);
    return forth_data_push(session, ok ? (int64_t)-1 : 0) ? 1 : -1;
}

static int forth_host_locals_brace(ForthSession *session) {
    uint8_t name[FORTH_NAME_MAX];
    uint32_t nlen = 0;
    int mode = 0;
    int got;

    if (!session->colon_open) return -1;
    for (;;) {
        got = forth_take_word(session, name, &nlen);
        if (got <= 0) return -1;
        if (nlen == 2 && name[0] == ':' && name[1] == '}')
            return forth_locals_close(session) ? 1 : -1;
        if (mode == 2) continue;
        if (nlen == 1 && name[0] == '|' && mode == 0) {
            mode = 1;
            continue;
        }
        if (nlen == 2 && name[0] == '-' && name[1] == '-' && mode < 2) {
            mode = 2;
            continue;
        }
        if (!forth_add_colon_local(session, name, nlen, mode == 0, false))
            return -1;
    }
}

static int forth_host_paren_local(ForthSession *session) {
    int64_t u = 0;
    int64_t caddr = 0;
    uint8_t name[FORTH_NAME_MAX];
    uint32_t i;

    if (!session->colon_open) return -1;
    if (!forth_data_pop(session, &u) || !forth_data_pop(session, &caddr))
        return -1;
    if (u == 0) return forth_locals_close(session) ? 1 : -1;
    if (u < 0 || (uint64_t)u > FORTH_NAME_MAX) return -1;
    for (i = 0; i < (uint32_t)u; i++) {
        if (!forth_fetch_byte(session, (uint64_t)caddr + i, &name[i]))
            return -1;
    }
    return forth_add_colon_local(session, name, (uint32_t)u, true, true) ? 1 : -1;
}

static int forth_host_dot_s(ForthSession *session) {
    uint32_t depth = forth_data_depth(session);
    uint32_t i;

    for (i = 0; i < depth; i++) {
        int64_t cell = 0;
        uint64_t addr = session->data_stack_addr
            + (uint64_t)i * FORTH_CELL_BYTES;
        char buf[32];
        int n;
        int k;
        if (!forth_fetch_cell(session, addr, &cell)) return -1;
        n = snprintf(buf, sizeof(buf), "%lld ", (long long)cell);
        if (n < 0) return -1;
        for (k = 0; k < n; k++) {
            if (!forth_emit_char(session, (uint8_t)buf[k])) return -1;
        }
    }
    return forth_emit_char(session, (uint8_t)'\n') ? 1 : -1;
}

static bool word_named(const uint8_t *name, uint32_t nlen, const char *lit) {
    return names_equal(name, nlen, (const uint8_t *)lit, (uint32_t)strlen(lit));
}

static int forth_host_bracket_else(ForthSession *session) {
    int64_t level = 1;
    uint8_t name[FORTH_NAME_MAX];
    uint32_t nlen = 0;

    for (;;) {
        int got = forth_take_word(session, name, &nlen);
        if (got < 0) return -1;
        if (got == 0) {
            if (!forth_refill(session)) return -1;
            continue;
        }
        if (word_named(name, nlen, "[IF]")) {
            level++;
        } else if (word_named(name, nlen, "[ELSE]")) {
            level--;
            if (level != 0) level++;
        } else if (word_named(name, nlen, "[THEN]")) {
            level--;
        }
        if (level == 0) return 1;
    }
}

static int forth_host_bracket_if(ForthSession *session) {
    int64_t flag = 0;
    if (!forth_data_pop(session, &flag)) return -1;
    if (flag == 0) return forth_host_bracket_else(session);
    return 1;
}

static int forth_host_cs_pick(ForthSession *session) {
    int64_t u = 0;
    uint32_t avail;
    uint32_t idx;
    ForthCtrlItem item;

    if (!session->colon_open) return -1;
    if (!forth_data_pop(session, &u) || u < 0) return -1;
    if (session->control_depth <= session->colon_saved_control_depth) return -1;
    avail = session->control_depth - session->colon_saved_control_depth;
    if ((uint64_t)u >= (uint64_t)avail) return -1;
    idx = session->control_depth - 1u - (uint32_t)u;
    if (session->control_depth >= FORTH_CONTROL_STACK_CELLS) return -1;
    item = session->control[idx];
    session->control[session->control_depth++] = item;
    return 1;
}

static int forth_host_cs_roll(ForthSession *session) {
    int64_t u = 0;
    uint32_t avail;
    uint32_t idx;
    uint32_t i;
    ForthCtrlItem item;

    if (!session->colon_open) return -1;
    if (!forth_data_pop(session, &u) || u < 0) return -1;
    if (session->control_depth <= session->colon_saved_control_depth) return -1;
    avail = session->control_depth - session->colon_saved_control_depth;
    if ((uint64_t)u >= (uint64_t)avail) return -1;
    if (u == 0) return 1;
    idx = session->control_depth - 1u - (uint32_t)u;
    item = session->control[idx];
    for (i = idx; i + 1 < session->control_depth; i++)
        session->control[i] = session->control[i + 1];
    session->control[session->control_depth - 1] = item;
    return 1;
}

static int forth_host_defined(ForthSession *session, bool want_defined) {
    uint8_t name[FORTH_NAME_MAX];
    uint32_t nlen = 0;
    ForthNt nt = 0;
    ForthXt xt = 0;
    bool immediate = false;
    int got;
    int64_t flag;

    got = forth_take_word(session, name, &nlen);
    if (got < 0) return -1;
    flag = 0;
    if (got > 0 && forth_find(session, (const char *)name, nlen, &nt, &xt, &immediate))
        flag = -1;
    if (!want_defined) flag = flag ? 0 : -1;
    return forth_data_push(session, flag) ? 1 : -1;
}

static int forth_host_n_to_r(ForthSession *session) {
    int64_t n = 0;
    int64_t cells[FORTH_STACK_CELLS];
    int64_t i;

    if (!forth_data_pop(session, &n) || n < 0) return -1;
    if (n > (int64_t)FORTH_STACK_CELLS) return -1;
    for (i = 0; i < n; i++) {
        if (!forth_data_pop(session, &cells[i])) return -1;
    }
    for (i = n - 1; i >= 0; i--) {
        if (!forth_return_push(session, cells[i])) return -1;
    }
    return forth_return_push(session, n) ? 1 : -1;
}

static int forth_host_nr_from(ForthSession *session) {
    int64_t n = 0;
    int64_t cells[FORTH_STACK_CELLS];
    int64_t i;

    if (!forth_return_pop(session, &n) || n < 0) return -1;
    if (n > (int64_t)FORTH_STACK_CELLS) return -1;
    for (i = 0; i < n; i++) {
        if (!forth_return_pop(session, &cells[i])) return -1;
    }
    for (i = n - 1; i >= 0; i--) {
        if (!forth_data_push(session, cells[i])) return -1;
    }
    return forth_data_push(session, n) ? 1 : -1;
}

static int forth_host_synonym(ForthSession *session) {
    uint8_t neu[FORTH_NAME_MAX];
    uint8_t alt[FORTH_NAME_MAX];
    uint32_t nnew = 0;
    uint32_t nold = 0;
    ForthNt old_nt = 0;
    ForthNt new_nt = 0;
    ForthXt xt = 0;
    bool imm = false;
    ForthHeader *oldh;
    ForthHeader *newh;

    if (forth_take_word(session, neu, &nnew) <= 0) return -1;
    if (forth_take_word(session, alt, &nold) <= 0) return -1;
    if (!forth_find(session, (const char *)alt, nold, &old_nt, &xt, &imm))
        return -1;
    oldh = header_at(session, old_nt);
    if (!oldh) return -1;
    if (!forth_define(session, (const char *)neu, nnew, oldh->xt, oldh->immediate,
                      false, &new_nt))
        return -1;
    newh = header_at(session, new_nt);
    if (!newh) return -1;
    newh->compile_only = oldh->compile_only;
    newh->host_kind = oldh->host_kind;
    newh->data_addr = oldh->data_addr;
    newh->body_cells = oldh->body_cells;
    return 1;
}

static int forth_host_traverse_wordlist(ForthSession *session) {
    int64_t wid = 0;
    int64_t xtc = 0;
    int64_t flag = 0;
    uint32_t start;
    uint32_t i;
    ForthXt xt;

    if (!forth_data_pop(session, &wid) || !forth_data_pop(session, &xtc))
        return -1;
    if (wid <= 0 || xtc < 0) return -1;
    if (!valid_wid(session, (ForthWid)wid)) return -1;
    xt = (ForthXt)xtc;
    start = session->header_count;
    for (i = start; i > 0; i--) {
        const ForthHeader *header = &session->headers[i - 1];
        if (!header->used || header->hidden || header->wid != (ForthWid)wid)
            continue;
        if (!forth_data_push(session, (int64_t)i)) return -1;
        if (forth_invoke_nested(session, xt) != VM_OK) return -1;
        if (!forth_data_pop(session, &flag)) return -1;
        if (flag == 0) return 1;
    }
    return 1;
}

static int forth_host_name_to_compile(ForthSession *session) {
    int64_t ntc = 0;
    ForthHeader *header;
    ForthNt helper_nt = 0;
    ForthXt helper_xt = 0;
    bool helper_imm = false;
    const char *helper;

    if (!forth_data_pop(session, &ntc) || ntc <= 0) return -1;
    header = header_at(session, (ForthNt)ntc);
    if (!header) return -1;
    helper = header->immediate ? "EXECUTE" : "COMPILE,";
    if (!forth_find(session, helper, (uint32_t)strlen(helper),
                    &helper_nt, &helper_xt, &helper_imm))
        return -1;
    if (!forth_data_push(session, (int64_t)header->xt)) return -1;
    return forth_data_push(session, (int64_t)helper_xt) ? 1 : -1;
}

static int forth_host_name_to_interpret(ForthSession *session) {
    int64_t ntc = 0;
    ForthHeader *header;

    if (!forth_data_pop(session, &ntc) || ntc <= 0) return -1;
    header = header_at(session, (ForthNt)ntc);
    if (!header) return -1;
    if (header->compile_only)
        return forth_data_push(session, 0) ? 1 : -1;
    return forth_data_push(session, (int64_t)header->xt) ? 1 : -1;
}

static int forth_host_name_to_string(ForthSession *session) {
    int64_t ntc = 0;
    ForthHeader *header;

    if (!forth_data_pop(session, &ntc) || ntc <= 0) return -1;
    header = header_at(session, (ForthNt)ntc);
    if (!header) return -1;
    if (!forth_data_push(session, (int64_t)header->name_addr)) return -1;
    return forth_data_push(session, (int64_t)header->name_len) ? 1 : -1;
}

static bool forth_mem_write(ForthSession *session, uint64_t addr,
                            const void *src, uint32_t n) {
    const uint8_t *bytes = (const uint8_t *)src;
    uint32_t i;
    for (i = 0; i < n; i++) {
        if (!forth_store_byte(session, addr + i, bytes[i])) return false;
    }
    return true;
}

static bool forth_mem_read(ForthSession *session, uint64_t addr, void *dest,
                           uint32_t n) {
    uint8_t *bytes = (uint8_t *)dest;
    uint32_t i;
    for (i = 0; i < n; i++) {
        if (!forth_fetch_byte(session, addr + i, &bytes[i])) return false;
    }
    return true;
}

static bool forth_compile_float(ForthSession *session, double r) {
    union {
        double d;
        int64_t i;
    } bits;
    bits.d = r;
    if (!forth_colon_literal(session, bits.i)) return false;
    return forth_colon_host_runtime(session, FORTH_HOST_F_LIT_BITS);
}

static int forth_fp_flag(ForthSession *session, bool value) {
    return forth_data_push(session, value ? (int64_t)-1 : 0) ? 1 : -1;
}

static int forth_fp_unop(ForthSession *session, double (*fn)(double)) {
    double x = 0.0;
    if (!forth_float_pop(session, &x)) return -1;
    return forth_float_push(session, fn(x)) ? 1 : -1;
}

static int forth_fp_binop(ForthSession *session, double (*fn)(double, double)) {
    double y = 0.0;
    double x = 0.0;
    if (!forth_float_pop(session, &y) || !forth_float_pop(session, &x))
        return -1;
    return forth_float_push(session, fn(x, y)) ? 1 : -1;
}

static double forth_fp_sub(double x, double y) { return x - y; }
static double forth_fp_div(double x, double y) { return x / y; }
static double forth_fp_add(double x, double y) { return x + y; }
static double forth_fp_mul(double x, double y) { return x * y; }
static double forth_fp_neg(double x) { return -x; }
static double forth_fp_alog(double x) { return pow(10.0, x); }
static double forth_fp_floor0(double x) {
    if (x == 0.0) return copysign(0.0, x);
    return floor(x);
}
static double forth_fp_max(double x, double y) { return fmax(x, y); }
static double forth_fp_min(double x, double y) { return fmin(x, y); }
static double forth_fp_abs(double x) { return fabs(x); }
static double forth_fp_round(double x) { return round(x); }
static double forth_fp_sqrt(double x) { return sqrt(x); }
static double forth_fp_sin(double x) { return sin(x); }
static double forth_fp_cos(double x) { return cos(x); }
static double forth_fp_tan(double x) { return tan(x); }
static double forth_fp_asin(double x) { return asin(x); }
static double forth_fp_acos(double x) { return acos(x); }
static double forth_fp_atan(double x) { return atan(x); }
static double forth_fp_exp(double x) { return exp(x); }
static double forth_fp_expm1(double x) { return expm1(x); }
static double forth_fp_ln(double x) { return log(x); }
static double forth_fp_log10(double x) { return log10(x); }
static double forth_fp_lnp1(double x) { return log1p(x); }
static double forth_fp_pow(double x, double y) { return pow(x, y); }
static double forth_fp_sinh(double x) { return sinh(x); }
static double forth_fp_cosh(double x) { return cosh(x); }
static double forth_fp_tanh(double x) { return tanh(x); }
static double forth_fp_asinh(double x) { return asinh(x); }
static double forth_fp_acosh(double x) { return acosh(x); }
static double forth_fp_atanh(double x) { return atanh(x); }

static int forth_fp_emit_text(ForthSession *session, const char *text) {
    size_t i;
    size_t n;
    if (!text) return -1;
    n = strlen(text);
    for (i = 0; i < n; i++) {
        if (!forth_emit_char(session, (uint8_t)text[i])) return -1;
    }
    return forth_emit_char(session, (uint8_t)' ') ? 1 : -1;
}

static int forth_fp_print(ForthSession *session, uint16_t host) {
    double r = 0.0;
    char buf[160];
    int prec;
    int expn;
    double mantissa;

    if (!forth_float_pop(session, &r)) return -1;
    prec = (int)session->fprecision;
    if (prec < 1) prec = 1;
    if (prec > 17) prec = 17;
    if (host == FORTH_HOST_FS_DOT) {
        snprintf(buf, sizeof(buf), "%.*E", prec - 1, r);
        return forth_fp_emit_text(session, buf);
    }
    if (host == FORTH_HOST_FE_DOT) {
        if (r == 0.0) {
            snprintf(buf, sizeof(buf), "%.*fE0", prec - 1, r);
            return forth_fp_emit_text(session, buf);
        }
        expn = (int)floor(log10(fabs(r)));
        expn -= ((expn % 3) + 3) % 3;
        mantissa = r / pow(10.0, (double)expn);
        snprintf(buf, sizeof(buf), "%.*fE%d", prec - 1, mantissa, expn);
        return forth_fp_emit_text(session, buf);
    }
    snprintf(buf, sizeof(buf), "%.*f", prec, r);
    return forth_fp_emit_text(session, buf);
}

static int forth_fp_represent(ForthSession *session) {
    int64_t u = 0;
    int64_t caddr = 0;
    double r = 0.0;
    int64_t n = 0;
    int64_t sign_flag = 0;
    int64_t valid = -1;
    uint32_t i;
    unsigned long long ival = 0;
    unsigned long long limit = 1;

    if (!forth_data_pop(session, &u) || u < 0) return -1;
    if (!forth_data_pop(session, &caddr)) return -1;
    if (!forth_float_pop(session, &r)) return -1;
    if (signbit(r)) sign_flag = -1;
    r = fabs(r);
    if (!isfinite(r)) {
        valid = 0;
        r = 0.0;
    }
    if (u > 64) u = 64;
    if (r == 0.0) {
        n = 0;
    } else {
        n = (int64_t)floor(log10(r)) + 1;
        ival = (unsigned long long)floor(r * pow(10.0, (double)((int)u - n))
                                         + 0.5);
        for (i = 0; i < (uint32_t)u; i++) limit *= 10ULL;
        if (limit > 0 && ival >= limit) {
            ival /= 10ULL;
            n++;
        }
    }
    for (i = 0; i < (uint32_t)u; i++) {
        unsigned long long div = 1;
        uint32_t j;
        uint8_t dig;
        for (j = i + 1; j < (uint32_t)u; j++) div *= 10ULL;
        dig = (uint8_t)((ival / div) % 10ULL);
        if (!forth_store_byte(session, (uint64_t)caddr + i,
                              (uint8_t)('0' + dig)))
            return -1;
    }
    if (!forth_data_push(session, n)) return -1;
    if (!forth_data_push(session, sign_flag)) return -1;
    return forth_data_push(session, valid) ? 1 : -1;
}

static int forth_host_fp(ForthSession *session, uint16_t host) {
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    int64_t cell = 0;
    int64_t addr = 0;
    int64_t lo = 0;
    int64_t hi = 0;
    union {
        double d;
        float f;
        int64_t i;
    } bits;

    switch (host) {
    case FORTH_HOST_D_TO_F:
        if (!forth_dpop(session, &lo, &hi)) return -1;
        return forth_float_push(session, (double)forth_pack_d(lo, hi)) ? 1 : -1;
    case FORTH_HOST_F_TO_D:
        if (!forth_float_pop(session, &a)) return -1;
        forth_unpack_d((__int128)a, &lo, &hi);
        return forth_dpush(session, lo, hi) ? 1 : -1;
    case FORTH_HOST_FDEPTH:
        return forth_data_push(session, (int64_t)forth_float_depth(session))
            ? 1 : -1;
    case FORTH_HOST_FDROP:
        return forth_float_pop(session, &a) ? 1 : -1;
    case FORTH_HOST_FDUP:
        if (!forth_float_pop(session, &a)) return -1;
        return (forth_float_push(session, a) && forth_float_push(session, a))
            ? 1 : -1;
    case FORTH_HOST_FSWAP:
        if (!forth_float_pop(session, &b) || !forth_float_pop(session, &a))
            return -1;
        return (forth_float_push(session, b) && forth_float_push(session, a))
            ? 1 : -1;
    case FORTH_HOST_FOVER:
        if (!forth_float_pop(session, &b) || !forth_float_pop(session, &a))
            return -1;
        return (forth_float_push(session, a) && forth_float_push(session, b)
                && forth_float_push(session, a)) ? 1 : -1;
    case FORTH_HOST_FROT:
        if (!forth_float_pop(session, &c) || !forth_float_pop(session, &b)
                || !forth_float_pop(session, &a))
            return -1;
        return (forth_float_push(session, b) && forth_float_push(session, c)
                && forth_float_push(session, a)) ? 1 : -1;
    case FORTH_HOST_FPLUS:
        return forth_fp_binop(session, forth_fp_add);
    case FORTH_HOST_FMINUS:
        return forth_fp_binop(session, forth_fp_sub);
    case FORTH_HOST_FSTAR:
        return forth_fp_binop(session, forth_fp_mul);
    case FORTH_HOST_FSLASH:
        return forth_fp_binop(session, forth_fp_div);
    case FORTH_HOST_FNEGATE:
        return forth_fp_unop(session, forth_fp_neg);
    case FORTH_HOST_FZERO_LESS:
        if (!forth_float_pop(session, &a)) return -1;
        return forth_fp_flag(session, a < 0.0);
    case FORTH_HOST_FZERO_EQUAL:
        if (!forth_float_pop(session, &a)) return -1;
        return forth_fp_flag(session, a == 0.0);
    case FORTH_HOST_FLESS:
        if (!forth_float_pop(session, &b) || !forth_float_pop(session, &a))
            return -1;
        return forth_fp_flag(session, a < b);
    case FORTH_HOST_FABS:
        return forth_fp_unop(session, forth_fp_abs);
    case FORTH_HOST_FMAX:
        return forth_fp_binop(session, forth_fp_max);
    case FORTH_HOST_FMIN:
        return forth_fp_binop(session, forth_fp_min);
    case FORTH_HOST_FTILDE:
        if (!forth_float_pop(session, &c) || !forth_float_pop(session, &b)
                || !forth_float_pop(session, &a))
            return -1;
        if (c > 0.0)
            return forth_fp_flag(session, fabs(a - b) < c);
        if (c == 0.0)
            return forth_fp_flag(session, a == b);
        return forth_fp_flag(session, fabs(a - b) < fabs(c) * (fabs(a) + fabs(b)));
    case FORTH_HOST_FSTORE:
    case FORTH_HOST_DFSTORE:
        if (!forth_data_pop(session, &addr) || !forth_float_pop(session, &a))
            return -1;
        bits.d = a;
        return forth_mem_write(session, (uint64_t)addr, &bits.d, 8) ? 1 : -1;
    case FORTH_HOST_FFETCH:
    case FORTH_HOST_DFFETCH:
        if (!forth_data_pop(session, &addr)) return -1;
        bits.d = 0.0;
        if (!forth_mem_read(session, (uint64_t)addr, &bits.d, 8)) return -1;
        return forth_float_push(session, bits.d) ? 1 : -1;
    case FORTH_HOST_SFSTORE:
        if (!forth_data_pop(session, &addr) || !forth_float_pop(session, &a))
            return -1;
        bits.f = (float)a;
        return forth_mem_write(session, (uint64_t)addr, &bits.f, 4) ? 1 : -1;
    case FORTH_HOST_SFFETCH:
        if (!forth_data_pop(session, &addr)) return -1;
        bits.f = 0.0f;
        if (!forth_mem_read(session, (uint64_t)addr, &bits.f, 4)) return -1;
        return forth_float_push(session, (double)bits.f) ? 1 : -1;
    case FORTH_HOST_FLITERAL:
        if (!session->colon_open) return -1;
        if (!forth_float_pop(session, &a)) return -1;
        return forth_compile_float(session, a) ? 1 : -1;
    case FORTH_HOST_F_LIT_BITS:
        if (!forth_data_pop(session, &cell)) return -1;
        bits.i = cell;
        return forth_float_push(session, bits.d) ? 1 : -1;
    case FORTH_HOST_FLOATS:
    case FORTH_HOST_DFLOATS:
        if (!forth_data_pop(session, &cell)) return -1;
        return forth_data_push(session, cell * 8) ? 1 : -1;
    case FORTH_HOST_SFLOATS:
        if (!forth_data_pop(session, &cell)) return -1;
        return forth_data_push(session, cell * 4) ? 1 : -1;
    case FORTH_HOST_TO_FLOAT:
        {
            int64_t u = 0;
            uint8_t buf[FORTH_WORD_MAX];
            uint32_t n;
            uint32_t i;
            if (!forth_data_pop(session, &u) || u < 0) return -1;
            if (!forth_data_pop(session, &addr)) return -1;
            n = u > (int64_t)(FORTH_WORD_MAX - 1) ? (FORTH_WORD_MAX - 1)
                                                  : (uint32_t)u;
            for (i = 0; i < n; i++) {
                if (!forth_fetch_byte(session, (uint64_t)addr + i, &buf[i]))
                    return -1;
            }
            if (forth_parse_to_float(buf, n, true, &a)) {
                if (!forth_float_push(session, a)) return -1;
                return forth_data_push(session, -1) ? 1 : -1;
            }
            return forth_data_push(session, 0) ? 1 : -1;
        }
    case FORTH_HOST_FLOOR:
        return forth_fp_unop(session, forth_fp_floor0);
    case FORTH_HOST_FROUND:
        return forth_fp_unop(session, forth_fp_round);
    case FORTH_HOST_FSQRT:
        return forth_fp_unop(session, forth_fp_sqrt);
    case FORTH_HOST_FSIN:
        return forth_fp_unop(session, forth_fp_sin);
    case FORTH_HOST_FCOS:
        return forth_fp_unop(session, forth_fp_cos);
    case FORTH_HOST_FTAN:
        return forth_fp_unop(session, forth_fp_tan);
    case FORTH_HOST_FASIN:
        return forth_fp_unop(session, forth_fp_asin);
    case FORTH_HOST_FACOS:
        return forth_fp_unop(session, forth_fp_acos);
    case FORTH_HOST_FATAN:
        return forth_fp_unop(session, forth_fp_atan);
    case FORTH_HOST_FATAN2:
        if (!forth_float_pop(session, &b) || !forth_float_pop(session, &a))
            return -1;
        return forth_float_push(session, atan2(a, b)) ? 1 : -1;
    case FORTH_HOST_FSINCOS:
        if (!forth_float_pop(session, &a)) return -1;
        return (forth_float_push(session, sin(a))
                && forth_float_push(session, cos(a))) ? 1 : -1;
    case FORTH_HOST_FEXP:
        return forth_fp_unop(session, forth_fp_exp);
    case FORTH_HOST_FEXPM1:
        return forth_fp_unop(session, forth_fp_expm1);
    case FORTH_HOST_FLN:
        return forth_fp_unop(session, forth_fp_ln);
    case FORTH_HOST_FLOG:
        return forth_fp_unop(session, forth_fp_log10);
    case FORTH_HOST_FLNP1:
        return forth_fp_unop(session, forth_fp_lnp1);
    case FORTH_HOST_FSTAR_STAR:
        return forth_fp_binop(session, forth_fp_pow);
    case FORTH_HOST_FALOG:
        return forth_fp_unop(session, forth_fp_alog);
    case FORTH_HOST_FSINH:
        return forth_fp_unop(session, forth_fp_sinh);
    case FORTH_HOST_FCOSH:
        return forth_fp_unop(session, forth_fp_cosh);
    case FORTH_HOST_FTANH:
        return forth_fp_unop(session, forth_fp_tanh);
    case FORTH_HOST_FASINH:
        return forth_fp_unop(session, forth_fp_asinh);
    case FORTH_HOST_FACOSH:
        return forth_fp_unop(session, forth_fp_acosh);
    case FORTH_HOST_FATANH:
        return forth_fp_unop(session, forth_fp_atanh);
    case FORTH_HOST_REPRESENT:
        return forth_fp_represent(session);
    case FORTH_HOST_PRECISION:
        return forth_data_push(session, (int64_t)session->fprecision) ? 1 : -1;
    case FORTH_HOST_SET_PRECISION:
        if (!forth_data_pop(session, &cell) || cell < 1) return -1;
        session->fprecision = (uint32_t)cell;
        return 1;
    case FORTH_HOST_FS_DOT:
    case FORTH_HOST_FE_DOT:
    case FORTH_HOST_F_DOT:
        return forth_fp_print(session, host);
    default:
        return -1;
    }
}

static uint32_t forth_utf8_len(uint8_t lead) {
    if (lead < 0x80u) return 1;
    if ((lead & 0xE0u) == 0xC0u) return 2;
    if ((lead & 0xF0u) == 0xE0u) return 3;
    if ((lead & 0xF8u) == 0xF0u) return 4;
    return 1;
}

static bool forth_utf8_decode_bytes(const uint8_t *p, uint32_t len, uint32_t *cp,
                                    uint32_t *n) {
    uint32_t need;
    uint32_t acc;
    uint32_t i;

    if (!p || !cp || !n || len == 0) return false;
    need = forth_utf8_len(p[0]);
    if (need > len) {
        *cp = p[0];
        *n = 1;
        return true;
    }
    if (need == 1) {
        *cp = p[0];
        *n = 1;
        return true;
    }
    acc = p[0] & ((need == 2) ? 0x1Fu : (need == 3) ? 0x0Fu : 0x07u);
    for (i = 1; i < need; i++) {
        if ((p[i] & 0xC0u) != 0x80u) {
            *cp = p[0];
            *n = 1;
            return true;
        }
        acc = (acc << 6) | (uint32_t)(p[i] & 0x3Fu);
    }
    *cp = acc;
    *n = need;
    return true;
}

static bool forth_utf8_encode(uint32_t cp, uint8_t out[4], uint32_t *n) {
    if (!out || !n) return false;
    if (cp > 0x10FFFFu || (cp >= 0xD800u && cp <= 0xDFFFu)) return false;
    if (cp < 0x80u) {
        out[0] = (uint8_t)cp;
        *n = 1;
    } else if (cp < 0x800u) {
        out[0] = (uint8_t)(0xC0u | (cp >> 6));
        out[1] = (uint8_t)(0x80u | (cp & 0x3Fu));
        *n = 2;
    } else if (cp < 0x10000u) {
        out[0] = (uint8_t)(0xE0u | (cp >> 12));
        out[1] = (uint8_t)(0x80u | ((cp >> 6) & 0x3Fu));
        out[2] = (uint8_t)(0x80u | (cp & 0x3Fu));
        *n = 3;
    } else {
        out[0] = (uint8_t)(0xF0u | (cp >> 18));
        out[1] = (uint8_t)(0x80u | ((cp >> 12) & 0x3Fu));
        out[2] = (uint8_t)(0x80u | ((cp >> 6) & 0x3Fu));
        out[3] = (uint8_t)(0x80u | (cp & 0x3Fu));
        *n = 4;
    }
    return true;
}

static bool forth_utf8_decode_addr(ForthSession *session, uint64_t addr,
                                   uint32_t avail, uint32_t *cp, uint32_t *n) {
    uint8_t buf[4];
    uint32_t i;
    uint32_t take;

    if (!session || !cp || !n || avail == 0) return false;
    if (!forth_fetch_byte(session, addr, &buf[0])) return false;
    take = forth_utf8_len(buf[0]);
    if (take > avail) take = 1;
    for (i = 1; i < take; i++) {
        if (!forth_fetch_byte(session, addr + i, &buf[i])) return false;
    }
    return forth_utf8_decode_bytes(buf, take, cp, n);
}

static int64_t forth_xc_width(uint32_t cp) {
    if (cp == 0) return 0;
    if (cp < 0x20u || (cp >= 0x7Fu && cp < 0xA0u)) return 0;
    if (cp >= 0x300u && cp <= 0x36Fu) return 0;
    if (cp >= 0x1100u && cp <= 0x115Fu) return 2;
    if (cp >= 0x2E80u && cp <= 0xA4CFu) return 2;
    if (cp >= 0xAC00u && cp <= 0xD7A3u) return 2;
    if (cp >= 0xF900u && cp <= 0xFAFFu) return 2;
    if (cp >= 0xFE10u && cp <= 0xFE6Fu) return 2;
    if (cp >= 0xFF00u && cp <= 0xFF60u) return 2;
    if (cp >= 0xFFE0u && cp <= 0xFFE6u) return 2;
    if (cp >= 0x20000u) return 2;
    return 1;
}

static bool forth_mem_copy(ForthSession *session, uint64_t dest, uint64_t src,
                           uint32_t n) {
    uint32_t i;
    uint8_t b = 0;
    for (i = 0; i < n; i++) {
        if (!forth_fetch_byte(session, src + i, &b)) return false;
        if (!forth_store_byte(session, dest + i, b)) return false;
    }
    return true;
}

static uint64_t forth_block_disk(const ForthSession *session, uint32_t blk) {
    return session->blocks_addr + (uint64_t)blk * FORTH_BLOCK_SIZE;
}

static uint64_t forth_block_cache(const ForthSession *session, uint32_t blk) {
    return session->block_cache_addr + (uint64_t)blk * FORTH_BLOCK_SIZE;
}

static bool forth_block_in_range(uint32_t blk) {
    return blk < FORTH_BLOCK_COUNT;
}

static bool forth_block_assign(ForthSession *session, uint32_t blk, bool load) {
    if (!session || !forth_block_in_range(blk)) return false;
    if (!session->block_assigned[blk]) {
        if (load) {
            if (!forth_mem_copy(session, forth_block_cache(session, blk),
                                forth_block_disk(session, blk), FORTH_BLOCK_SIZE))
                return false;
        }
        session->block_assigned[blk] = true;
    }
    session->block_current = (int32_t)blk;
    return true;
}

static bool forth_block_save_all(ForthSession *session) {
    uint32_t i;
    for (i = 0; i < FORTH_BLOCK_COUNT; i++) {
        if (!session->block_dirty[i]) continue;
        if (!forth_mem_copy(session, forth_block_disk(session, i),
                            forth_block_cache(session, i), FORTH_BLOCK_SIZE))
            return false;
        session->block_dirty[i] = false;
    }
    return true;
}

static void forth_block_empty(ForthSession *session) {
    uint32_t i;
    for (i = 0; i < FORTH_BLOCK_COUNT; i++) {
        session->block_assigned[i] = false;
        session->block_dirty[i] = false;
    }
    session->block_current = -1;
}

static int forth_host_xchar(ForthSession *session, uint16_t host) {
    int64_t a = 0;
    int64_t b = 0;
    int64_t u = 0;
    uint32_t cp = 0;
    uint32_t n = 0;
    uint8_t enc[4];
    uint32_t i;

    switch (host) {
    case FORTH_HOST_XC_FETCH_PLUS:
        if (!forth_data_pop(session, &a)) return -1;
        if (!forth_utf8_decode_addr(session, (uint64_t)a, 4, &cp, &n)) return -1;
        if (!forth_data_push(session, a + (int64_t)n)) return -1;
        return forth_data_push(session, (int64_t)cp) ? 1 : -1;
    case FORTH_HOST_XCHAR_PLUS:
        if (!forth_data_pop(session, &a)) return -1;
        if (!forth_utf8_decode_addr(session, (uint64_t)a, 4, &cp, &n)) return -1;
        return forth_data_push(session, a + (int64_t)n) ? 1 : -1;
    case FORTH_HOST_XCHAR_MINUS:
        if (!forth_data_pop(session, &a) || a <= 0) return -1;
        do {
            uint8_t ch = 0;
            a--;
            if (!forth_fetch_byte(session, (uint64_t)a, &ch)) return -1;
            if ((ch & 0xC0u) != 0x80u) break;
        } while (a > 0);
        return forth_data_push(session, a) ? 1 : -1;
    case FORTH_HOST_XC_SIZE:
        if (!forth_data_pop(session, &a) || a < 0) return -1;
        if (!forth_utf8_encode((uint32_t)a, enc, &n)) return -1;
        return forth_data_push(session, (int64_t)n) ? 1 : -1;
    case FORTH_HOST_X_SIZE:
        if (!forth_data_pop(session, &u) || u < 0) return -1;
        if (!forth_data_pop(session, &a)) return -1;
        if (u == 0) return forth_data_push(session, 0) ? 1 : -1;
        if (!forth_utf8_decode_addr(session, (uint64_t)a, (uint32_t)u, &cp, &n))
            return -1;
        return forth_data_push(session, (int64_t)n) ? 1 : -1;
    case FORTH_HOST_XC_STORE_PLUS:
        if (!forth_data_pop(session, &a) || !forth_data_pop(session, &b))
            return -1;
        if (!forth_utf8_encode((uint32_t)b, enc, &n)) return -1;
        if (!forth_mem_write(session, (uint64_t)a, enc, n)) return -1;
        return forth_data_push(session, a + (int64_t)n) ? 1 : -1;
    case FORTH_HOST_XC_STORE_PLUS_Q:
        if (!forth_data_pop(session, &u) || u < 0) return -1;
        if (!forth_data_pop(session, &a) || !forth_data_pop(session, &b))
            return -1;
        if (!forth_utf8_encode((uint32_t)b, enc, &n)) return -1;
        if ((int64_t)n > u) {
            if (!forth_data_push(session, a) || !forth_data_push(session, u))
                return -1;
            return forth_data_push(session, 0) ? 1 : -1;
        }
        if (!forth_mem_write(session, (uint64_t)a, enc, n)) return -1;
        if (!forth_data_push(session, a + (int64_t)n)) return -1;
        if (!forth_data_push(session, u - (int64_t)n)) return -1;
        return forth_data_push(session, -1) ? 1 : -1;
    case FORTH_HOST_XC_COMMA:
        if (!forth_data_pop(session, &b) || b < 0) return -1;
        if (!forth_utf8_encode((uint32_t)b, enc, &n)) return -1;
        a = (int64_t)session->bump;
        if (!forth_dict_allot(session, (int64_t)n)) return -1;
        return forth_mem_write(session, (uint64_t)a, enc, n) ? 1 : -1;
    case FORTH_HOST_XEMIT:
        if (!forth_data_pop(session, &b) || b < 0) return -1;
        if (!forth_utf8_encode((uint32_t)b, enc, &n)) return -1;
        for (i = 0; i < n; i++) {
            if (!forth_emit_char(session, enc[i])) return -1;
        }
        return 1;
    case FORTH_HOST_XHOLD:
        if (!forth_data_pop(session, &b) || b < 0) return -1;
        if (!forth_utf8_encode((uint32_t)b, enc, &n)) return -1;
        for (i = n; i > 0; i--) {
            if (!forth_pict_hold(session, enc[i - 1])) return -1;
        }
        return 1;
    case FORTH_HOST_PLUS_XSTRING:
        if (!forth_data_pop(session, &u) || u < 0) return -1;
        if (!forth_data_pop(session, &a)) return -1;
        if (u == 0) {
            if (!forth_data_push(session, a)) return -1;
            return forth_data_push(session, 0) ? 1 : -1;
        }
        if (!forth_utf8_decode_addr(session, (uint64_t)a, (uint32_t)u, &cp, &n))
            return -1;
        if (!forth_data_push(session, a + (int64_t)n)) return -1;
        return forth_data_push(session, u - (int64_t)n) ? 1 : -1;
    case FORTH_HOST_X_STRING_MINUS:
        if (!forth_data_pop(session, &u) || u < 0) return -1;
        if (!forth_data_pop(session, &a)) return -1;
        if (u == 0) {
            if (!forth_data_push(session, a)) return -1;
            return forth_data_push(session, 0) ? 1 : -1;
        }
        b = a + u;
        do {
            uint8_t ch = 0;
            b--;
            if (!forth_fetch_byte(session, (uint64_t)b, &ch)) return -1;
            if ((ch & 0xC0u) != 0x80u) break;
        } while (b > a);
        if (!forth_data_push(session, a)) return -1;
        return forth_data_push(session, b - a) ? 1 : -1;
    case FORTH_HOST_TRAILING_GARBAGE:
        if (!forth_data_pop(session, &u) || u < 0) return -1;
        if (!forth_data_pop(session, &a)) return -1;
        {
            int64_t used = 0;
            while (used < u) {
                if (!forth_utf8_decode_addr(session, (uint64_t)(a + used),
                                            (uint32_t)(u - used), &cp, &n))
                    return -1;
                if (used + (int64_t)n > u) break;
                if (n == 1) {
                    uint8_t lead = 0;
                    if (!forth_fetch_byte(session, (uint64_t)(a + used), &lead))
                        return -1;
                    if (forth_utf8_len(lead) > 1) break;
                }
                used += (int64_t)n;
            }
            if (!forth_data_push(session, a)) return -1;
            return forth_data_push(session, used) ? 1 : -1;
        }
    case FORTH_HOST_XC_WIDTH:
        if (!forth_data_pop(session, &b) || b < 0) return -1;
        return forth_data_push(session, forth_xc_width((uint32_t)b)) ? 1 : -1;
    case FORTH_HOST_X_WIDTH:
        if (!forth_data_pop(session, &u) || u < 0) return -1;
        if (!forth_data_pop(session, &a)) return -1;
        {
            int64_t width = 0;
            int64_t used = 0;
            while (used < u) {
                if (!forth_utf8_decode_addr(session, (uint64_t)(a + used),
                                            (uint32_t)(u - used), &cp, &n))
                    return -1;
                width += forth_xc_width(cp);
                used += (int64_t)n;
            }
            return forth_data_push(session, width) ? 1 : -1;
        }
    case FORTH_HOST_EKEY_TO_XCHAR:
        if (!forth_data_pop(session, &b)) return -1;
        if (b < 0 || b > 0x10FFFF || (b >= 0xD800 && b <= 0xDFFF))
            return forth_data_push(session, 0) ? 1 : -1;
        if (!forth_data_push(session, b)) return -1;
        return forth_data_push(session, -1) ? 1 : -1;
    case FORTH_HOST_XKEY: {
        uint8_t buf[4];
        uint64_t caddr = 0;
        uint64_t su = 0;
        int64_t to_in = 0;
        uint32_t have = 0;
        if (forth_source_id(session) != 0) {
            int got;
            if (!isatty(STDIN_FILENO)) return -1;
            got = getchar();
            if (got == EOF) return -1;
            buf[0] = (uint8_t)got;
            n = forth_utf8_len(buf[0]);
            for (i = 1; i < n; i++) {
                got = getchar();
                if (got == EOF) break;
                buf[i] = (uint8_t)got;
            }
            if (!forth_utf8_decode_bytes(buf, i < n ? i : n, &cp, &n)) return -1;
            return forth_data_push(session, (int64_t)cp) ? 1 : -1;
        }
        if (!forth_source(session, &caddr, &su)) return -1;
        if (!forth_fetch_cell(session, session->sysvars, &to_in)) return -1;
        if (to_in < 0 || (uint64_t)to_in >= su) return -1;
        have = (uint32_t)(su - (uint64_t)to_in);
        if (have > 4) have = 4;
        for (i = 0; i < have; i++) {
            if (!forth_fetch_byte(session, caddr + (uint64_t)to_in + i, &buf[i]))
                return -1;
        }
        if (!forth_utf8_decode_bytes(buf, have, &cp, &n)) return -1;
        to_in += (int64_t)n;
        if (!forth_store_cell(session, session->sysvars, to_in)) return -1;
        return forth_data_push(session, (int64_t)cp) ? 1 : -1;
    }
    case FORTH_HOST_XKEY_Q: {
        uint64_t caddr = 0;
        uint64_t su = 0;
        int64_t to_in = 0;
        if (forth_source_id(session) != 0)
            return forth_data_push(session, 0) ? 1 : -1;
        if (!forth_source(session, &caddr, &su)) return -1;
        if (!forth_fetch_cell(session, session->sysvars, &to_in)) return -1;
        return forth_data_push(session,
                               (to_in >= 0 && (uint64_t)to_in < su) ? (int64_t)-1
                                                                    : 0)
            ? 1 : -1;
    }
    default:
        return -1;
    }
}

static int forth_host_block(ForthSession *session, uint16_t host) {
    int64_t u = 0;
    int64_t v = 0;
    uint32_t blk;
    uint32_t i;
    ForthNt nt = 0;
    ForthXt xt = 0;
    bool imm = false;

    switch (host) {
    case FORTH_HOST_BLOCK:
        if (!forth_data_pop(session, &u) || u < 0 || !forth_block_in_range((uint32_t)u))
            return -1;
        blk = (uint32_t)u;
        if (!forth_block_assign(session, blk, true)) return -1;
        return forth_data_push(session, (int64_t)forth_block_cache(session, blk))
            ? 1 : -1;
    case FORTH_HOST_BUFFER:
        if (!forth_data_pop(session, &u) || u < 0 || !forth_block_in_range((uint32_t)u))
            return -1;
        blk = (uint32_t)u;
        if (!forth_block_assign(session, blk, false)) return -1;
        return forth_data_push(session, (int64_t)forth_block_cache(session, blk))
            ? 1 : -1;
    case FORTH_HOST_UPDATE:
        if (session->block_current < 0) return 1;
        session->block_dirty[session->block_current] = true;
        return 1;
    case FORTH_HOST_SAVE_BUFFERS:
        return forth_block_save_all(session) ? 1 : -1;
    case FORTH_HOST_EMPTY_BUFFERS:
        forth_block_empty(session);
        return 1;
    case FORTH_HOST_FLUSH:
        if (!forth_block_save_all(session)) return -1;
        forth_block_empty(session);
        return 1;
    case FORTH_HOST_LOAD:
        if (!forth_data_pop(session, &u) || u < 0 || !forth_block_in_range((uint32_t)u))
            return -1;
        blk = (uint32_t)u;
        if (!forth_block_assign(session, blk, true)) return -1;
        if (!forth_source_push_block(session, blk)) return -1;
        if (!forth_interpret_loop(session)) {
            forth_source_pop(session);
            return -1;
        }
        if (!forth_source_pop(session)) return -1;
        return 1;
    case FORTH_HOST_LIST: {
        ForthHeader *scr;
        if (!forth_data_pop(session, &u) || u < 0 || !forth_block_in_range((uint32_t)u))
            return -1;
        blk = (uint32_t)u;
        if (!forth_block_assign(session, blk, true)) return -1;
        if (!forth_find(session, "SCR", 3, &nt, &xt, &imm)) return -1;
        (void)xt;
        (void)imm;
        scr = header_at(session, nt);
        if (!scr || !forth_store_cell(session, scr->data_addr, u)) return -1;
        for (i = 0; i < 16; i++) {
            uint64_t line = forth_block_cache(session, blk)
                + (uint64_t)i * 64u;
            if (!forth_type_range(session, line, 64)) return -1;
            if (!forth_emit_char(session, (uint8_t)'\n')) return -1;
        }
        return 1;
    }
    case FORTH_HOST_THRU:
        if (!forth_data_pop(session, &v) || !forth_data_pop(session, &u))
            return -1;
        if (u < 0 || v < 0) return -1;
        while (u <= v) {
            if (!forth_data_push(session, u)) return -1;
            if (forth_host_block(session, FORTH_HOST_LOAD) < 0) return -1;
            u++;
        }
        return 1;
    default:
        return -1;
    }
}

static int forth_run_host(ForthSession *session, uint16_t host, int64_t state) {
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
    case FORTH_HOST_NONAME:
        if (forth_colon_is_open(session)) return -1;
        if (!forth_colon_begin_noname(session)) return -1;
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
    case FORTH_HOST_AHEAD:
        return forth_colon_ahead(session) ? 1 : -1;
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
        if (!forth_colon_is_open(session)) return -1;
        if (!forth_data_pop(session, &cell)) return -1;
        return forth_colon_literal(session, cell) ? 1 : -1;
    case FORTH_HOST_TWO_LITERAL:
        if (!forth_colon_is_open(session)) return -1;
        {
            int64_t lo = 0;
            int64_t hi = 0;
            if (!forth_dpop(session, &lo, &hi)) return -1;
            if (!forth_colon_literal(session, lo)) return -1;
            return forth_colon_literal(session, hi) ? 1 : -1;
        }
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
        {
            uint32_t cp = 0;
            uint32_t n = 0;
            if (!forth_utf8_decode_bytes(name, nlen, &cp, &n)) return -1;
            return forth_data_push(session, (int64_t)cp) ? 1 : -1;
        }
    case FORTH_HOST_BRACKET_CHAR:
        if (state == 0) return -1;
        got = forth_take_word(session, name, &nlen);
        if (got <= 0 || nlen == 0) return -1;
        {
            uint32_t cp = 0;
            uint32_t n = 0;
            if (!forth_utf8_decode_bytes(name, nlen, &cp, &n)) return -1;
            return forth_colon_literal(session, (int64_t)cp) ? 1 : -1;
        }
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
    case FORTH_HOST_BACKSLASH: {
        ForthSourceFrame *frame = source_top(session);
        if (frame && frame->kind == FORTH_SRC_BLOCK) {
            int64_t to_in = 0;
            if (!forth_fetch_cell(session, session->sysvars, &to_in)) return -1;
            if (to_in < 0) return -1;
            to_in = ((to_in / 64) + 1) * 64;
            if ((uint64_t)to_in > frame->u) to_in = (int64_t)frame->u;
            return forth_store_cell(session, session->sysvars, to_in) ? 1 : -1;
        }
        return forth_skip_until(session, 0, true) ? 1 : -1;
    }
    case FORTH_HOST_PAREN: {
        for (;;) {
            uint64_t caddr = 0;
            uint64_t u = 0;
            int64_t to_in = 0;
            ForthSourceFrame *frame;
            if (!forth_source(session, &caddr, &u)) return -1;
            if (!forth_fetch_cell(session, session->sysvars, &to_in)) return -1;
            while ((uint64_t)to_in < u) {
                uint8_t ch = 0;
                if (!forth_fetch_byte(session, caddr + (uint64_t)to_in, &ch))
                    return -1;
                to_in++;
                if (ch == (uint8_t)')') {
                    return forth_store_cell(session, session->sysvars, to_in)
                        ? 1 : -1;
                }
            }
            if (!forth_store_cell(session, session->sysvars, to_in)) return -1;
            frame = source_top(session);
            if (!frame || frame->kind != FORTH_SRC_FILE) return 1;
            if (!forth_refill(session)) return 1;
        }
    }
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
            if (!forth_locals_close(session)) return -1;
            if (!session->colon_does_pending) {
                session->colon_parent_local_count = session->colon_local_count;
                forth_reset_colon_locals(session);
                session->colon_does_pending = true;
                session->colon_does_off = session->colon_code_len;
                return 1;
            }
            if (session->colon_does_chain_off != 0) return -1;
            session->colon_does_a_local_count = session->colon_local_count;
            forth_reset_colon_locals(session);
            session->colon_does_chain_off = session->colon_code_len;
            return 1;
        }
        return forth_host_does(session) ? 1 : -1;
    case FORTH_HOST_DOES_ENTER:
        if (!forth_data_pop(session, &cell) || cell < 0) return -1;
        session->does_child_nt = (cell == 0) ? 0 : (ForthNt)cell;
        return 1;
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
            if (!forth_store_s_quote(session, src, wlen, NULL, &dest)) return -1;
            if (!forth_data_push(session, (int64_t)dest)) return -1;
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
    case FORTH_HOST_DOT_PAREN:
        {
            uint64_t src = 0;
            uint32_t wlen = 0;
            if (!forth_skip_blanks(session)) return -1;
            if (!forth_parse_delimited(session, (uint8_t)')', false, &src, &wlen))
                return -1;
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
        return forth_throw_now(session, -1) ? 1 : -1;
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
            if (!forth_data_pop(session, &dest)) return -1;
            if (!forth_data_pop(session, &src)) return -1;
            if (u == 0) return 1;
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
    case FORTH_HOST_PLUSLOOP:
        return forth_plusloop_step(session) ? 1 : -1;
    case FORTH_HOST_SOURCE_ID:
        if (state != 0) return 0;
        return forth_data_push(session, forth_source_id(session)) ? 1 : -1;
    case FORTH_HOST_REFILL:
        if (state != 0) return 0;
        return forth_host_refill_word(session);
    case FORTH_HOST_PARSE_NAME:
        if (state != 0) return 0;
        return forth_host_parse_name(session);
    case FORTH_HOST_VALUE:
        if (state != 0) return 0;
        return forth_host_value(session);
    case FORTH_HOST_TWO_VALUE:
        if (state != 0) return 0;
        return forth_host_two_value(session);
    case FORTH_HOST_TO:
        return forth_host_body_op(session, state, false);
    case FORTH_HOST_IS:
        return forth_host_body_op(session, state, false);
    case FORTH_HOST_ACTION_OF:
        return forth_host_body_op(session, state, true);
    case FORTH_HOST_MARKER:
        if (state != 0) return 0;
        return forth_host_marker(session);
    case FORTH_HOST_MARKER_RUN:
        if (state != 0) return 0;
        return forth_host_marker_run(session);
    case FORTH_HOST_CASE:
        if (state == 0) return -1;
        return forth_host_case(session);
    case FORTH_HOST_OF:
        if (state == 0) return -1;
        return forth_host_of(session);
    case FORTH_HOST_ENDOF:
        if (state == 0) return -1;
        return forth_host_endof(session);
    case FORTH_HOST_ENDCASE:
        if (state == 0) return -1;
        return forth_host_endcase(session);
    case FORTH_HOST_C_QUOTE:
        return forth_host_c_quote(session, state);
    case FORTH_HOST_S_BACKSLASH:
        return forth_host_s_backslash(session, state);
    case FORTH_HOST_DOT_R:
        if (state != 0) return 0;
        {
            int64_t width = 0;
            if (!forth_data_pop(session, &width)) return -1;
            if (!forth_data_pop(session, &cell)) return -1;
            return forth_print_aligned(session, cell, false, width);
        }
    case FORTH_HOST_UDOT_R:
        if (state != 0) return 0;
        {
            int64_t width = 0;
            if (!forth_data_pop(session, &width)) return -1;
            if (!forth_data_pop(session, &cell)) return -1;
            return forth_print_aligned(session, cell, true, width);
        }
    case FORTH_HOST_HOLDS:
        if (state != 0) return 0;
        return forth_host_holds(session);
    case FORTH_HOST_UNUSED:
        if (state != 0) return 0;
        return forth_data_push(session,
                               (int64_t)(FORTH_ADDR_MAX - session->bump)) ? 1 : -1;
    case FORTH_HOST_SAVE_INPUT:
        if (state != 0) return 0;
        return forth_host_save_input(session);
    case FORTH_HOST_RESTORE_INPUT:
        if (state != 0) return 0;
        return forth_host_restore_input(session);
    case FORTH_HOST_DPLUS:
    case FORTH_HOST_DMINUS:
    case FORTH_HOST_DNEGATE:
    case FORTH_HOST_DTWO_STAR:
    case FORTH_HOST_DTWO_SLASH:
    case FORTH_HOST_DLESS:
    case FORTH_HOST_DEQUAL:
    case FORTH_HOST_DABS:
    case FORTH_HOST_DMAX:
    case FORTH_HOST_DMIN:
    case FORTH_HOST_MPLUS:
    case FORTH_HOST_DULESS:
    case FORTH_HOST_M_STAR_SLASH:
        if (state != 0) return 0;
        return forth_host_dmath(session, host);
    case FORTH_HOST_TRAILING:
    case FORTH_HOST_CMOVE:
    case FORTH_HOST_CMOVE_UP:
    case FORTH_HOST_COMPARE:
    case FORTH_HOST_SEARCH:
    case FORTH_HOST_UNESCAPE:
    case FORTH_HOST_REPLACES:
    case FORTH_HOST_SUBSTITUTE:
        if (state != 0) return 0;
        return forth_host_string(session, host);
    case FORTH_HOST_SLITERAL:
        return forth_host_sliteral(session);
    case FORTH_HOST_WORDLIST:
    case FORTH_HOST_GET_ORDER:
    case FORTH_HOST_SET_ORDER:
    case FORTH_HOST_GET_CURRENT:
    case FORTH_HOST_SET_CURRENT:
    case FORTH_HOST_FORTH_WORDLIST:
    case FORTH_HOST_ALSO:
    case FORTH_HOST_PREVIOUS:
    case FORTH_HOST_ONLY:
    case FORTH_HOST_FORTH:
    case FORTH_HOST_DEFINITIONS:
    case FORTH_HOST_SEARCH_WORDLIST:
    case FORTH_HOST_ORDER:
        if (state != 0) return 0;
        return forth_host_search_order(session, host);
    case FORTH_HOST_BIN:
    case FORTH_HOST_OPEN_FILE:
    case FORTH_HOST_CREATE_FILE:
    case FORTH_HOST_CLOSE_FILE:
    case FORTH_HOST_DELETE_FILE:
    case FORTH_HOST_READ_FILE:
    case FORTH_HOST_READ_LINE:
    case FORTH_HOST_WRITE_FILE:
    case FORTH_HOST_WRITE_LINE:
    case FORTH_HOST_FILE_POSITION:
    case FORTH_HOST_FILE_SIZE:
    case FORTH_HOST_REPOSITION_FILE:
    case FORTH_HOST_RESIZE_FILE:
    case FORTH_HOST_FLUSH_FILE:
    case FORTH_HOST_RENAME_FILE:
    case FORTH_HOST_FILE_STATUS:
    case FORTH_HOST_INCLUDED:
    case FORTH_HOST_INCLUDE:
    case FORTH_HOST_INCLUDE_FILE:
    case FORTH_HOST_REQUIRED:
    case FORTH_HOST_REQUIRE:
        if (state != 0) return 0;
        return forth_host_file(session, host);
    case FORTH_HOST_ALLOCATE:
    case FORTH_HOST_MEM_FREE:
    case FORTH_HOST_MEM_RESIZE:
        if (state != 0) return 0;
        return forth_host_mem(session, host);
    case FORTH_HOST_LOCALS_BRACE:
        return forth_host_locals_brace(session);
    case FORTH_HOST_LOCAL:
        return forth_host_paren_local(session);
    case FORTH_HOST_DOT_S:
        if (state != 0) return 0;
        return forth_host_dot_s(session);
    case FORTH_HOST_BRACKET_IF:
        return forth_host_bracket_if(session);
    case FORTH_HOST_BRACKET_ELSE:
        return forth_host_bracket_else(session);
    case FORTH_HOST_BRACKET_THEN:
        return 1;
    case FORTH_HOST_CS_PICK:
        return forth_host_cs_pick(session);
    case FORTH_HOST_CS_ROLL:
        return forth_host_cs_roll(session);
    case FORTH_HOST_DEFINED:
        return forth_host_defined(session, true);
    case FORTH_HOST_UNDEFINED:
        return forth_host_defined(session, false);
    case FORTH_HOST_N_TO_R:
        if (state != 0) return 0;
        return forth_host_n_to_r(session);
    case FORTH_HOST_NR_FROM:
        if (state != 0) return 0;
        return forth_host_nr_from(session);
    case FORTH_HOST_SYNONYM:
        if (state != 0) return 0;
        return forth_host_synonym(session);
    case FORTH_HOST_TRAVERSE_WORDLIST:
        if (state != 0) return 0;
        return forth_host_traverse_wordlist(session);
    case FORTH_HOST_NAME_TO_COMPILE:
        if (state != 0) return 0;
        return forth_host_name_to_compile(session);
    case FORTH_HOST_NAME_TO_INTERPRET:
        if (state != 0) return 0;
        return forth_host_name_to_interpret(session);
    case FORTH_HOST_NAME_TO_STRING:
        if (state != 0) return 0;
        return forth_host_name_to_string(session);
    case FORTH_HOST_D_TO_F:
    case FORTH_HOST_F_TO_D:
    case FORTH_HOST_FDEPTH:
    case FORTH_HOST_FDROP:
    case FORTH_HOST_FDUP:
    case FORTH_HOST_FSWAP:
    case FORTH_HOST_FOVER:
    case FORTH_HOST_FROT:
    case FORTH_HOST_FPLUS:
    case FORTH_HOST_FMINUS:
    case FORTH_HOST_FSTAR:
    case FORTH_HOST_FSLASH:
    case FORTH_HOST_FNEGATE:
    case FORTH_HOST_FZERO_LESS:
    case FORTH_HOST_FZERO_EQUAL:
    case FORTH_HOST_FLESS:
    case FORTH_HOST_FABS:
    case FORTH_HOST_FMAX:
    case FORTH_HOST_FMIN:
    case FORTH_HOST_FTILDE:
    case FORTH_HOST_FFETCH:
    case FORTH_HOST_FSTORE:
    case FORTH_HOST_SFFETCH:
    case FORTH_HOST_SFSTORE:
    case FORTH_HOST_DFFETCH:
    case FORTH_HOST_DFSTORE:
    case FORTH_HOST_FLITERAL:
    case FORTH_HOST_F_LIT_BITS:
    case FORTH_HOST_FLOATS:
    case FORTH_HOST_SFLOATS:
    case FORTH_HOST_DFLOATS:
    case FORTH_HOST_TO_FLOAT:
    case FORTH_HOST_FLOOR:
    case FORTH_HOST_FROUND:
    case FORTH_HOST_FSQRT:
    case FORTH_HOST_FSIN:
    case FORTH_HOST_FCOS:
    case FORTH_HOST_FTAN:
    case FORTH_HOST_FASIN:
    case FORTH_HOST_FACOS:
    case FORTH_HOST_FATAN:
    case FORTH_HOST_FATAN2:
    case FORTH_HOST_FSINCOS:
    case FORTH_HOST_FEXP:
    case FORTH_HOST_FEXPM1:
    case FORTH_HOST_FLN:
    case FORTH_HOST_FLOG:
    case FORTH_HOST_FLNP1:
    case FORTH_HOST_FSTAR_STAR:
    case FORTH_HOST_FALOG:
    case FORTH_HOST_FSINH:
    case FORTH_HOST_FCOSH:
    case FORTH_HOST_FTANH:
    case FORTH_HOST_FASINH:
    case FORTH_HOST_FACOSH:
    case FORTH_HOST_FATANH:
    case FORTH_HOST_REPRESENT:
    case FORTH_HOST_PRECISION:
    case FORTH_HOST_SET_PRECISION:
    case FORTH_HOST_FS_DOT:
    case FORTH_HOST_FE_DOT:
    case FORTH_HOST_F_DOT:
        return forth_host_fp(session, host);
    case FORTH_HOST_XCHAR_PLUS:
    case FORTH_HOST_XCHAR_MINUS:
    case FORTH_HOST_XC_FETCH_PLUS:
    case FORTH_HOST_XC_STORE_PLUS:
    case FORTH_HOST_XC_STORE_PLUS_Q:
    case FORTH_HOST_XC_SIZE:
    case FORTH_HOST_X_SIZE:
    case FORTH_HOST_XC_COMMA:
    case FORTH_HOST_XEMIT:
    case FORTH_HOST_XKEY:
    case FORTH_HOST_XKEY_Q:
    case FORTH_HOST_PLUS_XSTRING:
    case FORTH_HOST_X_STRING_MINUS:
    case FORTH_HOST_TRAILING_GARBAGE:
    case FORTH_HOST_X_WIDTH:
    case FORTH_HOST_XC_WIDTH:
    case FORTH_HOST_XHOLD:
    case FORTH_HOST_EKEY_TO_XCHAR:
        return forth_host_xchar(session, host);
    case FORTH_HOST_BLOCK:
    case FORTH_HOST_BUFFER:
    case FORTH_HOST_UPDATE:
    case FORTH_HOST_FLUSH:
    case FORTH_HOST_SAVE_BUFFERS:
    case FORTH_HOST_EMPTY_BUFFERS:
    case FORTH_HOST_LOAD:
    case FORTH_HOST_LIST:
    case FORTH_HOST_THRU:
        return forth_host_block(session, host);
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

static bool forth_throw_now(ForthSession *session, int64_t code) {
    if (!session) return false;
    if (code == 0) return true;
    if (!forth_store_cell(session, session->throw_code_addr, code)) return false;
    if (session->vm.frame_count != 0)
        vm_request_halt(&session->vm);
    return false;
}

bool forth_interpret_loop(ForthSession *session) {
    uint8_t name[FORTH_NAME_MAX];
    uint32_t nlen = 0;

    for (;;) {
        ForthNt nt = 0;
        ForthXt xt = 0;
        bool immediate = false;
        ForthParsedNumber number;
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
        if (state != 0) {
            int slot = forth_local_slot(session, name, nlen);
            if (slot >= 0) {
                if (!forth_colon_local_fetch(session, slot)) return false;
                continue;
            }
        }
        if (forth_find(session, (const char *)name, nlen, &nt, &xt, &immediate)) {
            header = header_at(session, nt);
            if (state != 0 && !immediate) {
                if (!forth_colon_call(session, xt)) return false;
                continue;
            }
            host_rc = forth_run_host(session,
                                     header ? header->host_kind : FORTH_HOST_NONE,
                                     state);
            if (host_rc < 0) return false;
            if (session->exit_requested || session->quit_requested) return true;
            if (host_rc == 0 && header && header->host_kind != FORTH_HOST_NONE
                    && state == 0)
                return false;
            if (host_rc > 0) continue;
            if (header && header->compile_only && state == 0) return false;
            if (session->vm.frame_count != 0) {
                ran = forth_invoke_nested(session, xt);
            } else {
                ran = forth_session_invoke(session, xt, NULL, 0, NULL);
            }
            if (ran != VM_OK) return false;
            if (session->exit_requested || session->quit_requested) return true;
            if (forth_throw_pending(session)) return false;
            continue;
        }
        if (!forth_parse_number(session, name, nlen, &number)) {
            double fvalue = 0.0;
            if (!forth_parse_to_float(name, nlen, false, &fvalue))
                return forth_throw_now(session, -13);
            if (state != 0) {
                if (!forth_compile_float(session, fvalue)) return false;
            } else if (!forth_float_push(session, fvalue)) {
                return false;
            }
            continue;
        }
        if (state != 0) {
            if (!forth_colon_literal(session, number.lo)) return false;
            if (number.is_double && !forth_colon_literal(session, number.hi))
                return false;
        } else if (!forth_data_push(session, number.lo)) {
            return false;
        } else if (number.is_double && !forth_data_push(session, number.hi)) {
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
    if (ok && forth_colon_is_open(session)) ok = false;
    if (!ok && forth_colon_is_open(session)) forth_colon_abort(session);
    return ok;
}

bool forth_interpret_file(ForthSession *session, const char *path) {
    uint32_t fileid = 0;
    ForthSession *prev;
    bool ok = true;

    if (!session || !path || path[0] == '\0') return false;
    if (!forth_file_open(session, path, "r", &fileid)) return false;
    if (!forth_source_push_file(session, fileid)) {
        forth_file_close(session, fileid);
        return false;
    }
    if (!forth_store_cell(session, session->throw_code_addr, 0)) {
        forth_source_pop(session);
        forth_file_close(session, fileid);
        return false;
    }
    session->quit_requested = false;
    prev = g_forth;
    g_forth = session;
    while (forth_refill(session)) {
        if (!forth_interpret_loop(session)) {
            ok = false;
            break;
        }
        if (session->exit_requested) break;
        if (session->quit_requested) {
            session->quit_requested = false;
            break;
        }
    }
    g_forth = prev;
    if (ok && forth_colon_is_open(session)) ok = false;
    forth_source_pop(session);
    forth_file_close(session, fileid);
    return ok;
}

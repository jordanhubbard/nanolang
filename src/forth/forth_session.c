/*
 * Forth session runtime — one module, one VM, Forth stacks, virtual memory.
 */

#include "forth_session.h"
#include "nanoisa/verifier.h"

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
    bool used;
    uint8_t name[FORTH_NAME_MAX];
} ForthHeader;

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
    uint32_t dpush_fn;
    uint32_t dpop_fn;
    uint32_t throw_fn;
    uint64_t throw_code_addr;
    int64_t ret[FORTH_RETURN_STACK_CELLS];
    uint32_t ret_depth;
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
};

static bool forth_allocate_ex(ForthSession *session, uint64_t bytes, uint64_t *addr,
                              bool pinned);
static bool forth_install_dpush(ForthSession *session);

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

    if (!forth_allocate_ex(session, FORTH_CELL_BYTES * 3, &session->sysvars, true))
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

    if (!forth_allocate_ex(session, FORTH_CELL_BYTES, &session->data_depth_addr, true))
        return false;
    if (!forth_allocate_ex(session,
                           (uint64_t)FORTH_STACK_CELLS * (uint64_t)FORTH_CELL_BYTES,
                           &session->data_stack_addr, true))
        return false;
    if (!forth_store_cell(session, session->data_depth_addr, 0)) return false;
    if (!forth_allocate_ex(session, FORTH_CELL_BYTES, &session->throw_code_addr, true))
        return false;
    if (!forth_store_cell(session, session->throw_code_addr, 0)) return false;

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
    return session;
}

void forth_session_destroy(ForthSession *session) {
    uint32_t i;
    if (!session) return;
    for (i = 0; i < FORTH_FILE_SLOTS; i++) {
        if (session->files[i].fp) fclose(session->files[i].fp);
    }
    vm_destroy(&session->vm);
    nvm_module_free(session->module);
    free(session->regions);
    free(session->headers);
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
    if (!session) return VM_ERR_UNDEFINED_FUNCTION;
    return vm_invoke(&session->vm, fn_idx, args, arg_count, out_result);
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
    if (!session || session->ret_depth >= FORTH_RETURN_STACK_CELLS) return false;
    session->ret[session->ret_depth++] = cell;
    return true;
}

bool forth_return_pop(ForthSession *session, int64_t *out) {
    if (!session || session->ret_depth == 0 || !out) return false;
    *out = session->ret[--session->ret_depth];
    return true;
}

uint32_t forth_return_depth(const ForthSession *session) {
    return session ? session->ret_depth : 0;
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
    uint8_t code[256];
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
    saved_ret = session->ret_depth;
    saved_fp = session->fp_depth;
    saved_control = session->control_depth;
    saved_source = session->source_depth;
    if (!forth_fetch_cell(session, session->sysvars, &saved_to_in)) return false;
    if (!forth_fetch_cell(session, session->sysvars + FORTH_CELL_BYTES, &saved_blk))
        return false;
    if (!forth_store_cell(session, session->throw_code_addr, 0)) return false;

    ran = forth_session_invoke(session, xt, NULL, 0, NULL);
    if (ran != VM_OK) return false;
    if (!forth_fetch_cell(session, session->throw_code_addr, &thrown)) return false;
    if (thrown == 0) {
        *code = 0;
        return true;
    }

    while (session->source_depth > saved_source) {
        if (!forth_source_pop(session)) return false;
    }
    session->ret_depth = saved_ret;
    session->fp_depth = saved_fp;
    session->control_depth = saved_control;
    if (!forth_store_cell(session, session->data_depth_addr, (int64_t)saved_data))
        return false;
    if (!forth_store_cell(session, session->sysvars, saved_to_in)) return false;
    if (!forth_store_cell(session, session->sysvars + FORTH_CELL_BYTES, saved_blk))
        return false;
    if (!forth_store_cell(session, session->throw_code_addr, 0)) return false;
    *code = thrown;
    return true;
}

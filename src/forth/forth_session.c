/*
 * Forth session runtime — one module, one VM, Forth stacks, virtual memory.
 */

#include "forth_session.h"

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

struct ForthSession {
    NvmModule *module;
    VmState vm;
    int64_t data[FORTH_STACK_CELLS];
    uint32_t data_depth;
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
};

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
    if (!session || session->data_depth >= FORTH_STACK_CELLS) return false;
    session->data[session->data_depth++] = cell;
    return true;
}

bool forth_data_pop(ForthSession *session, int64_t *out) {
    if (!session || session->data_depth == 0 || !out) return false;
    *out = session->data[--session->data_depth];
    return true;
}

uint32_t forth_data_depth(const ForthSession *session) {
    return session ? session->data_depth : 0;
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

bool forth_allocate(ForthSession *session, uint64_t bytes, uint64_t *addr) {
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
            session->region_count++;
            region->size = size;
        }
        region->used = true;
        *addr = region->addr;
        return true;
    }

    if (session->bump > UINT64_MAX - size) return false;
    if (!ensure_memory(session, session->bump + size)) return false;
    if (!regions_reserve(session, 1)) return false;
    session->regions[session->region_count].addr = session->bump;
    session->regions[session->region_count].size = size;
    session->regions[session->region_count].used = true;
    session->region_count++;
    *addr = session->bump;
    session->bump += size;
    return true;
}

bool forth_free(ForthSession *session, uint64_t addr) {
    int idx;
    if (!session || addr == 0) return false;
    idx = find_region_at(session, addr);
    if (idx < 0) return false;
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

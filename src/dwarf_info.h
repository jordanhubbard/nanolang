/*
 * dwarf_info.h — DWARF v4 debug info builder for nanolang
 *
 * Emits GNU assembler debug directives (.section .debug_info, .debug_abbrev,
 * .debug_line, .debug_str) compatible with DWARF v4.
 *
 * For the LLVM IR path, use LLVM metadata directly (see llvm_backend.c).
 * For the RISC-V path, use dwarf_emit_asm_sections() plus .file/.loc directives.
 *
 * Usage:
 *   DwarfBuilder *db = dwarf_begin("example.nano");
 *   dwarf_function(db, "foo", 0, 3, 0, TYPE_INT);
 *   dwarf_variable(db, "x", TYPE_INT, 4, 4);
 *   dwarf_line(db, 0, 3, 0, 0);
 *   dwarf_emit_asm_sections(db, out);
 *   dwarf_free(db);
 */

#ifndef NANOLANG_DWARF_INFO_H
#define NANOLANG_DWARF_INFO_H

#include <stdio.h>
#include <stdbool.h>

#define DWARF_MAX_FILES  32
#define DWARF_MAX_FUNCS  256
#define DWARF_MAX_VARS   512
#define DWARF_MAX_LOCS   4096

typedef struct {
    char name[128];
    int  file_idx;
    int  line;
    int  col;
    int  ret_type;  /* nanolang Type enum value */
} DwarfFunc;

typedef struct {
    char name[64];
    int  type;
    int  line;
    int  col;
} DwarfVar;

typedef struct {
    int file_idx;
    int line;
    int col;
    int addr_offset;
} DwarfLoc;

typedef struct {
    char      source_file[256];
    char      files[DWARF_MAX_FILES][256];
    int       file_count;
    DwarfFunc funcs[DWARF_MAX_FUNCS];
    int       func_count;
    DwarfVar  vars[DWARF_MAX_VARS];
    int       var_count;
    DwarfLoc  locs[DWARF_MAX_LOCS];
    int       loc_count;
} DwarfBuilder;

/* Create a new DWARF builder for the given source file. */
DwarfBuilder *dwarf_begin(const char *filename);

/* Record a function definition. file_idx = 0 for the primary source file. */
void dwarf_function(DwarfBuilder *db, const char *name,
                    int file_idx, int line, int col, int ret_type);

/* Record a local variable within the current function scope. */
void dwarf_variable(DwarfBuilder *db, const char *name,
                    int type, int line, int col);

/* Record a source-location → address-offset mapping for .debug_line. */
void dwarf_line(DwarfBuilder *db, int file_idx, int line, int col,
                int addr_offset);

/*
 * Emit .debug_abbrev, .debug_info, .debug_line, and .debug_str sections
 * as GNU assembler directives to `out`.
 * These can be assembled with `as` or `riscv64-unknown-elf-as`.
 */
void dwarf_emit_asm_sections(DwarfBuilder *db, FILE *out);

/* Release all resources held by `db`. */
void dwarf_free(DwarfBuilder *db);

#endif /* NANOLANG_DWARF_INFO_H */

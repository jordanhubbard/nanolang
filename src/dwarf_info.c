/*
 * dwarf_info.c — DWARF v4 debug info builder for nanolang
 *
 * Emits minimal but valid DWARF v4 sections as GNU assembler directives.
 * The emitted sections cover:
 *   .debug_abbrev  — abbreviation table (compile unit + subprogram entries)
 *   .debug_info    — compilation unit DIE with nested subprogram DIEs
 *   .debug_line    — stub (real line info comes from .loc directives in .text)
 *   .debug_str     — producer/filename strings
 *
 * DWARF v4 spec refs:
 *   §7.5.3  Abbreviations Tables
 *   §6.2    Line Number Information
 *   §3.1    Compilation Unit Entries
 *   §3.3.1  General Subroutine and Entry Point Information
 */

#include "dwarf_info.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── DW_TAG / DW_AT / DW_FORM constants ────────────────────────────────── */

#define DW_TAG_compile_unit   0x11
#define DW_TAG_subprogram     0x2e
#define DW_TAG_base_type      0x24

#define DW_AT_producer        0x25
#define DW_AT_language        0x13
#define DW_AT_name            0x03
#define DW_AT_comp_dir        0x1b
#define DW_AT_stmt_list       0x10
#define DW_AT_low_pc          0x11
#define DW_AT_high_pc         0x12
#define DW_AT_decl_file       0x3a
#define DW_AT_decl_line       0x3b
#define DW_AT_decl_column     0x39
#define DW_AT_external        0x3f
#define DW_AT_frame_base      0x40
#define DW_AT_byte_size       0x0b
#define DW_AT_encoding        0x3e

#define DW_FORM_addr          0x01
#define DW_FORM_string        0x08
#define DW_FORM_data1         0x0b
#define DW_FORM_data4         0x06
#define DW_FORM_flag_present  0x19
#define DW_FORM_udata         0x0f
#define DW_FORM_sec_offset    0x17
#define DW_FORM_exprloc       0x18

#define DW_LANG_C99           0x0001
#define DW_CHILDREN_yes       0x01
#define DW_CHILDREN_no        0x00

/* Abbreviation codes used in .debug_info */
#define ABBREV_COMPILE_UNIT   1
#define ABBREV_SUBPROGRAM     2

/* ── Helpers ────────────────────────────────────────────────────────────── */

DwarfBuilder *dwarf_begin(const char *filename) {
    DwarfBuilder *db = calloc(1, sizeof(DwarfBuilder));
    if (!db) return NULL;
    if (filename)
        snprintf(db->source_file, sizeof(db->source_file), "%s", filename);
    /* Register the primary file as file index 0 */
    if (filename)
        snprintf(db->files[db->file_count++], 256, "%s", filename);
    return db;
}

void dwarf_function(DwarfBuilder *db, const char *name,
                    int file_idx, int line, int col, int ret_type) {
    if (!db || db->func_count >= DWARF_MAX_FUNCS) return;
    DwarfFunc *f = &db->funcs[db->func_count++];
    if (name) snprintf(f->name, sizeof(f->name), "%s", name);
    f->file_idx = file_idx;
    f->line     = line;
    f->col      = col;
    f->ret_type = ret_type;
}

void dwarf_variable(DwarfBuilder *db, const char *name,
                    int type, int line, int col) {
    if (!db || db->var_count >= DWARF_MAX_VARS) return;
    DwarfVar *v = &db->vars[db->var_count++];
    if (name) snprintf(v->name, sizeof(v->name), "%s", name);
    v->type = type;
    v->line = line;
    v->col  = col;
}

void dwarf_line(DwarfBuilder *db, int file_idx, int line, int col,
                int addr_offset) {
    if (!db || db->loc_count >= DWARF_MAX_LOCS) return;
    DwarfLoc *l  = &db->locs[db->loc_count++];
    l->file_idx  = file_idx;
    l->line      = line;
    l->col       = col;
    l->addr_offset = addr_offset;
}

/* ── Section emission ───────────────────────────────────────────────────── */

/*
 * Emit the DWARF abbreviation table.
 * We define two abbreviation codes:
 *   1 = DW_TAG_compile_unit  (has children)
 *   2 = DW_TAG_subprogram    (no children, for brevity)
 */
static void emit_debug_abbrev(DwarfBuilder *db __attribute__((unused)),
                               FILE *out) {
    fprintf(out,
        "\t.section\t.debug_abbrev,\"\",@progbits\n"
        "\t.byte\t%d\t\t\t# abbrev code %d\n"
        "\t.byte\t0x%02x\t\t\t# DW_TAG_compile_unit\n"
        "\t.byte\t%d\t\t\t# DW_CHILDREN_yes\n"
        "\t.byte\t0x%02x\t\t\t# DW_AT_producer\n"
        "\t.byte\t0x%02x\t\t\t# DW_FORM_string\n"
        "\t.byte\t0x%02x\t\t\t# DW_AT_language\n"
        "\t.byte\t0x%02x\t\t\t# DW_FORM_udata\n"
        "\t.byte\t0x%02x\t\t\t# DW_AT_name\n"
        "\t.byte\t0x%02x\t\t\t# DW_FORM_string\n"
        "\t.byte\t0x%02x\t\t\t# DW_AT_stmt_list\n"
        "\t.byte\t0x%02x\t\t\t# DW_FORM_sec_offset\n"
        "\t.byte\t0x%02x\t\t\t# DW_AT_comp_dir\n"
        "\t.byte\t0x%02x\t\t\t# DW_FORM_string\n"
        "\t.byte\t0, 0\t\t\t# end attributes\n"
        "\n"
        "\t.byte\t%d\t\t\t# abbrev code %d\n"
        "\t.byte\t0x%02x\t\t\t# DW_TAG_subprogram\n"
        "\t.byte\t%d\t\t\t# DW_CHILDREN_no\n"
        "\t.byte\t0x%02x\t\t\t# DW_AT_external\n"
        "\t.byte\t0x%02x\t\t\t# DW_FORM_flag_present\n"
        "\t.byte\t0x%02x\t\t\t# DW_AT_name\n"
        "\t.byte\t0x%02x\t\t\t# DW_FORM_string\n"
        "\t.byte\t0x%02x\t\t\t# DW_AT_decl_file\n"
        "\t.byte\t0x%02x\t\t\t# DW_FORM_udata\n"
        "\t.byte\t0x%02x\t\t\t# DW_AT_decl_line\n"
        "\t.byte\t0x%02x\t\t\t# DW_FORM_udata\n"
        "\t.byte\t0, 0\t\t\t# end attributes\n"
        "\t.byte\t0\t\t\t# end abbreviation table\n",
        ABBREV_COMPILE_UNIT, ABBREV_COMPILE_UNIT,
        DW_TAG_compile_unit, DW_CHILDREN_yes,
        DW_AT_producer,  DW_FORM_string,
        DW_AT_language,  DW_FORM_udata,
        DW_AT_name,      DW_FORM_string,
        DW_AT_stmt_list, DW_FORM_sec_offset,
        DW_AT_comp_dir,  DW_FORM_string,
        ABBREV_SUBPROGRAM, ABBREV_SUBPROGRAM,
        DW_TAG_subprogram, DW_CHILDREN_no,
        DW_AT_external,  DW_FORM_flag_present,
        DW_AT_name,      DW_FORM_string,
        DW_AT_decl_file, DW_FORM_udata,
        DW_AT_decl_line, DW_FORM_udata);
}

/* Emit a null-terminated DWARF string in assembly (.ascii + .byte 0) */
static void emit_dwarf_string(FILE *out, const char *s) {
    fprintf(out, "\t.ascii\t\"");
    for (const char *p = s; *p; p++) {
        if (*p == '"' || *p == '\\')
            fprintf(out, "\\%c", *p);
        else if ((unsigned char)*p < 0x20)
            fprintf(out, "\\%03o", (unsigned char)*p);
        else
            fputc(*p, out);
    }
    fprintf(out, "\"\n\t.byte\t0\t\t# null terminator\n");
}

static void emit_debug_info(DwarfBuilder *db, FILE *out) {
    const char *fname = db->source_file[0] ? db->source_file : "<unknown>";

    fprintf(out,
        "\n\t.section\t.debug_info,\"\",@progbits\n"
        ".Ldebug_info_begin:\n"
        "\t.long\t.Ldebug_info_end - .Ldebug_info_begin - 4\t# unit length\n"
        "\t.short\t4\t\t\t\t# DWARF version 4\n"
        "\t.long\t.debug_abbrev\t\t\t# abbrev offset\n"
        "\t.byte\t8\t\t\t\t# address size (64-bit)\n"
        "\t.byte\t%d\t\t\t\t# abbrev code: DW_TAG_compile_unit\n",
        ABBREV_COMPILE_UNIT);

    /* DW_AT_producer */
    emit_dwarf_string(out, "nanoc");
    /* DW_AT_language: DW_LANG_C99 = 0x0001 (uleb128) */
    fprintf(out, "\t.uleb128\t0x%04x\t\t\t# DW_LANG_C99\n", DW_LANG_C99);
    /* DW_AT_name */
    emit_dwarf_string(out, fname);
    /* DW_AT_stmt_list: offset 0 into .debug_line */
    fprintf(out, "\t.long\t0\t\t\t\t# DW_AT_stmt_list\n");
    /* DW_AT_comp_dir */
    emit_dwarf_string(out, ".");

    /* Subprogram DIEs */
    for (int i = 0; i < db->func_count; i++) {
        DwarfFunc *f = &db->funcs[i];
        fprintf(out,
            "\t# subprogram: %s (line %d)\n"
            "\t.byte\t%d\t\t\t\t# abbrev code: DW_TAG_subprogram\n",
            f->name[0] ? f->name : "?", f->line,
            ABBREV_SUBPROGRAM);
        /* DW_AT_external: flag_present — zero-width attribute */
        /* DW_AT_name */
        emit_dwarf_string(out, f->name[0] ? f->name : "?");
        /* DW_AT_decl_file (uleb128) */
        fprintf(out, "\t.uleb128\t%d\t\t\t# DW_AT_decl_file\n", f->file_idx + 1);
        /* DW_AT_decl_line (uleb128) */
        fprintf(out, "\t.uleb128\t%d\t\t\t# DW_AT_decl_line\n", f->line);
    }

    /* End of children (compile_unit) */
    fprintf(out, "\t.byte\t0\t\t\t\t# end of children\n");
    fprintf(out, ".Ldebug_info_end:\n");
}

static void emit_debug_line_stub(DwarfBuilder *db __attribute__((unused)),
                                  FILE *out) {
    /*
     * The real line number information for the RISC-V path comes from the
     * .loc directives emitted inline in .text.  GAS synthesizes the
     * .debug_line section automatically from those directives.
     * We emit a minimal header here for completeness / standalone use.
     */
    fprintf(out,
        "\n\t.section\t.debug_line,\"\",@progbits\n"
        "\t# .debug_line synthesized by assembler from .loc directives\n");
}

static void emit_debug_str(DwarfBuilder *db, FILE *out) {
    fprintf(out, "\n\t.section\t.debug_str,\"MS\",@progbits,1\n");
    fprintf(out, ".Lproducer_str:\n");
    emit_dwarf_string(out, "nanoc");
    if (db->source_file[0]) {
        fprintf(out, ".Lsource_file_str:\n");
        emit_dwarf_string(out, db->source_file);
    }
}

void dwarf_emit_asm_sections(DwarfBuilder *db, FILE *out) {
    if (!db || !out) return;
    emit_debug_abbrev(db, out);
    emit_debug_info(db, out);
    emit_debug_line_stub(db, out);
    emit_debug_str(db, out);
}

void dwarf_free(DwarfBuilder *db) {
    free(db);
}

/**
 * @file toon_output.c
 * @brief TOON format output implementation for LLM-friendly diagnostics
 *
 * Provides diagnostic output in Token-Oriented Object Notation (TOON)
 * format for reduced token consumption in LLM agent workflows.
 *
 * WORKAROUND: We initially vendored TOONc (https://github.com/UsboKirishima/TOONc)
 * but its printObject/printRoot functions output debug format with type
 * annotations and "{...}" placeholders, not valid serialized TOON.
 * We implement TOON tabular format directly until upstream provides a
 * proper emitter. TOONc removed to avoid dead code.
 *
 * See: https://toonformat.dev/
 */

#include "toon_output.h"
#include <stdlib.h>
#include <string.h>

/* Maximum diagnostics to accumulate */
#define MAX_TOON_DIAGNOSTICS 256

/* Diagnostic entry structure */
typedef struct {
    char *severity;
    char *code;
    char *message;
    char *file;
    int line;
    int column;
} ToonDiagnostic;

/* Global state */
static bool g_toon_output_enabled = false;
static ToonDiagnostic g_toon_diagnostics[MAX_TOON_DIAGNOSTICS];
static int g_toon_diagnostic_count = 0;

/* Helper: duplicate string safely */
static char *toon_strdup(const char *s) {
    if (s == NULL) return NULL;
    size_t len = strlen(s);
    char *copy = malloc(len + 1);
    if (copy) {
        memcpy(copy, s, len + 1);
    }
    return copy;
}

void toon_diagnostics_enable(void) {
    g_toon_output_enabled = true;
}

bool toon_diagnostics_enabled(void) {
    return g_toon_output_enabled;
}

void toon_diagnostics_add(const char *severity, const char *code,
                          const char *message, const char *file,
                          int line, int column) {
    if (!g_toon_output_enabled) return;
    if (g_toon_diagnostic_count >= MAX_TOON_DIAGNOSTICS) return;

    ToonDiagnostic *d = &g_toon_diagnostics[g_toon_diagnostic_count++];
    d->severity = toon_strdup(severity);
    d->code = toon_strdup(code);
    d->message = toon_strdup(message);
    d->file = toon_strdup(file);
    d->line = line;
    d->column = column;
}

/* Write a string field, escaping tabs, newlines, and backslashes */
static void toon_write_field(FILE *fp, const char *s) {
    if (!s) return;
    for (const char *p = s; *p; p++) {
        switch (*p) {
        case '\t': fprintf(fp, "\\t"); break;
        case '\n': fprintf(fp, "\\n"); break;
        case '\r': fprintf(fp, "\\r"); break;
        case '\\': fprintf(fp, "\\\\"); break;
        default:   fputc(*p, fp); break;
        }
    }
}

void toon_diagnostics_output(FILE *fp, const char *input_file,
                             const char *output_file, int exit_code) {
    if (!g_toon_output_enabled || !fp) return;

    /* Metadata */
    fprintf(fp, "tool: nanoc_c\n");
    fprintf(fp, "success: %s\n", exit_code == 0 ? "true" : "false");
    fprintf(fp, "exit_code: %d\n", exit_code);
    fprintf(fp, "input_file: %s\n", input_file ? input_file : "");
    fprintf(fp, "output_file: %s\n", output_file ? output_file : "");

    /* Diagnostics table */
    fprintf(fp, "diagnostics[%d]:\n", g_toon_diagnostic_count);

    if (g_toon_diagnostic_count > 0) {
        /* Column headers (tab-separated) */
        fprintf(fp, "  severity\tcode\tmessage\tfile\tline\tcolumn\n");

        /* Data rows */
        for (int i = 0; i < g_toon_diagnostic_count; i++) {
            ToonDiagnostic *d = &g_toon_diagnostics[i];
            fprintf(fp, "  ");
            toon_write_field(fp, d->severity ? d->severity : "");
            fputc('\t', fp);
            toon_write_field(fp, d->code ? d->code : "");
            fputc('\t', fp);
            toon_write_field(fp, d->message ? d->message : "");
            fputc('\t', fp);
            toon_write_field(fp, d->file ? d->file : "");
            fprintf(fp, "\t%d\t%d\n", d->line, d->column);
        }
    }

    /* Summary */
    fprintf(fp, "diagnostic_count: %d\n", g_toon_diagnostic_count);
    fprintf(fp, "has_diagnostics: %s\n", g_toon_diagnostic_count > 0 ? "true" : "false");
}

bool toon_diagnostics_output_to_file(const char *path, const char *input_file,
                                     const char *output_file, int exit_code) {
    if (!path) return false;

    FILE *fp = fopen(path, "w");
    if (!fp) return false;

    toon_diagnostics_output(fp, input_file, output_file, exit_code);
    fclose(fp);
    return true;
}

void toon_diagnostics_cleanup(void) {
    for (int i = 0; i < g_toon_diagnostic_count; i++) {
        ToonDiagnostic *d = &g_toon_diagnostics[i];
        free(d->severity);
        free(d->code);
        free(d->message);
        free(d->file);
    }
    g_toon_diagnostic_count = 0;
    /* Note: Do not reset g_toon_output_enabled here - that's controlled by --llm-diags-toon flag */
}

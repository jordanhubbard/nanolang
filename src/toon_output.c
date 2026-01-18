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

void toon_diagnostics_output(FILE *fp) {
    if (!g_toon_output_enabled || !fp) return;

    /*
     * Output TOON format directly.
     * TOON uses tabular format for uniform arrays of objects.
     * Format:
     *   diagnostics[N]:
     *     severity	code	message	file	line	column
     *     error	E001	msg	file.nano	10	5
     *   error_count: N
     *   has_errors: true/false
     */

    /* Header */
    fprintf(fp, "diagnostics[%d]:\n", g_toon_diagnostic_count);

    if (g_toon_diagnostic_count > 0) {
        /* Column headers (tab-separated) */
        fprintf(fp, "  severity\tcode\tmessage\tfile\tline\tcolumn\n");

        /* Data rows */
        for (int i = 0; i < g_toon_diagnostic_count; i++) {
            ToonDiagnostic *d = &g_toon_diagnostics[i];
            fprintf(fp, "  %s\t%s\t%s\t%s\t%d\t%d\n",
                    d->severity ? d->severity : "",
                    d->code ? d->code : "",
                    d->message ? d->message : "",
                    d->file ? d->file : "",
                    d->line,
                    d->column);
        }
    }

    /* Summary fields */
    fprintf(fp, "error_count: %d\n", g_toon_diagnostic_count);
    fprintf(fp, "has_errors: %s\n", g_toon_diagnostic_count > 0 ? "true" : "false");
}

bool toon_diagnostics_output_to_file(const char *path) {
    if (!path) return false;

    FILE *fp = fopen(path, "w");
    if (!fp) return false;

    toon_diagnostics_output(fp);
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

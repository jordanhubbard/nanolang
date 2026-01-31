/* json_diagnostics.c - Structured JSON error output implementation */

#include "json_diagnostics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Global state */
bool g_json_output_enabled = false;
Diagnostic *g_diagnostics = NULL;
int g_diagnostic_count = 0;
int g_diagnostic_capacity = 0;

void json_diagnostics_init(void) {
    g_diagnostic_count = 0;
    g_diagnostic_capacity = 10;
    g_diagnostics = (Diagnostic *)malloc(sizeof(Diagnostic) * g_diagnostic_capacity);
}

void json_diagnostics_enable(void) {
    g_json_output_enabled = true;
}

static char *strdup_safe(const char *s) {
    if (s == NULL) return NULL;
    char *copy = (char *)malloc(strlen(s) + 1);
    strcpy(copy, s);
    return copy;
}

void json_diagnostics_add(DiagnosticSeverity severity, const char *code,
                          const char *message, const char *file,
                          int line, int column, const char *suggestion) {
    if (!g_json_output_enabled) return;
    
    /* Expand capacity if needed */
    if (g_diagnostic_count >= g_diagnostic_capacity) {
        g_diagnostic_capacity *= 2;
        g_diagnostics = (Diagnostic *)realloc(g_diagnostics,
                                              sizeof(Diagnostic) * g_diagnostic_capacity);
    }
    
    /* Add diagnostic */
    Diagnostic *diag = &g_diagnostics[g_diagnostic_count++];
    diag->severity = severity;
    diag->code = strdup_safe(code);
    diag->message = strdup_safe(message);
    diag->file = strdup_safe(file);
    diag->line = line;
    diag->column = column;
    diag->suggestion = strdup_safe(suggestion);
}

/* Escape string for JSON */
static void print_json_string(FILE *f, const char *s) {
    if (s == NULL) {
        fprintf(f, "null");
        return;
    }
    
    fprintf(f, "\"");
    for (const char *p = s; *p; p++) {
        switch (*p) {
            case '"':  fprintf(f, "\\\""); break;
            case '\\': fprintf(f, "\\\\"); break;
            case '\n': fprintf(f, "\\n"); break;
            case '\r': fprintf(f, "\\r"); break;
            case '\t': fprintf(f, "\\t"); break;
            default:
                if (*p < 32) {
                    fprintf(f, "\\u%04x", (unsigned char)*p);
                } else {
                    fputc(*p, f);
                }
        }
    }
    fprintf(f, "\"");
}

static const char *severity_to_string(DiagnosticSeverity sev) {
    switch (sev) {
        case DIAG_ERROR:   return "error";
        case DIAG_WARNING: return "warning";
        case DIAG_INFO:    return "info";
        case DIAG_HINT:    return "hint";
        default:           return "unknown";
    }
}

void json_diagnostics_output(void) {
    if (!g_json_output_enabled) return;
    
    fprintf(stdout, "{\n");
    fprintf(stdout, "  \"diagnostics\": [\n");
    
    for (int i = 0; i < g_diagnostic_count; i++) {
        Diagnostic *diag = &g_diagnostics[i];
        
        fprintf(stdout, "    {\n");
        fprintf(stdout, "      \"severity\": \"%s\",\n", severity_to_string(diag->severity));
        fprintf(stdout, "      \"code\": ");
        print_json_string(stdout, diag->code);
        fprintf(stdout, ",\n");
        
        fprintf(stdout, "      \"message\": ");
        print_json_string(stdout, diag->message);
        fprintf(stdout, ",\n");
        
        fprintf(stdout, "      \"location\": {\n");
        fprintf(stdout, "        \"file\": ");
        print_json_string(stdout, diag->file);
        fprintf(stdout, ",\n");
        fprintf(stdout, "        \"line\": %d,\n", diag->line);
        fprintf(stdout, "        \"column\": %d\n", diag->column);
        fprintf(stdout, "      }");
        
        if (diag->suggestion != NULL) {
            fprintf(stdout, ",\n");
            fprintf(stdout, "      \"suggestion\": ");
            print_json_string(stdout, diag->suggestion);
        }
        
        fprintf(stdout, "\n    }");
        if (i < g_diagnostic_count - 1) {
            fprintf(stdout, ",");
        }
        fprintf(stdout, "\n");
    }
    
    fprintf(stdout, "  ],\n");
    fprintf(stdout, "  \"error_count\": %d,\n", g_diagnostic_count);
    fprintf(stdout, "  \"has_errors\": %s\n", g_diagnostic_count > 0 ? "true" : "false");
    fprintf(stdout, "}\n");
}

void json_diagnostics_cleanup(void) {
    for (int i = 0; i < g_diagnostic_count; i++) {
        free(g_diagnostics[i].code);
        free(g_diagnostics[i].message);
        free(g_diagnostics[i].file);
        free(g_diagnostics[i].suggestion);
    }
    free(g_diagnostics);
    g_diagnostics = NULL;
    g_diagnostic_count = 0;
    g_diagnostic_capacity = 0;
}

void json_error(const char *code, const char *message, const char *file,
                int line, int column, const char *suggestion) {
    json_diagnostics_add(DIAG_ERROR, code, message, file, line, column, suggestion);
}

void json_warning(const char *code, const char *message, const char *file,
                  int line, int column, const char *suggestion) {
    json_diagnostics_add(DIAG_WARNING, code, message, file, line, column, suggestion);
}

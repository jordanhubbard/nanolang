/* =============================================================================
 * Standard Library: Logging C Backend
 * =============================================================================
 * C implementation for stateful logging operations.
 * Provides global state management for log levels, output modes, and tracing.
 * ============================================================================= */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Log levels */
#define LOG_LEVEL_DEBUG 0
#define LOG_LEVEL_INFO  1
#define LOG_LEVEL_WARN  2
#define LOG_LEVEL_ERROR 3
#define LOG_LEVEL_FATAL 4

/* Output modes */
#define OUTPUT_MODE_TEXT 0
#define OUTPUT_MODE_JSON 1

/* Global state */
static int g_current_log_level = LOG_LEVEL_INFO;
static int g_output_mode = OUTPUT_MODE_TEXT;
static FILE *g_log_file = NULL;
static int g_next_trace_id = 1;

/* =============================================================================
 * Configuration Functions
 * ============================================================================= */

void nl_log_set_level(int64_t level) {
    if (level >= LOG_LEVEL_DEBUG && level <= LOG_LEVEL_FATAL) {
        g_current_log_level = (int)level;
    }
}

int64_t nl_log_get_level(void) {
    return (int64_t)g_current_log_level;
}

void nl_log_set_output_mode(int64_t mode) {
    if (mode == OUTPUT_MODE_TEXT || mode == OUTPUT_MODE_JSON) {
        g_output_mode = (int)mode;
    }
}

int64_t nl_log_get_output_mode(void) {
    return (int64_t)g_output_mode;
}

void nl_log_set_file(const char *path) {
    /* Close existing file if open */
    if (g_log_file != NULL && g_log_file != stdout && g_log_file != stderr) {
        fclose(g_log_file);
        g_log_file = NULL;
    }
    
    /* Empty path means stdout */
    if (path == NULL || strlen(path) == 0) {
        g_log_file = NULL;
        return;
    }
    
    /* Open new log file */
    g_log_file = fopen(path, "a");
    if (g_log_file == NULL) {
        fprintf(stderr, "Warning: Failed to open log file: %s\n", path);
    }
}

/* =============================================================================
 * Logging Functions
 * ============================================================================= */

static const char* level_to_string(int level) {
    switch (level) {
        case LOG_LEVEL_DEBUG: return "DEBUG";
        case LOG_LEVEL_INFO:  return "INFO";
        case LOG_LEVEL_WARN:  return "WARN";
        case LOG_LEVEL_ERROR: return "ERROR";
        case LOG_LEVEL_FATAL: return "FATAL";
        default:              return "UNKNOWN";
    }
}

static FILE* get_output_stream(void) {
    return (g_log_file != NULL) ? g_log_file : stdout;
}

static void json_escape(FILE *out, const char *s) {
    if (!s) return;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        unsigned char c = *p;
        switch (c) {
            case '\\': fputs("\\\\", out); break;
            case '"': fputs("\\\"", out); break;
            case '\n': fputs("\\n", out); break;
            case '\r': fputs("\\r", out); break;
            case '\t': fputs("\\t", out); break;
            default:
                if (c < 0x20) fprintf(out, "\\u%04x", (unsigned int)c);
                else fputc((int)c, out);
        }
    }
}

void nl_log_write(int64_t level, const char *message) {
    if (level < g_current_log_level) {
        return;  /* Filtered by log level */
    }
    
    FILE *out = get_output_stream();
    
    if (g_output_mode == OUTPUT_MODE_JSON) {
        /* JSON format */
        fputs("{\"level\":\"", out);
        json_escape(out, level_to_string((int)level));
        fputs("\",\"message\":\"", out);
        json_escape(out, message);
        fputs("\"}\n", out);
    } else {
        /* Text format */
        fprintf(out, "[%s] %s\n", level_to_string((int)level), message);
    }
    
    fflush(out);
}

/* =============================================================================
 * Tracing Functions
 * ============================================================================= */

int64_t nl_log_trace_enter(const char *fn_name) {
    int64_t trace_id = g_next_trace_id++;
    
    if (LOG_LEVEL_DEBUG >= g_current_log_level) {
        FILE *out = get_output_stream();
        if (g_output_mode == OUTPUT_MODE_JSON) {
            fputs("{\"level\":\"DEBUG\",\"type\":\"trace_enter\",\"function\":\"", out);
            json_escape(out, fn_name);
            fprintf(out, "\",\"trace_id\":%lld}\n", (long long)trace_id);
        } else {
            fprintf(out, "[DEBUG] ENTER %s (trace_id=%lld)\n", fn_name, (long long)trace_id);
        }
        fflush(out);
    }
    
    return trace_id;
}

void nl_log_trace_exit(int64_t trace_id, const char *fn_name) {
    if (LOG_LEVEL_DEBUG >= g_current_log_level) {
        FILE *out = get_output_stream();
        if (g_output_mode == OUTPUT_MODE_JSON) {
            fputs("{\"level\":\"DEBUG\",\"type\":\"trace_exit\",\"function\":\"", out);
            json_escape(out, fn_name);
            fprintf(out, "\",\"trace_id\":%lld}\n", (long long)trace_id);
        } else {
            fprintf(out, "[DEBUG] EXIT %s (trace_id=%lld)\n", fn_name, (long long)trace_id);
        }
        fflush(out);
    }
}

void nl_log_trace_event(const char *name, const char *data) {
    if (LOG_LEVEL_DEBUG >= g_current_log_level) {
        FILE *out = get_output_stream();
        if (g_output_mode == OUTPUT_MODE_JSON) {
            fputs("{\"level\":\"DEBUG\",\"type\":\"trace_event\",\"name\":\"", out);
            json_escape(out, name);
            fputs("\",\"data\":\"", out);
            json_escape(out, data);
            fputs("\"}\n", out);
        } else {
            fprintf(out, "[DEBUG] EVENT %s: %s\n", name, data);
        }
        fflush(out);
    }
}

/* =============================================================================
 * Cleanup
 * ============================================================================= */

void nl_log_cleanup(void) {
    if (g_log_file != NULL && g_log_file != stdout && g_log_file != stderr) {
        fclose(g_log_file);
        g_log_file = NULL;
    }
}

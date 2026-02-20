#include "nano_tools.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

#define MAX_OUTPUT 65536

static const char* find_nanoc(void) {
    static char nanoc_path[1024] = "";
    if (nanoc_path[0]) return nanoc_path;

    const char *root = getenv("NANO_PROJECT_ROOT");
    if (root) {
        snprintf(nanoc_path, sizeof(nanoc_path), "%s/bin/nanoc", root);
        if (access(nanoc_path, X_OK) == 0) return nanoc_path;
    }

    const char *candidates[] = {
        "./bin/nanoc", "../bin/nanoc", "../../bin/nanoc", NULL
    };
    for (int i = 0; candidates[i]; i++) {
        if (access(candidates[i], X_OK) == 0) {
            snprintf(nanoc_path, sizeof(nanoc_path), "%s", candidates[i]);
            return nanoc_path;
        }
    }

    snprintf(nanoc_path, sizeof(nanoc_path), "./bin/nanoc");
    return nanoc_path;
}

/* Helper to escape JSON strings */
static char* json_escape_string(const char* input) {
    if (!input) return strdup("");
    
    size_t len = strlen(input);
    char* output = malloc(len * 2 + 1);  /* Worst case: every char escaped */
    if (!output) return strdup("");
    
    size_t j = 0;
    for (size_t i = 0; i < len; i++) {
        char c = input[i];
        switch (c) {
            case '"':  output[j++] = '\\'; output[j++] = '"'; break;
            case '\\': output[j++] = '\\'; output[j++] = '\\'; break;
            case '\n': output[j++] = '\\'; output[j++] = 'n'; break;
            case '\r': output[j++] = '\\'; output[j++] = 'r'; break;
            case '\t': output[j++] = '\\'; output[j++] = 't'; break;
            default:   output[j++] = c; break;
        }
    }
    output[j] = '\0';
    return output;
}

int64_t eval(const char *source) {
    if (!source) {
        return 1;
    }

    char src_template[] = "/tmp/nano_eval_XXXXXX";
    int src_fd = mkstemp(src_template);
    if (src_fd < 0) {
        return 1;
    }

    FILE *src_file = fdopen(src_fd, "w");
    if (!src_file) {
        close(src_fd);
        unlink(src_template);
        return 1;
    }

    fputs(source, src_file);
    fclose(src_file);

    char bin_template[] = "/tmp/nano_eval_bin_XXXXXX";
    int bin_fd = mkstemp(bin_template);
    if (bin_fd < 0) {
        unlink(src_template);
        return 1;
    }
    close(bin_fd);

    char compile_cmd[2048];
    snprintf(compile_cmd, sizeof(compile_cmd), "%s %s -o %s 2>&1", find_nanoc(), src_template, bin_template);
    int compile_status = system(compile_cmd);

    if (compile_status != 0) {
        unlink(src_template);
        unlink(bin_template);
        return 1;
    }

    int run_status = system(bin_template);

    unlink(src_template);
    unlink(bin_template);

    if (WIFEXITED(run_status)) {
        return WEXITSTATUS(run_status) == 0 ? 0 : 1;
    }

    return 1;
}

/* Eval with output capture - returns JSON result */
char* eval_with_output(const char *source) {
    char* result = malloc(MAX_OUTPUT);
    if (!result) return strdup("{\"success\":false,\"error\":\"Memory allocation failed\"}");
    
    if (!source || strlen(source) == 0) {
        snprintf(result, MAX_OUTPUT, "{\"success\":false,\"error\":\"No code provided\"}");
        return result;
    }

    /* Create temp file for source */
    char src_template[] = "/tmp/nano_eval_XXXXXX";
    int src_fd = mkstemp(src_template);
    if (src_fd < 0) {
        snprintf(result, MAX_OUTPUT, "{\"success\":false,\"error\":\"Failed to create temp file\"}");
        return result;
    }

    FILE *src_file = fdopen(src_fd, "w");
    if (!src_file) {
        close(src_fd);
        unlink(src_template);
        snprintf(result, MAX_OUTPUT, "{\"success\":false,\"error\":\"Failed to open temp file\"}");
        return result;
    }

    fputs(source, src_file);
    fclose(src_file);

    /* Create temp file for binary */
    char bin_template[] = "/tmp/nano_eval_bin_XXXXXX";
    int bin_fd = mkstemp(bin_template);
    if (bin_fd < 0) {
        unlink(src_template);
        snprintf(result, MAX_OUTPUT, "{\"success\":false,\"error\":\"Failed to create binary temp file\"}");
        return result;
    }
    close(bin_fd);

    /* Compile with output capture */
    char compile_cmd[4096];
    char compile_output_file[] = "/tmp/nano_compile_out_XXXXXX";
    int compile_out_fd = mkstemp(compile_output_file);
    if (compile_out_fd < 0) {
        unlink(src_template);
        unlink(bin_template);
        snprintf(result, MAX_OUTPUT, "{\"success\":false,\"error\":\"Failed to create output file\"}");
        return result;
    }
    close(compile_out_fd);

    snprintf(compile_cmd, sizeof(compile_cmd), 
             "NANO_MODULE_PATH=modules %s %s -o %s >%s 2>&1", 
             find_nanoc(), src_template, bin_template, compile_output_file);
    
    int compile_status = system(compile_cmd);

    /* Read compile output */
    char compile_output[MAX_OUTPUT / 2] = "";
    FILE* compile_out = fopen(compile_output_file, "r");
    if (compile_out) {
        size_t read_len = fread(compile_output, 1, sizeof(compile_output) - 1, compile_out);
        compile_output[read_len] = '\0';
        fclose(compile_out);
    }
    unlink(compile_output_file);

    if (compile_status != 0) {
        unlink(src_template);
        unlink(bin_template);
        char* escaped_output = json_escape_string(compile_output);
        snprintf(result, MAX_OUTPUT, 
                 "{\"success\":false,\"error\":\"Compilation failed\",\"output\":\"%s\"}", 
                 escaped_output);
        free(escaped_output);
        return result;
    }

    /* Run with output capture */
    char run_output_file[] = "/tmp/nano_run_out_XXXXXX";
    int run_out_fd = mkstemp(run_output_file);
    if (run_out_fd < 0) {
        unlink(src_template);
        unlink(bin_template);
        snprintf(result, MAX_OUTPUT, "{\"success\":false,\"error\":\"Failed to create run output file\"}");
        return result;
    }
    close(run_out_fd);

    char run_cmd[4096];
    snprintf(run_cmd, sizeof(run_cmd), "%s >%s 2>&1", bin_template, run_output_file);
    int run_status = system(run_cmd);

    /* Read run output */
    char run_output[MAX_OUTPUT / 2] = "";
    FILE* run_out = fopen(run_output_file, "r");
    if (run_out) {
        size_t read_len = fread(run_output, 1, sizeof(run_output) - 1, run_out);
        run_output[read_len] = '\0';
        fclose(run_out);
    }
    unlink(run_output_file);

    /* Cleanup */
    unlink(src_template);
    unlink(bin_template);

    /* Build result JSON */
    int success = WIFEXITED(run_status) && WEXITSTATUS(run_status) == 0;
    char* escaped_compile = json_escape_string(compile_output);
    char* escaped_run = json_escape_string(run_output);
    
    snprintf(result, MAX_OUTPUT,
             "{\"success\":%s,\"compile_output\":\"%s\",\"output\":\"%s\",\"exit_code\":%d}",
             success ? "true" : "false",
             escaped_compile,
             escaped_run,
             WIFEXITED(run_status) ? WEXITSTATUS(run_status) : -1);
    
    free(escaped_compile);
    free(escaped_run);
    
    return result;
}

void eval_free_result(char *result) {
    if (result) free(result);
}

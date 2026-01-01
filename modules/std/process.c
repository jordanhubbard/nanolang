#include "process.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

/* Helper: Create nanolang array for strings */
static DynArray* create_string_array(int64_t initial_capacity) {
    DynArray* arr = (DynArray*)malloc(sizeof(DynArray));
    if (!arr) return NULL;
    
    arr->length = 0;
    arr->capacity = initial_capacity;
    arr->elem_type = ELEM_STRING;
    arr->elem_size = sizeof(char*);
    arr->data = calloc(initial_capacity, sizeof(char*));
    
    if (!arr->data) {
        free(arr);
        return NULL;
    }
    
    return arr;
}

/* Helper: Append string to array */
static void array_append_string(DynArray* arr, const char* str) {
    if (!arr || !str) return;
    
    /* Grow if needed */
    if (arr->length >= arr->capacity) {
        int64_t new_capacity = arr->capacity * 2;
        char** new_data = (char**)realloc(arr->data, new_capacity * sizeof(char*));
        if (!new_data) return;
        arr->data = new_data;
        arr->capacity = new_capacity;
    }
    
    /* Duplicate string and add to array */
    char* dup = strdup(str);
    if (dup) {
        ((char**)arr->data)[arr->length++] = dup;
    }
}

/* Run a command and capture stdout/stderr
 * Returns array<string> with [exit_code, stdout, stderr]
 */
DynArray* nl_os_process_run(const char* command) {
    DynArray* result = create_string_array(3);
    if (!result) return NULL;
    
    /* Create temporary files for stdout and stderr */
    char stdout_file[] = "/tmp/nanolang_stdout_XXXXXX";
    char stderr_file[] = "/tmp/nanolang_stderr_XXXXXX";
    
    int stdout_fd = mkstemp(stdout_file);
    int stderr_fd = mkstemp(stderr_file);
    
    if (stdout_fd == -1 || stderr_fd == -1) {
        if (stdout_fd != -1) {
            close(stdout_fd);
            unlink(stdout_file);
        }
        if (stderr_fd != -1) {
            close(stderr_fd);
            unlink(stderr_file);
        }
        array_append_string(result, "-1");
        array_append_string(result, "");
        array_append_string(result, "Failed to create temp files");
        return result;
    }
    
    close(stdout_fd);
    close(stderr_fd);
    
    /* Build command with redirects */
    char full_command[4096];
    snprintf(full_command, sizeof(full_command), "%s > %s 2> %s", 
             command, stdout_file, stderr_file);
    
    /* Execute command */
    int exit_code = system(full_command);
    if (exit_code == -1) {
        unlink(stdout_file);
        unlink(stderr_file);
        array_append_string(result, "-1");
        array_append_string(result, "");
        array_append_string(result, "Failed to execute command");
        return result;
    }
    
    /* Get actual exit code */
    int actual_exit = WIFEXITED(exit_code) ? WEXITSTATUS(exit_code) : -1;
    
    /* Read stdout */
    FILE* stdout_fp = fopen(stdout_file, "r");
    char* stdout_content = NULL;
    size_t stdout_size = 0;
    
    if (stdout_fp) {
        fseek(stdout_fp, 0, SEEK_END);
        stdout_size = ftell(stdout_fp);
        fseek(stdout_fp, 0, SEEK_SET);
        
        stdout_content = (char*)malloc(stdout_size + 1);
        if (stdout_content) {
            size_t read = fread(stdout_content, 1, stdout_size, stdout_fp);
            stdout_content[read] = '\0';
        }
        fclose(stdout_fp);
    }
    
    /* Read stderr */
    FILE* stderr_fp = fopen(stderr_file, "r");
    char* stderr_content = NULL;
    size_t stderr_size = 0;
    
    if (stderr_fp) {
        fseek(stderr_fp, 0, SEEK_END);
        stderr_size = ftell(stderr_fp);
        fseek(stderr_fp, 0, SEEK_SET);
        
        stderr_content = (char*)malloc(stderr_size + 1);
        if (stderr_content) {
            size_t read = fread(stderr_content, 1, stderr_size, stderr_fp);
            stderr_content[read] = '\0';
        }
        fclose(stderr_fp);
    }
    
    /* Clean up temp files */
    unlink(stdout_file);
    unlink(stderr_file);
    
    /* Build result array */
    char exit_code_str[32];
    snprintf(exit_code_str, sizeof(exit_code_str), "%d", actual_exit);
    array_append_string(result, exit_code_str);
    array_append_string(result, stdout_content ? stdout_content : "");
    array_append_string(result, stderr_content ? stderr_content : "");
    
    if (stdout_content) free(stdout_content);
    if (stderr_content) free(stderr_content);
    
    return result;
}


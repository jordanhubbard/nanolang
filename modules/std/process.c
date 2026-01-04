#include "process.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

/* Use runtime DynArray API */
extern DynArray* dyn_array_new_with_capacity(ElementType elem_type, int64_t initial_capacity);
extern DynArray* dyn_array_push_string_copy(DynArray* arr, const char* value);

/* Run a command and capture stdout/stderr
 * Returns array<string> with [exit_code, stdout, stderr]
 */
DynArray* nl_os_process_run(const char* command) {
    DynArray* result = dyn_array_new_with_capacity(ELEM_STRING, 3);
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
        dyn_array_push_string_copy(result, "-1");
        dyn_array_push_string_copy(result, "");
        dyn_array_push_string_copy(result, "Failed to create temp files");
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
        dyn_array_push_string_copy(result, "-1");
        dyn_array_push_string_copy(result, "");
        dyn_array_push_string_copy(result, "Failed to execute command");
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
    dyn_array_push_string_copy(result, exit_code_str);
    dyn_array_push_string_copy(result, stdout_content ? stdout_content : "");
    dyn_array_push_string_copy(result, stderr_content ? stderr_content : "");
    
    if (stdout_content) free(stdout_content);
    if (stderr_content) free(stderr_content);
    
    return result;
}


/* Spawn a process non-blocking
 * Returns process ID (pid) or -1 on error
 */
int64_t nl_os_process_spawn(const char* command) {
    pid_t pid = fork();
    
    if (pid < 0) {
        /* Fork failed */
        return -1;
    } else if (pid == 0) {
        /* Child process */
        /* Use sh -c to execute command string */
        execl("/bin/sh", "sh", "-c", command, (char*)NULL);
        /* If exec fails, exit child */
        _exit(127);
    } else {
        /* Parent process - return child PID */
        return (int64_t)pid;
    }
}

/* Check if a process is still running
 * Returns 1 if running, 0 if exited, -1 on error
 */
int64_t nl_os_process_is_running(int64_t pid) {
    if (pid <= 0) return -1;
    
    int status;
    pid_t result = waitpid((pid_t)pid, &status, WNOHANG);
    
    if (result == 0) {
        /* Process is still running */
        return 1;
    } else if (result == (pid_t)pid) {
        /* Process has exited */
        return 0;
    } else {
        /* Error (e.g., no such process) */
        return -1;
    }
}

/* Wait for a process to complete
 * Returns exit code of the process, or -1 on error
 */
int64_t nl_os_process_wait(int64_t pid) {
    if (pid <= 0) return -1;
    
    int status;
    pid_t result = waitpid((pid_t)pid, &status, 0);
    
    if (result == (pid_t)pid) {
        if (WIFEXITED(status)) {
            return (int64_t)WEXITSTATUS(status);
        } else if (WIFSIGNALED(status)) {
            /* Process terminated by signal - return negative signal number */
            return -(int64_t)WTERMSIG(status);
        } else {
            return -1;
        }
    } else {
        /* waitpid failed */
        return -1;
    }
}

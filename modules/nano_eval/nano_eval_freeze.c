#include "nano_eval_freeze.h"
#include "nano_eval_ipc.h"

#include <ctype.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static int copy_path(char *dst, size_t n, const char *src) {
    size_t len;
    if (!src || !dst || n == 0) {
        return -1;
    }
    len = strlen(src);
    if (len + 1 > n) {
        return -1;
    }
    memcpy(dst, src, len + 1);
    return 0;
}

static int find_tool(const char *env_name, const char *rel, char *out, size_t n) {
    const char *e = getenv(env_name);
    if (e && e[0] && access(e, X_OK) == 0) {
        return copy_path(out, n, e);
    }
    if (access(rel, X_OK) == 0) {
        return copy_path(out, n, rel);
    }
    return -1;
}

static int write_file(const char *path, const char *text) {
    FILE *f = fopen(path, "w");
    if (!f) {
        return -1;
    }
    if (fputs(text, f) == EOF) {
        fclose(f);
        return -1;
    }
    if (fclose(f) != 0) {
        return -1;
    }
    return 0;
}

static int run_timeout(char *const argv[], int timeout_sec, char *out, size_t outn, int *status_out) {
    int pipefd[2];
    pid_t pid;
    size_t got = 0;
    int status = 0;
    if (pipe(pipefd) != 0) {
        return -1;
    }
    pid = fork();
    if (pid < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        return -1;
    }
    if (pid == 0) {
        close(pipefd[0]);
        if (dup2(pipefd[1], STDOUT_FILENO) < 0) {
            _exit(127);
        }
        if (dup2(pipefd[1], STDERR_FILENO) < 0) {
            _exit(127);
        }
        close(pipefd[1]);
        alarm((unsigned)timeout_sec);
        execv(argv[0], argv);
        _exit(127);
    }
    close(pipefd[1]);
    while (got + 1 < outn) {
        ssize_t r = read(pipefd[0], out + got, outn - 1 - got);
        if (r < 0) {
            if (errno == EINTR) {
                continue;
            }
            break;
        }
        if (r == 0) {
            break;
        }
        got += (size_t)r;
    }
    out[got] = '\0';
    close(pipefd[0]);
    if (waitpid(pid, &status, 0) < 0) {
        return -1;
    }
    if (status_out) {
        *status_out = status;
    }
    if (WIFSIGNALED(status) && WTERMSIG(status) == SIGALRM) {
        return -2;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        return -3;
    }
    return 0;
}

static int fn_name_from_extract(const char *src, char *name, size_t n) {
    const char *p;
    size_t i = 0;
    if (!src || strncmp(src, "fn ", 3) != 0) {
        return -1;
    }
    p = src + 3;
    while (*p && (isalnum((unsigned char)*p) || *p == '_') && i + 1 < n) {
        name[i++] = *p++;
    }
    name[i] = '\0';
    return i > 0 ? 0 : -1;
}

int nano_eval_freeze_defun(const char *source, int64_t point, char *out, size_t outn) {
    char extract[65536];
    char module[70000];
    char name[64];
    char virt[512];
    char vm[512];
    char nano_path[256];
    char nvm_path[256];
    char captured[4096];
    int status = 0;
    int rc;
    pid_t self = getpid();
    char *argv_virt[8];
    char *argv_vm[4];
    if (!source || !out || outn == 0) {
        return -1;
    }
    out[0] = '\0';
    if (nano_eval_extract_defun(source, point, extract, sizeof(extract)) != 0) {
        snprintf(out, outn, "I cannot find a top-level fn at point.");
        return -1;
    }
    if (nano_eval_source_has_ed(extract)) {
        snprintf(out, outn, "I refuse to freeze a fn that calls ed_*. Frozen v1 is pure.");
        return -1;
    }
    if (fn_name_from_extract(extract, name, sizeof(name)) != 0) {
        snprintf(out, outn, "I cannot read the fn name.");
        return -1;
    }
    if (strstr(extract, "shadow ") == NULL) {
        snprintf(module, sizeof(module),
                 "%s\n\nshadow %s {\n    assert true\n}\n\nfn main() -> int {\n    return 0\n}\n\nshadow main {\n    assert true\n}\n",
                 extract, name);
    } else {
        snprintf(module, sizeof(module),
                 "%s\n\nfn main() -> int {\n    return 0\n}\n\nshadow main {\n    assert true\n}\n",
                 extract);
    }
    if (find_tool("NANO_VIRT", "bin/nano_virt", virt, sizeof(virt)) != 0) {
        snprintf(out, outn, "I cannot find nano_virt. Build it first.");
        return -1;
    }
    if (find_tool("NANO_VM", "bin/nano_vm", vm, sizeof(vm)) != 0) {
        snprintf(out, outn, "I cannot find nano_vm. Build it first.");
        return -1;
    }
    snprintf(nano_path, sizeof(nano_path), "/tmp/nano_freeze_%d.nano", (int)self);
    snprintf(nvm_path, sizeof(nvm_path), "/tmp/nano_freeze_%d.nvm", (int)self);
    if (write_file(nano_path, module) != 0) {
        snprintf(out, outn, "I cannot write the freeze source.");
        return -1;
    }
    argv_virt[0] = virt;
    argv_virt[1] = nano_path;
    argv_virt[2] = "--emit-nvm";
    argv_virt[3] = "-o";
    argv_virt[4] = nvm_path;
    argv_virt[5] = NULL;
    rc = run_timeout(argv_virt, 30, captured, sizeof(captured), &status);
    if (rc != 0) {
        unlink(nano_path);
        unlink(nvm_path);
        if (rc == -2) {
            snprintf(out, outn, "I timed out compiling the frozen fn.");
        } else {
            snprintf(out, outn, "I cannot compile the frozen fn: %s", captured);
        }
        return -1;
    }
    argv_vm[0] = vm;
    argv_vm[1] = nvm_path;
    argv_vm[2] = NULL;
    rc = run_timeout(argv_vm, 10, captured, sizeof(captured), &status);
    unlink(nano_path);
    unlink(nvm_path);
    if (rc == -2) {
        snprintf(out, outn, "I timed out running the frozen fn.");
        return -1;
    }
    if (rc != 0) {
        if (captured[0]) {
            snprintf(out, outn, "The frozen child failed: %s", captured);
        } else {
            snprintf(out, outn, "The frozen child crashed. I am still here.");
        }
        return -1;
    }
    if (captured[0]) {
        snprintf(out, outn, "%s", captured);
    } else {
        snprintf(out, outn, "froze %s", name);
    }
    return 0;
}

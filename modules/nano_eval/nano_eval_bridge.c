#define _POSIX_C_SOURCE 200809L
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "nano_eval.h"
#include "nano_eval_freeze.h"
#include "nano_eval_ipc.h"

#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#define MAX_SESS 8
#define CRASH_MSG "The walker crashed. I restarted it and kept your buffers."

typedef struct {
    int used;
    pid_t pid;
    int fd;
    uint64_t remote;
    char *buffer;
    int64_t point;
    char *result;
    char *error;
    int cmd_n;
    int cmd_kind[NE_MAX_CMD];
    char *cmd_arg[NE_MAX_CMD];
} ParentSess;

static ParentSess g_sess[MAX_SESS];
static char g_freeze_out[8192];
static int g_sigpipe_ok = 0;

static void ignore_sigpipe(void) {
    if (!g_sigpipe_ok) {
        signal(SIGPIPE, SIG_IGN);
        g_sigpipe_ok = 1;
    }
}

static void set_str(char **dst, const char *s) {
    free(*dst);
    *dst = strdup(s ? s : "");
}

static void free_cmds(ParentSess *s) {
    int i;
    for (i = 0; i < s->cmd_n; i++) {
        free(s->cmd_arg[i]);
        s->cmd_arg[i] = NULL;
        s->cmd_kind[i] = 0;
    }
    s->cmd_n = 0;
}

static ParentSess *sess_from(int64_t handle) {
    int slot;
    if (handle <= 0 || handle > MAX_SESS) {
        return NULL;
    }
    slot = (int)handle - 1;
    if (!g_sess[slot].used) {
        return NULL;
    }
    return &g_sess[slot];
}

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

static int find_worker(char *out, size_t n) {
    const char *e = getenv("NANO_EMACS_WORKER");
    if (e && e[0] && access(e, X_OK) == 0) {
        return copy_path(out, n, e);
    }
    if (access("bin/nano_emacs_worker", X_OK) == 0) {
        return copy_path(out, n, "bin/nano_emacs_worker");
    }
    if (access("./bin/nano_emacs_worker", X_OK) == 0) {
        return copy_path(out, n, "./bin/nano_emacs_worker");
    }
    return -1;
}

static void close_worker(ParentSess *s) {
    if (s->fd >= 0) {
        close(s->fd);
        s->fd = -1;
    }
    if (s->pid > 0) {
        kill(s->pid, SIGKILL);
        waitpid(s->pid, NULL, 0);
        s->pid = 0;
    }
}

static int spawn_worker(ParentSess *s) {
    int sv[2];
    pid_t pid;
    char path[1024];
    char *argv[2];
    NeBuf req;
    unsigned char *resp = NULL;
    uint32_t rn = 0;
    NeCur c;
    unsigned st = 0;
    uint64_t remote = 0;
    if (find_worker(path, sizeof(path)) != 0) {
        return -1;
    }
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) {
        return -1;
    }
    pid = fork();
    if (pid < 0) {
        close(sv[0]);
        close(sv[1]);
        return -1;
    }
    if (pid == 0) {
        close(sv[0]);
        if (dup2(sv[1], STDIN_FILENO) < 0 || dup2(sv[1], STDOUT_FILENO) < 0) {
            _exit(127);
        }
        if (sv[1] != STDIN_FILENO && sv[1] != STDOUT_FILENO) {
            close(sv[1]);
        }
        argv[0] = path;
        argv[1] = NULL;
        execv(path, argv);
        _exit(127);
    }
    close(sv[1]);
    s->pid = pid;
    s->fd = sv[0];
#ifdef FD_CLOEXEC
    fcntl(s->fd, F_SETFD, FD_CLOEXEC);
#endif
    ne_buf_init(&req);
    ne_buf_u8(&req, NE_OP_CREATE);
    if (ne_write_frame(s->fd, req.p, (uint32_t)req.n) != 0 ||
        ne_read_frame(s->fd, &resp, &rn) != 0) {
        ne_buf_free(&req);
        close_worker(s);
        return -1;
    }
    ne_buf_free(&req);
    c.p = resp;
    c.n = rn;
    c.off = 0;
    if (ne_cur_u8(&c, &st) != 0 || st != NE_OK || ne_cur_u64(&c, &remote) != 0) {
        free(resp);
        close_worker(s);
        return -1;
    }
    s->remote = remote;
    free(resp);
    return 0;
}

static int bind_remote(ParentSess *s) {
    NeBuf req;
    unsigned char *resp = NULL;
    uint32_t rn = 0;
    NeCur c;
    unsigned st = 0;
    ne_buf_init(&req);
    ne_buf_u8(&req, NE_OP_BIND);
    ne_buf_u64(&req, s->remote);
    ne_buf_i64(&req, s->point);
    ne_buf_str(&req, s->buffer ? s->buffer : "");
    if (ne_write_frame(s->fd, req.p, (uint32_t)req.n) != 0 ||
        ne_read_frame(s->fd, &resp, &rn) != 0) {
        ne_buf_free(&req);
        return -1;
    }
    ne_buf_free(&req);
    c.p = resp;
    c.n = rn;
    c.off = 0;
    if (ne_cur_u8(&c, &st) != 0 || st != NE_OK) {
        free(resp);
        return -1;
    }
    free(resp);
    return 0;
}

static int restart_keep_buffers(ParentSess *s) {
    close_worker(s);
    if (spawn_worker(s) != 0) {
        return -1;
    }
    if (bind_remote(s) != 0) {
        close_worker(s);
        return -1;
    }
    return 0;
}

static int parse_eval_reply(ParentSess *s, unsigned char *resp, uint32_t rn) {
    NeCur c;
    unsigned st = 0;
    char *result = NULL;
    char *err = NULL;
    char *buf = NULL;
    int64_t point = 0;
    uint32_t ncmd = 0;
    uint32_t i;
    c.p = resp;
    c.n = rn;
    c.off = 0;
    if (ne_cur_u8(&c, &st) != 0 || st != NE_OK) {
        return -1;
    }
    if (ne_cur_str(&c, &result) != 0 || ne_cur_str(&c, &err) != 0 ||
        ne_cur_i64(&c, &point) != 0 || ne_cur_str(&c, &buf) != 0 ||
        ne_cur_u32(&c, &ncmd) != 0) {
        free(result);
        free(err);
        free(buf);
        return -1;
    }
    set_str(&s->result, result);
    set_str(&s->error, err);
    set_str(&s->buffer, buf);
    s->point = point;
    free_cmds(s);
    if (ncmd > NE_MAX_CMD) {
        ncmd = NE_MAX_CMD;
    }
    for (i = 0; i < ncmd; i++) {
        int64_t kind = 0;
        char *arg = NULL;
        if (ne_cur_i64(&c, &kind) != 0 || ne_cur_str(&c, &arg) != 0) {
            free(result);
            free(err);
            free(buf);
            return -1;
        }
        s->cmd_kind[i] = (int)kind;
        s->cmd_arg[i] = arg;
        s->cmd_n++;
    }
    free(result);
    free(err);
    free(buf);
    return 0;
}

int64_t nano_eval_create(void) {
    int i;
    ignore_sigpipe();
    for (i = 0; i < MAX_SESS; i++) {
        if (!g_sess[i].used) {
            memset(&g_sess[i], 0, sizeof(g_sess[i]));
            g_sess[i].fd = -1;
            g_sess[i].used = 1;
            set_str(&g_sess[i].buffer, "");
            set_str(&g_sess[i].result, "");
            set_str(&g_sess[i].error, "");
            if (spawn_worker(&g_sess[i]) != 0) {
                g_sess[i].used = 0;
                return 0;
            }
            return (int64_t)(i + 1);
        }
    }
    return 0;
}

void nano_eval_destroy(int64_t session) {
    ParentSess *s = sess_from(session);
    NeBuf req;
    if (!s) {
        return;
    }
    if (s->fd >= 0) {
        ne_buf_init(&req);
        ne_buf_u8(&req, NE_OP_DESTROY);
        (void)ne_write_frame(s->fd, req.p, (uint32_t)req.n);
        ne_buf_free(&req);
    }
    close_worker(s);
    free_cmds(s);
    free(s->buffer);
    free(s->result);
    free(s->error);
    memset(s, 0, sizeof(*s));
    s->fd = -1;
}

void nano_eval_bind_buffer(int64_t session, const char *text, int64_t point) {
    ParentSess *s = sess_from(session);
    if (!s) {
        return;
    }
    set_str(&s->buffer, text ? text : "");
    s->point = point;
    if (bind_remote(s) != 0) {
        set_str(&s->error, CRASH_MSG);
        (void)restart_keep_buffers(s);
    }
}

const char *nano_eval_string(int64_t session, const char *source) {
    ParentSess *s = sess_from(session);
    NeBuf req;
    unsigned char *resp = NULL;
    uint32_t rn = 0;
    if (!s) {
        return "";
    }
    ne_buf_init(&req);
    ne_buf_u8(&req, NE_OP_EVAL);
    ne_buf_u64(&req, s->remote);
    ne_buf_str(&req, source ? source : "");
    if (ne_write_frame(s->fd, req.p, (uint32_t)req.n) != 0 ||
        ne_read_frame(s->fd, &resp, &rn) != 0) {
        ne_buf_free(&req);
        set_str(&s->result, "");
        set_str(&s->error, CRASH_MSG);
        (void)restart_keep_buffers(s);
        return s->result ? s->result : "";
    }
    ne_buf_free(&req);
    if (parse_eval_reply(s, resp, rn) != 0) {
        free(resp);
        set_str(&s->result, "");
        set_str(&s->error, CRASH_MSG);
        (void)restart_keep_buffers(s);
        return s->result ? s->result : "";
    }
    free(resp);
    return s->result ? s->result : "";
}

const char *nano_eval_error(int64_t session) {
    ParentSess *s = sess_from(session);
    if (!s || !s->error) {
        return "";
    }
    return s->error;
}

const char *nano_eval_buffer(int64_t session) {
    ParentSess *s = sess_from(session);
    if (!s || !s->buffer) {
        return "";
    }
    return s->buffer;
}

int64_t nano_eval_point(int64_t session) {
    ParentSess *s = sess_from(session);
    if (!s) {
        return 0;
    }
    return s->point;
}

int64_t nano_eval_cmd_count(int64_t session) {
    ParentSess *s = sess_from(session);
    if (!s) {
        return 0;
    }
    return s->cmd_n;
}

int64_t nano_eval_cmd_kind(int64_t session, int64_t index) {
    ParentSess *s = sess_from(session);
    if (!s || index < 0 || index >= s->cmd_n) {
        return NANO_EVAL_CMD_NONE;
    }
    return s->cmd_kind[index];
}

const char *nano_eval_cmd_arg(int64_t session, int64_t index) {
    ParentSess *s = sess_from(session);
    if (!s || index < 0 || index >= s->cmd_n || !s->cmd_arg[index]) {
        return "";
    }
    return s->cmd_arg[index];
}

void nano_eval_cmd_clear(int64_t session) {
    ParentSess *s = sess_from(session);
    NeBuf req;
    unsigned char *resp = NULL;
    uint32_t rn = 0;
    if (!s) {
        return;
    }
    free_cmds(s);
    ne_buf_init(&req);
    ne_buf_u8(&req, NE_OP_CLEAR);
    if (ne_write_frame(s->fd, req.p, (uint32_t)req.n) == 0) {
        (void)ne_read_frame(s->fd, &resp, &rn);
        free(resp);
    }
    ne_buf_free(&req);
}

int64_t nano_eval_worker_pid(int64_t session) {
    ParentSess *s = sess_from(session);
    if (!s) {
        return 0;
    }
    return (int64_t)s->pid;
}

const char *nano_eval_freeze(const char *source, int64_t point) {
    if (nano_eval_freeze_defun(source, point, g_freeze_out, sizeof(g_freeze_out)) != 0) {
        if (g_freeze_out[0] == '\0') {
            snprintf(g_freeze_out, sizeof(g_freeze_out), "I cannot freeze that fn.");
        }
    }
    return g_freeze_out;
}

int64_t ed_message(const char *text) {
    (void)text;
    return 0;
}

int64_t ed_insert(const char *text) {
    (void)text;
    return 0;
}

int64_t ed_buffer_string(void) {
    return 0;
}

int64_t ed_point(void) {
    return 0;
}

int64_t ed_goto_char(int64_t pos) {
    (void)pos;
    return 0;
}

int64_t ed_find_file(const char *path) {
    (void)path;
    return 0;
}

int64_t ed_save_buffer(void) {
    return 0;
}

int64_t ed_split_window(void) {
    return 0;
}

int64_t ed_other_window(void) {
    return 0;
}

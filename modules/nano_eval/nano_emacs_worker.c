#define _POSIX_C_SOURCE 200809L
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "nano_eval.h"
#include "nano_eval_ipc.h"

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    int64_t session = 0;
    signal(SIGPIPE, SIG_IGN);
    for (;;) {
        unsigned char *frame = NULL;
        uint32_t n = 0;
        NeCur c;
        unsigned op = 0;
        NeBuf out;
        if (ne_read_frame(STDIN_FILENO, &frame, &n) != 0) {
            break;
        }
        c.p = frame;
        c.n = n;
        c.off = 0;
        ne_buf_init(&out);
        if (ne_cur_u8(&c, &op) != 0) {
            ne_buf_free(&out);
            free(frame);
            break;
        }
        if (op == NE_OP_PING) {
            ne_buf_u8(&out, NE_OK);
        } else if (op == NE_OP_CREATE) {
            if (session != 0) {
                nano_eval_destroy(session);
            }
            session = nano_eval_create();
            if (session == 0) {
                ne_buf_u8(&out, NE_ERR);
                ne_buf_u64(&out, 0);
            } else {
                ne_buf_u8(&out, NE_OK);
                ne_buf_u64(&out, (uint64_t)session);
            }
        } else if (op == NE_OP_DESTROY) {
            if (session != 0) {
                nano_eval_destroy(session);
                session = 0;
            }
            ne_buf_u8(&out, NE_OK);
            (void)ne_write_frame(STDOUT_FILENO, out.p, (uint32_t)out.n);
            ne_buf_free(&out);
            free(frame);
            return 0;
        } else if (op == NE_OP_BIND) {
            uint64_t sid = 0;
            int64_t point = 0;
            char *text = NULL;
            if (ne_cur_u64(&c, &sid) != 0 || ne_cur_i64(&c, &point) != 0 ||
                ne_cur_str(&c, &text) != 0) {
                ne_buf_u8(&out, NE_ERR);
            } else {
                nano_eval_bind_buffer(session, text, point);
                ne_buf_u8(&out, NE_OK);
                free(text);
                (void)sid;
            }
        } else if (op == NE_OP_EVAL) {
            uint64_t sid = 0;
            char *src = NULL;
            if (ne_cur_u64(&c, &sid) != 0 || ne_cur_str(&c, &src) != 0) {
                ne_buf_u8(&out, NE_ERR);
            } else if (src && strcmp(src, "__hang__") == 0) {
                for (;;) {
                    pause();
                }
            } else {
                const char *result;
                const char *err;
                int64_t ncmd;
                int64_t i;
                (void)sid;
                result = nano_eval_string(session, src ? src : "");
                err = nano_eval_error(session);
                ncmd = nano_eval_cmd_count(session);
                ne_buf_u8(&out, NE_OK);
                ne_buf_str(&out, result ? result : "");
                ne_buf_str(&out, err ? err : "");
                ne_buf_i64(&out, nano_eval_point(session));
                ne_buf_str(&out, nano_eval_buffer(session));
                ne_buf_u32(&out, (uint32_t)ncmd);
                for (i = 0; i < ncmd && i < (int64_t)NE_MAX_CMD; i++) {
                    ne_buf_i64(&out, nano_eval_cmd_kind(session, i));
                    ne_buf_str(&out, nano_eval_cmd_arg(session, i));
                }
                free(src);
            }
        } else if (op == NE_OP_CLEAR) {
            if (session != 0) {
                nano_eval_cmd_clear(session);
            }
            ne_buf_u8(&out, NE_OK);
        } else {
            ne_buf_u8(&out, NE_ERR);
        }
        (void)ne_write_frame(STDOUT_FILENO, out.p, (uint32_t)out.n);
        ne_buf_free(&out);
        free(frame);
    }
    if (session != 0) {
        nano_eval_destroy(session);
    }
    return 0;
}

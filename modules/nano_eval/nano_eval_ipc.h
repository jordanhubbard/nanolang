#ifndef NANO_EVAL_IPC_H
#define NANO_EVAL_IPC_H

#include <stddef.h>
#include <stdint.h>

#define NE_OP_CREATE 1
#define NE_OP_DESTROY 2
#define NE_OP_BIND 3
#define NE_OP_EVAL 4
#define NE_OP_CLEAR 6
#define NE_OP_PING 7

#define NE_OK 0
#define NE_ERR 1

#define NE_MAX_FRAME (16u * 1024u * 1024u)
#define NE_MAX_CMD 64

int ne_write_all(int fd, const void *p, size_t n);
int ne_read_all(int fd, void *p, size_t n);
int ne_write_frame(int fd, const void *p, uint32_t n);
int ne_read_frame(int fd, unsigned char **out, uint32_t *n);

typedef struct {
    unsigned char *p;
    size_t n;
    size_t cap;
} NeBuf;

void ne_buf_init(NeBuf *b);
void ne_buf_free(NeBuf *b);
int ne_buf_u8(NeBuf *b, unsigned v);
int ne_buf_u32(NeBuf *b, uint32_t v);
int ne_buf_u64(NeBuf *b, uint64_t v);
int ne_buf_i64(NeBuf *b, int64_t v);
int ne_buf_str(NeBuf *b, const char *s);

typedef struct {
    const unsigned char *p;
    uint32_t n;
    uint32_t off;
} NeCur;

int ne_cur_u8(NeCur *c, unsigned *v);
int ne_cur_u32(NeCur *c, uint32_t *v);
int ne_cur_u64(NeCur *c, uint64_t *v);
int ne_cur_i64(NeCur *c, int64_t *v);
int ne_cur_str(NeCur *c, char **out);

int nano_eval_extract_defun(const char *text, int64_t point, char *out, size_t outn);
int nano_eval_source_has_ed(const char *text);

#endif

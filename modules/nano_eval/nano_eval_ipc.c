#include "nano_eval_ipc.h"

#include <ctype.h>
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int ne_write_all(int fd, const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    size_t off = 0;
    while (off < n) {
        ssize_t w = write(fd, b + off, n - off);
        if (w < 0) {
            if (errno == EINTR) {
                continue;
            }
            return -1;
        }
        if (w == 0) {
            return -1;
        }
        off += (size_t)w;
    }
    return 0;
}

int ne_read_all(int fd, void *p, size_t n) {
    unsigned char *b = (unsigned char *)p;
    size_t off = 0;
    while (off < n) {
        ssize_t r = read(fd, b + off, n - off);
        if (r < 0) {
            if (errno == EINTR) {
                continue;
            }
            return -1;
        }
        if (r == 0) {
            return -1;
        }
        off += (size_t)r;
    }
    return 0;
}

int ne_write_frame(int fd, const void *p, uint32_t n) {
    unsigned char hdr[4];
    hdr[0] = (unsigned char)(n & 0xffu);
    hdr[1] = (unsigned char)((n >> 8) & 0xffu);
    hdr[2] = (unsigned char)((n >> 16) & 0xffu);
    hdr[3] = (unsigned char)((n >> 24) & 0xffu);
    if (ne_write_all(fd, hdr, 4) != 0) {
        return -1;
    }
    if (n == 0) {
        return 0;
    }
    return ne_write_all(fd, p, n);
}

int ne_read_frame(int fd, unsigned char **out, uint32_t *n) {
    unsigned char hdr[4];
    uint32_t len;
    unsigned char *buf;
    if (ne_read_all(fd, hdr, 4) != 0) {
        return -1;
    }
    len = (uint32_t)hdr[0]
        | ((uint32_t)hdr[1] << 8)
        | ((uint32_t)hdr[2] << 16)
        | ((uint32_t)hdr[3] << 24);
    if (len > NE_MAX_FRAME) {
        return -1;
    }
    buf = (unsigned char *)malloc(len == 0 ? 1 : len);
    if (!buf) {
        return -1;
    }
    if (len > 0 && ne_read_all(fd, buf, len) != 0) {
        free(buf);
        return -1;
    }
    *out = buf;
    *n = len;
    return 0;
}

void ne_buf_init(NeBuf *b) {
    b->p = NULL;
    b->n = 0;
    b->cap = 0;
}

void ne_buf_free(NeBuf *b) {
    free(b->p);
    b->p = NULL;
    b->n = 0;
    b->cap = 0;
}

static int ne_buf_grow(NeBuf *b, size_t add) {
    size_t need = b->n + add;
    unsigned char *np;
    size_t cap;
    if (need <= b->cap) {
        return 0;
    }
    cap = b->cap == 0 ? 64 : b->cap;
    while (cap < need) {
        if (cap > (SIZE_MAX / 2)) {
            return -1;
        }
        cap *= 2;
    }
    np = (unsigned char *)realloc(b->p, cap);
    if (!np) {
        return -1;
    }
    b->p = np;
    b->cap = cap;
    return 0;
}

int ne_buf_u8(NeBuf *b, unsigned v) {
    if (ne_buf_grow(b, 1) != 0) {
        return -1;
    }
    b->p[b->n++] = (unsigned char)v;
    return 0;
}

int ne_buf_u32(NeBuf *b, uint32_t v) {
    if (ne_buf_grow(b, 4) != 0) {
        return -1;
    }
    b->p[b->n++] = (unsigned char)(v & 0xffu);
    b->p[b->n++] = (unsigned char)((v >> 8) & 0xffu);
    b->p[b->n++] = (unsigned char)((v >> 16) & 0xffu);
    b->p[b->n++] = (unsigned char)((v >> 24) & 0xffu);
    return 0;
}

int ne_buf_u64(NeBuf *b, uint64_t v) {
    int i;
    if (ne_buf_grow(b, 8) != 0) {
        return -1;
    }
    for (i = 0; i < 8; i++) {
        b->p[b->n++] = (unsigned char)((v >> (8 * i)) & 0xffu);
    }
    return 0;
}

int ne_buf_i64(NeBuf *b, int64_t v) {
    return ne_buf_u64(b, (uint64_t)v);
}

int ne_buf_str(NeBuf *b, const char *s) {
    uint32_t len;
    if (!s) {
        s = "";
    }
    len = (uint32_t)strlen(s);
    if (ne_buf_u32(b, len) != 0) {
        return -1;
    }
    if (len == 0) {
        return 0;
    }
    if (ne_buf_grow(b, len) != 0) {
        return -1;
    }
    memcpy(b->p + b->n, s, len);
    b->n += len;
    return 0;
}

int ne_cur_u8(NeCur *c, unsigned *v) {
    if (c->off + 1 > c->n) {
        return -1;
    }
    *v = c->p[c->off++];
    return 0;
}

int ne_cur_u32(NeCur *c, uint32_t *v) {
    uint32_t x;
    if (c->off + 4 > c->n) {
        return -1;
    }
    x = (uint32_t)c->p[c->off]
        | ((uint32_t)c->p[c->off + 1] << 8)
        | ((uint32_t)c->p[c->off + 2] << 16)
        | ((uint32_t)c->p[c->off + 3] << 24);
    c->off += 4;
    *v = x;
    return 0;
}

int ne_cur_u64(NeCur *c, uint64_t *v) {
    uint64_t x = 0;
    int i;
    if (c->off + 8 > c->n) {
        return -1;
    }
    for (i = 0; i < 8; i++) {
        x |= ((uint64_t)c->p[c->off + i]) << (8 * i);
    }
    c->off += 8;
    *v = x;
    return 0;
}

int ne_cur_i64(NeCur *c, int64_t *v) {
    uint64_t x;
    if (ne_cur_u64(c, &x) != 0) {
        return -1;
    }
    *v = (int64_t)x;
    return 0;
}

int ne_cur_str(NeCur *c, char **out) {
    uint32_t len;
    char *s;
    if (ne_cur_u32(c, &len) != 0) {
        return -1;
    }
    if (c->off + len > c->n) {
        return -1;
    }
    s = (char *)malloc(len + 1);
    if (!s) {
        return -1;
    }
    if (len > 0) {
        memcpy(s, c->p + c->off, len);
    }
    s[len] = '\0';
    c->off += len;
    *out = s;
    return 0;
}

int nano_eval_source_has_ed(const char *text) {
    const char *p;
    if (!text) {
        return 0;
    }
    p = text;
    while (*p) {
        if (p[0] == 'e' && p[1] == 'd' && p[2] == '_') {
            int before = (p == text) || !(isalnum((unsigned char)p[-1]) || p[-1] == '_');
            if (before) {
                return 1;
            }
        }
        p++;
    }
    return 0;
}

static int skip_ws_and_comments(const char *text, size_t n, size_t i) {
    while (i < n) {
        if (text[i] == ' ' || text[i] == '\t' || text[i] == '\n' || text[i] == '\r') {
            i++;
            continue;
        }
        if (text[i] == '/' && i + 1 < n && text[i + 1] == '/') {
            i += 2;
            while (i < n && text[i] != '\n') {
                i++;
            }
            continue;
        }
        break;
    }
    return (int)i;
}

static int find_matching_brace(const char *text, size_t n, size_t open_i) {
    int depth = 0;
    size_t i = open_i;
    int in_str = 0;
    if (open_i >= n || text[open_i] != '{') {
        return -1;
    }
    for (; i < n; i++) {
        char c = text[i];
        if (in_str) {
            if (c == '\\' && i + 1 < n) {
                i++;
                continue;
            }
            if (c == '"') {
                in_str = 0;
            }
            continue;
        }
        if (c == '"') {
            in_str = 1;
            continue;
        }
        if (c == '{') {
            depth++;
        } else if (c == '}') {
            depth--;
            if (depth == 0) {
                return (int)i;
            }
        }
    }
    return -1;
}

int nano_eval_extract_defun(const char *text, int64_t point, char *out, size_t outn) {
    size_t n;
    size_t i;
    size_t best_start = 0;
    size_t best_end = 0;
    int found = 0;
    size_t pt;
    if (!text || !out || outn == 0) {
        return -1;
    }
    n = strlen(text);
    if (point < 0) {
        pt = 0;
    } else if ((uint64_t)point > n) {
        pt = n;
    } else {
        pt = (size_t)point;
    }
    i = 0;
    while (i < n) {
        size_t fn_start;
        size_t name_i;
        size_t brace;
        int close;
        size_t end;
        i = (size_t)skip_ws_and_comments(text, n, i);
        if (i >= n) {
            break;
        }
        if (!(i + 3 <= n && text[i] == 'f' && text[i + 1] == 'n' &&
              (text[i + 2] == ' ' || text[i + 2] == '\t'))) {
            i++;
            continue;
        }
        fn_start = i;
        name_i = i + 3;
        while (name_i < n && (isalnum((unsigned char)text[name_i]) || text[name_i] == '_')) {
            name_i++;
        }
        brace = name_i;
        while (brace < n && text[brace] != '{') {
            if (text[brace] == '\n' && brace > name_i + 200) {
                break;
            }
            brace++;
        }
        if (brace >= n || text[brace] != '{') {
            i = name_i;
            continue;
        }
        close = find_matching_brace(text, n, brace);
        if (close < 0) {
            return -1;
        }
        end = (size_t)close + 1;
        {
            size_t k = (size_t)skip_ws_and_comments(text, n, end);
            if (k + 7 <= n && strncmp(text + k, "shadow", 6) == 0 &&
                (text[k + 6] == ' ' || text[k + 6] == '\t')) {
                size_t sb;
                int sc;
                sb = k;
                while (sb < n && text[sb] != '{') {
                    sb++;
                }
                if (sb < n && text[sb] == '{') {
                    sc = find_matching_brace(text, n, sb);
                    if (sc >= 0) {
                        end = (size_t)sc + 1;
                    }
                }
            }
        }
        if (pt >= fn_start && pt <= end) {
            best_start = fn_start;
            best_end = end;
            found = 1;
            break;
        }
        if (fn_start <= pt) {
            best_start = fn_start;
            best_end = end;
            found = 1;
        }
        i = end;
    }
    if (!found) {
        return -1;
    }
    {
        size_t len = best_end - best_start;
        if (len + 1 > outn) {
            return -1;
        }
        memcpy(out, text + best_start, len);
        out[len] = '\0';
    }
    return 0;
}

#define _GNU_SOURCE
#include "websocket_helpers.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>

#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
  typedef SOCKET sock_t;
  #define INVALID_SOCK INVALID_SOCKET
  #define CLOSE_SOCK(s) closesocket(s)
#else
  #include <unistd.h>
  #include <sys/socket.h>
  #include <netdb.h>
  #include <netinet/in.h>
  #include <arpa/inet.h>
  #include <poll.h>
  typedef int sock_t;
  #define INVALID_SOCK (-1)
  #define CLOSE_SOCK(s) close(s)
#endif

/* ============================================================
 * Base64 encoding (for Sec-WebSocket-Key)
 * ============================================================ */
static const char b64_table[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static void base64_encode(const unsigned char *in, size_t len, char *out) {
    size_t i = 0, j = 0;
    while (i < len) {
        unsigned int a = (i < len) ? (unsigned char)in[i++] : 0;
        unsigned int b = (i < len) ? (unsigned char)in[i++] : 0;
        unsigned int c = (i < len) ? (unsigned char)in[i++] : 0;
        unsigned int triple = (a << 16) | (b << 8) | c;
        out[j++] = b64_table[(triple >> 18) & 0x3F];
        out[j++] = b64_table[(triple >> 12) & 0x3F];
        out[j++] = (i > len + 1) ? '=' : b64_table[(triple >> 6) & 0x3F];
        out[j++] = (i > len)     ? '=' : b64_table[triple & 0x3F];
    }
    out[j] = '\0';
}

/* ============================================================
 * Context
 * ============================================================ */
#define MAX_RECV_BUF (1 << 20) /* 1 MB */
#define WS_MAGIC "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

typedef struct {
    sock_t fd;
    int    connected;
    char   last_error[256];
    char  *recv_buf;        /* heap, reused across calls */
    size_t recv_buf_size;
} WsCtx;

static WsCtx *ws_alloc(void) {
    WsCtx *c = calloc(1, sizeof(WsCtx));
    if (!c) return NULL;
    c->fd = INVALID_SOCK;
    c->recv_buf_size = 4096;
    c->recv_buf = malloc(c->recv_buf_size);
    if (!c->recv_buf) { free(c); return NULL; }
    c->recv_buf[0] = '\0';
    return c;
}

static void ws_set_error(WsCtx *c, const char *msg) {
    strncpy(c->last_error, msg ? msg : "", sizeof(c->last_error) - 1);
    c->last_error[sizeof(c->last_error) - 1] = '\0';
}

/* ============================================================
 * URL parsing
 * ============================================================ */
static int parse_ws_url(const char *url, char *host, size_t hlen,
                        char *port, size_t plen, char *path, size_t pathlen) {
    const char *p = url;
    if (strncmp(p, "ws://", 5) == 0)       p += 5;
    else if (strncmp(p, "wss://", 6) == 0) p += 6; /* wss not truly supported */
    else return -1;

    const char *slash = strchr(p, '/');
    const char *colon = strchr(p, ':');
    size_t host_end = slash ? (size_t)(slash - p) : strlen(p);
    if (colon && (!slash || colon < slash)) {
        size_t colon_pos = (size_t)(colon - p);
        if (colon_pos >= hlen) return -1;
        memcpy(host, p, colon_pos);
        host[colon_pos] = '\0';
        size_t plen2 = host_end - colon_pos - 1;
        if (plen2 >= plen) return -1;
        memcpy(port, colon + 1, plen2);
        port[plen2] = '\0';
    } else {
        if (host_end >= hlen) return -1;
        memcpy(host, p, host_end);
        host[host_end] = '\0';
        strncpy(port, "80", plen - 1);
    }
    strncpy(path, slash ? slash : "/", pathlen - 1);
    return 0;
}

/* ============================================================
 * TCP connect
 * ============================================================ */
static sock_t tcp_connect(const char *host, const char *port) {
    struct addrinfo hints = {0}, *res = NULL;
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, port, &hints, &res) != 0 || !res)
        return INVALID_SOCK;

    sock_t fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd == INVALID_SOCK) { freeaddrinfo(res); return INVALID_SOCK; }
    if (connect(fd, res->ai_addr, res->ai_addrlen) != 0) {
        CLOSE_SOCK(fd); freeaddrinfo(res); return INVALID_SOCK;
    }
    freeaddrinfo(res);
    return fd;
}

/* ============================================================
 * Send all bytes
 * ============================================================ */
static int send_all(sock_t fd, const char *buf, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = send(fd, buf + sent, len - sent, 0);
        if (n <= 0) return -1;
        sent += (size_t)n;
    }
    return 0;
}

/* ============================================================
 * WebSocket handshake
 * ============================================================ */
static int ws_handshake(sock_t fd, const char *host, const char *path) {
    /* Build a fixed nonce key (not cryptographically random, adequate for a client) */
    unsigned char nonce[16] = {
        0xde, 0xad, 0xbe, 0xef, 0xca, 0xfe, 0xba, 0xbe,
        0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef
    };
    char key_b64[32];
    base64_encode(nonce, 16, key_b64);

    char req[1024];
    int rlen = snprintf(req, sizeof(req),
        "GET %s HTTP/1.1\r\n"
        "Host: %s\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Key: %s\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "\r\n",
        path, host, key_b64);
    if (rlen < 0 || rlen >= (int)sizeof(req)) return -1;
    if (send_all(fd, req, (size_t)rlen) != 0) return -1;

    /* Read HTTP response until \r\n\r\n */
    char resp[4096];
    size_t total = 0;
    while (total < sizeof(resp) - 1) {
        ssize_t n = recv(fd, resp + total, 1, 0);
        if (n <= 0) return -1;
        total++;
        if (total >= 4 &&
            resp[total-4]=='\r' && resp[total-3]=='\n' &&
            resp[total-2]=='\r' && resp[total-1]=='\n') break;
    }
    resp[total] = '\0';

    /* Verify 101 Switching Protocols */
    if (!strstr(resp, "101")) return -1;
    return 0;
}

/* ============================================================
 * WebSocket frame reading
 * ============================================================ */
static int recv_exact(sock_t fd, unsigned char *buf, size_t n) {
    size_t got = 0;
    while (got < n) {
        ssize_t r = recv(fd, buf + got, n - got, 0);
        if (r <= 0) return -1;
        got += (size_t)r;
    }
    return 0;
}

/* Returns 0 on success, fills *out_data and *out_len.
 * *out_data is heap-allocated by this call; caller must free. */
static int ws_recv_frame(sock_t fd, char **out_data, size_t *out_len,
                         int timeout_ms) {
    *out_data = NULL;
    *out_len  = 0;

#ifndef _WIN32
    if (timeout_ms >= 0) {
        struct pollfd pfd = { fd, POLLIN, 0 };
        int r = poll(&pfd, 1, timeout_ms);
        if (r <= 0) return (r == 0) ? 1 : -1; /* 1 = timeout */
    }
#endif

    unsigned char hdr[2];
    if (recv_exact(fd, hdr, 2) != 0) return -1;

    int fin      = (hdr[0] >> 7) & 1;
    int opcode   = hdr[0] & 0x0F;
    int masked   = (hdr[1] >> 7) & 1;
    uint64_t payload_len = hdr[1] & 0x7F;

    (void)fin;

    if (payload_len == 126) {
        unsigned char ext[2];
        if (recv_exact(fd, ext, 2) != 0) return -1;
        payload_len = ((uint64_t)ext[0] << 8) | ext[1];
    } else if (payload_len == 127) {
        unsigned char ext[8];
        if (recv_exact(fd, ext, 8) != 0) return -1;
        payload_len = 0;
        for (int i = 0; i < 8; i++)
            payload_len = (payload_len << 8) | ext[i];
    }

    unsigned char mask_key[4] = {0};
    if (masked) {
        if (recv_exact(fd, mask_key, 4) != 0) return -1;
    }

    if (payload_len > MAX_RECV_BUF) return -1;

    char *data = malloc(payload_len + 1);
    if (!data) return -1;

    if (payload_len > 0) {
        if (recv_exact(fd, (unsigned char *)data, (size_t)payload_len) != 0) {
            free(data); return -1;
        }
        if (masked) {
            for (size_t i = 0; i < payload_len; i++)
                data[i] ^= mask_key[i % 4];
        }
    }
    data[payload_len] = '\0';

    /* Handle close (opcode 8) */
    if (opcode == 8) {
        free(data);
        return -2; /* closed */
    }
    /* Ping (opcode 9) → pong (opcode 10) */
    if (opcode == 9) {
        unsigned char pong[2] = { 0x8A, 0x00 };
        send_all(fd, (char *)pong, 2);
        free(data);
        *out_data = strdup("");
        *out_len  = 0;
        return 0;
    }

    *out_data = data;
    *out_len  = (size_t)payload_len;
    return 0;
}

/* ============================================================
 * WebSocket frame writing (client: must mask)
 * ============================================================ */
static int ws_send_text(sock_t fd, const char *msg) {
    size_t len = strlen(msg);

    /* Fixed mask (not secure but valid per spec) */
    unsigned char mask[4] = { 0x37, 0xfa, 0x21, 0x3d };

    size_t header_len;
    unsigned char header[10];
    header[0] = 0x81; /* FIN + text */
    if (len <= 125) {
        header[1] = (unsigned char)(0x80 | len);
        header_len = 2;
    } else if (len <= 65535) {
        header[1] = 0x80 | 126;
        header[2] = (unsigned char)((len >> 8) & 0xFF);
        header[3] = (unsigned char)(len & 0xFF);
        header_len = 4;
    } else {
        header[1] = 0x80 | 127;
        for (int i = 0; i < 8; i++)
            header[2 + i] = (unsigned char)((len >> (56 - 8*i)) & 0xFF);
        header_len = 10;
    }
    memcpy(header + header_len, mask, 4);
    header_len += 4;

    if (send_all(fd, (char *)header, header_len) != 0) return -1;

    /* Send masked payload in chunks */
    char chunk[4096];
    size_t sent = 0;
    while (sent < len) {
        size_t to_send = len - sent;
        if (to_send > sizeof(chunk)) to_send = sizeof(chunk);
        for (size_t i = 0; i < to_send; i++)
            chunk[i] = msg[sent + i] ^ mask[(sent + i) % 4];
        if (send_all(fd, chunk, to_send) != 0) return -1;
        sent += to_send;
    }
    return 0;
}

/* ============================================================
 * Public API
 * ============================================================ */

int64_t nl_ws_connect(const char *url) {
    char host[256], port[16], path[1024];
    if (!url || parse_ws_url(url, host, sizeof(host),
                             port, sizeof(port),
                             path, sizeof(path)) != 0)
        return 0;

    WsCtx *c = ws_alloc();
    if (!c) return 0;

    c->fd = tcp_connect(host, port);
    if (c->fd == INVALID_SOCK) {
        ws_set_error(c, "TCP connect failed");
        free(c->recv_buf); free(c);
        return 0;
    }

    if (ws_handshake(c->fd, host, path) != 0) {
        ws_set_error(c, "WebSocket handshake failed");
        CLOSE_SOCK(c->fd);
        free(c->recv_buf); free(c);
        return 0;
    }

    c->connected = 1;
    return (int64_t)(uintptr_t)c;
}

int64_t nl_ws_send(int64_t handle, const char *message) {
    WsCtx *c = (WsCtx *)(uintptr_t)handle;
    if (!c || !c->connected || c->fd == INVALID_SOCK) return -1;
    if (!message) message = "";
    if (ws_send_text(c->fd, message) != 0) {
        c->connected = 0;
        ws_set_error(c, "send failed");
        return -1;
    }
    return 0;
}

static const char *ws_recv_impl(WsCtx *c, int timeout_ms) {
    char   *data = NULL;
    size_t  len  = 0;
    int     rc   = ws_recv_frame(c->fd, &data, &len, timeout_ms);
    if (rc == 1) { /* timeout */
        c->recv_buf[0] = '\0';
        return c->recv_buf;
    }
    if (rc != 0) {
        c->connected = 0;
        c->recv_buf[0] = '\0';
        return c->recv_buf;
    }
    /* Copy into persistent buf */
    if (len + 1 > c->recv_buf_size) {
        char *nb = realloc(c->recv_buf, len + 1);
        if (!nb) { free(data); c->recv_buf[0] = '\0'; return c->recv_buf; }
        c->recv_buf = nb;
        c->recv_buf_size = len + 1;
    }
    memcpy(c->recv_buf, data, len + 1);
    free(data);
    return c->recv_buf;
}

const char *nl_ws_receive(int64_t handle) {
    WsCtx *c = (WsCtx *)(uintptr_t)handle;
    if (!c || !c->connected) return "";
    return ws_recv_impl(c, -1);
}

const char *nl_ws_receive_timeout(int64_t handle, int64_t timeout_ms) {
    WsCtx *c = (WsCtx *)(uintptr_t)handle;
    if (!c || !c->connected) return "";
    return ws_recv_impl(c, (int)timeout_ms);
}

int64_t nl_ws_close(int64_t handle) {
    WsCtx *c = (WsCtx *)(uintptr_t)handle;
    if (!c) return -1;
    if (c->fd != INVALID_SOCK) {
        /* Send WebSocket close frame */
        unsigned char close_frame[6] = { 0x88, 0x82, 0x00, 0x00, 0x00, 0x00 };
        send_all(c->fd, (char *)close_frame, 6);
        CLOSE_SOCK(c->fd);
        c->fd = INVALID_SOCK;
    }
    c->connected = 0;
    free(c->recv_buf);
    free(c);
    return 0;
}

int64_t nl_ws_is_connected(int64_t handle) {
    WsCtx *c = (WsCtx *)(uintptr_t)handle;
    return (c && c->connected) ? 1 : 0;
}

const char *nl_ws_last_error(int64_t handle) {
    WsCtx *c = (WsCtx *)(uintptr_t)handle;
    if (!c) return "Invalid handle";
    return c->last_error;
}

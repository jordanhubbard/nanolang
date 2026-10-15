/* I exercise counted host resolution and owned buffer I/O through real sockets,
 * with deterministic resolver and short-I/O controls in the included-source run. */
#include "../src/nsi_socket.h"
#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x); exit(1); } } while (0)

#ifdef NL_SOCKET_NETWORK_INSTRUMENT
static int resolver_fault, resolver_mode, send_fault, receive_fault;
static unsigned resolver_calls, resolver_frees, send_calls, receive_calls;
static size_t send_limit, receive_limit;
static struct addrinfo fake_answers[257];
static struct sockaddr_in fake_addresses[257];
static struct sockaddr_in6 fake_v6;
static int checked_getaddrinfo(const char *name, const char *port,
                              const struct addrinfo *hints, struct addrinfo **out) {
    resolver_calls++;
    CHECK(hints->ai_family == AF_UNSPEC && hints->ai_socktype == SOCK_STREAM &&
          hints->ai_protocol == IPPROTO_TCP && hints->ai_flags == AI_NUMERICSERV);
    if (resolver_fault) { errno = EACCES; return resolver_fault; }
    if (!resolver_mode) return getaddrinfo(name, port, hints, out);
    CHECK(!strcmp(name, "example.test") && !strcmp(port, "12345"));
    memset(fake_answers, 0, sizeof(fake_answers));
    memset(fake_addresses, 0, sizeof(fake_addresses));
    unsigned count = resolver_mode == 4 ? 17 : resolver_mode == 5 ? 257 : 2;
    for (unsigned i = 0; i < count; i++) {
        fake_addresses[i].sin_family = AF_INET;
        fake_addresses[i].sin_port = htons(12345);
        fake_addresses[i].sin_addr.s_addr = htonl(0x7f000001u + (resolver_mode == 4 ? i : 0));
        fake_answers[i].ai_family = AF_INET;
        fake_answers[i].ai_socktype = SOCK_STREAM;
        fake_answers[i].ai_protocol = IPPROTO_TCP;
        fake_answers[i].ai_addr = (struct sockaddr *)&fake_addresses[i];
        fake_answers[i].ai_addrlen = sizeof(fake_addresses[i]);
        fake_answers[i].ai_next = i + 1 < count ? &fake_answers[i + 1] : NULL;
    }
    if (resolver_mode == 2) fake_answers[1].ai_addrlen = 1;
    if (resolver_mode == 3) for (unsigned i = 0; i < count; i++) fake_answers[i].ai_socktype = SOCK_DGRAM;
    if (resolver_mode == 6) fake_answers[1].ai_addr = NULL;
    if (resolver_mode == 7) fake_addresses[1].sin_port = htons(80);
    if (resolver_mode == 8) fake_addresses[1].sin_family = AF_INET6;
    if (resolver_mode == 10 || resolver_mode == 11) {
        memset(&fake_v6, 0, sizeof(fake_v6));
        fake_v6.sin6_family = AF_INET6;
        fake_v6.sin6_port = htons(12345);
        fake_v6.sin6_scope_id = 42;
        fake_v6.sin6_addr.s6_addr[0] = 0xfe;
        fake_v6.sin6_addr.s6_addr[1] = 0x80;
        fake_v6.sin6_addr.s6_addr[15] = 1;
        fake_answers[1].ai_family = AF_INET6;
        fake_answers[1].ai_addr = (struct sockaddr *)&fake_v6;
        fake_answers[1].ai_addrlen = resolver_mode == 11 ? 1 : sizeof(fake_v6);
    }
    *out = resolver_mode == 9 ? NULL : fake_answers;
    return 0;
}
static void checked_freeaddrinfo(struct addrinfo *answers) {
    resolver_frees++;
    if (answers != fake_answers) freeaddrinfo(answers);
    else memset(fake_addresses, 0xA5, sizeof(fake_addresses));
    errno = ERANGE; /* I must not replace the original resolver error. */
}
static ssize_t checked_send(int fd, const void *bytes, size_t count, int flags) {
    send_calls++;
    if (send_fault) { errno = send_fault; return -1; }
    if (send_limit && count > send_limit) count = send_limit;
    return send(fd, bytes, count, flags);
}
static ssize_t checked_recv(int fd, void *bytes, size_t count, int flags) {
    receive_calls++;
    if (receive_fault) { errno = receive_fault; return -1; }
    if (receive_limit && count > receive_limit) count = receive_limit;
    return recv(fd, bytes, count, flags);
}
#define getaddrinfo checked_getaddrinfo
#define freeaddrinfo checked_freeaddrinfo
#define send checked_send
#define recv checked_recv
#include "../src/nsi_socket.c"
#undef getaddrinfo
#undef freeaddrinfo
#undef send
#undef recv
#endif

static void resolution_contract(NlSocketService *s) {
    NlSocketResolution out, before;
    memset(&out, 0xA5, sizeof(out));
    before = out;
    const char *invalid[] = {"", "127.1", "2130706433", "127.000.0.1", "::ffff:127.000.0.1", "::ffff:127.1", "a..b", ".host", "-host", "host-", "host-.test", "ws://host", "a/b", "host:80", "[::1]", "fe80::1%lo0", "bad host", "bad\nhost", "\xff"};
    for (size_t i = 0; i < sizeof(invalid) / sizeof(*invalid); i++) {
        NlSocketResolveResult invalid_result = nl_socket_resolve_tcp(s, invalid[i], strlen(invalid[i]), 80, true, &out);
        if (invalid_result.status != NL_SOCKET_ARGUMENT) fprintf(stderr, "invalid host [%s] returned %d\n", invalid[i], invalid_result.status);
        CHECK(invalid_result.status == NL_SOCKET_ARGUMENT);
        CHECK(!memcmp(&out, &before, sizeof(out)));
    }
    char embedded[] = "local\0host";
    CHECK(nl_socket_resolve_tcp(s, embedded, sizeof(embedded)-1, 80, true, &out).status == NL_SOCKET_ARGUMENT);
    char long_label[65]; memset(long_label, 'a', sizeof(long_label));
    CHECK(nl_socket_resolve_tcp(s, long_label, sizeof(long_label), 80, true, &out).status == NL_SOCKET_ARGUMENT);
    CHECK(nl_socket_resolve_tcp(s, "localhost", 9, 80, false, &out).status == NL_SOCKET_RIGHTS);
    CHECK(nl_socket_resolve_tcp(s, "localhost.", 10, 80, false, &out).status == NL_SOCKET_RIGHTS);
    CHECK(nl_socket_resolve_tcp(s, "localhost", 9, 0, true, &out).status == NL_SOCKET_ARGUMENT);
    CHECK(nl_socket_resolve_tcp(s, NULL, 9, 80, true, &out).status == NL_SOCKET_ARGUMENT);
    CHECK(nl_socket_resolve_tcp(s, "localhost", 254, 80, true, &out).status == NL_SOCKET_ARGUMENT);
    CHECK(nl_socket_resolve_tcp(s, "localhost", 9, 80, true, NULL).status == NL_SOCKET_ARGUMENT);
    CHECK(nl_socket_resolve_tcp(s, (char *)&out, 9, 80, true, &out).status == NL_SOCKET_ARGUMENT);
    CHECK(!memcmp(&out, &before, sizeof(out)));
#ifdef NL_SOCKET_NETWORK_INSTRUMENT
    CHECK(!resolver_calls);
#endif
    const char counted[] = {'1','2','7','.','0','.','0','.','1'};
    CHECK(nl_socket_resolve_tcp(s, counted, sizeof(counted), 65535, false, &out).status == NL_SOCKET_OK);
    CHECK(out.count == 1 && out.addresses[0].family == NL_SOCKET_IPV4 && out.addresses[0].port == 65535);
    CHECK(out.addresses[0].address[0] == 127 && out.addresses[0].address[3] == 1 && !out.addresses[0].scope_id);
    for (unsigned i = 4; i < 16; i++) CHECK(!out.addresses[0].address[i]);
    CHECK(nl_socket_resolve_tcp(s, "::1", 3, 80, false, &out).status == NL_SOCKET_OK);
    CHECK(out.count == 1 && out.addresses[0].family == NL_SOCKET_IPV6 && out.addresses[0].address[15] == 1);
#ifdef NL_SOCKET_NETWORK_INSTRUMENT
    CHECK(!resolver_calls);
    resolver_mode = 1;
    CHECK(nl_socket_resolve_tcp(s, "example.test", 12, 12345, true, &out).status == NL_SOCKET_OK);
    CHECK(resolver_calls == 1 && resolver_frees == 1 && out.count == 1);
    CHECK(out.addresses[0].address[0] == 127 && out.addresses[0].address[3] == 1);
    before = out;
    for (resolver_mode = 2; resolver_mode <= 9; resolver_mode++) {
        unsigned frees = resolver_frees;
        NlSocketResolveResult r = nl_socket_resolve_tcp(s, "example.test", 12, 12345, true, &out);
        CHECK(r.status == (resolver_mode == 4 ? NL_SOCKET_CAPACITY : resolver_mode == 5 ? NL_SOCKET_LIMIT : NL_SOCKET_IO));
        CHECK(resolver_frees == frees + (resolver_mode != 9));
        CHECK(!memcmp(&out, &before, sizeof(out)));
    }
    resolver_mode = 10;
    CHECK(nl_socket_resolve_tcp(s, "example.test", 12, 12345, true, &out).status == NL_SOCKET_OK);
    CHECK(out.count == 2 && out.addresses[1].family == NL_SOCKET_IPV6 && out.addresses[1].scope_id == 42);
    CHECK(out.addresses[1].address[0] == 0xfe && out.addresses[1].address[15] == 1);
    before = out;
    resolver_mode = 11;
    CHECK(nl_socket_resolve_tcp(s, "example.test", 12, 12345, true, &out).status == NL_SOCKET_IO);
    CHECK(!memcmp(&out, &before, sizeof(out)));
    resolver_mode = 0;
    int errors[] = {EAI_AGAIN, EAI_MEMORY, EAI_NONAME, EAI_SYSTEM};
    for (unsigned i = 0; i < 4; i++) {
        resolver_fault = errors[i];
        unsigned frees = resolver_frees;
        NlSocketResolveResult r = nl_socket_resolve_tcp(s, "example.test", 12, 12345, true, &out);
        CHECK(r.resolver_error == resolver_fault && r.host_errno == (resolver_fault == EAI_SYSTEM ? EACCES : 0));
        CHECK(r.status == (i == 0 ? NL_SOCKET_WOULD_BLOCK : i == 1 ? NL_SOCKET_MEMORY : NL_SOCKET_IO));
        CHECK(frees == resolver_frees && !memcmp(&out, &before, sizeof(out)));
    }
    resolver_fault = 0;
#endif
    CHECK(nl_socket_resolve_tcp(s, "localhost", 9, 12345, true, &out).status == NL_SOCKET_OK);
    CHECK(out.count > 0 && out.count <= NL_SOCKET_RESOLVE_MAX);
    for (size_t i = 0; i < out.count; i++) CHECK(out.addresses[i].port == 12345);
    puts("PASS counted IPv4/IPv6/hostname resolution and output preservation");
}

static void buffer_contract(NlSocketService *s) {
    NlSocketPair pair;
    CHECK(nl_socket_acquire_pair(s, NL_CAP_READ | NL_CAP_WRITE | NL_CAP_TRANSFER, NL_CAP_READ | NL_CAP_WRITE, &pair).status == NL_SOCKET_OK);
    unsigned char input[1024], output[1024];
    for (size_t i = 0; i < sizeof(input); i++) input[i] = (unsigned char)i;
    memset(output, 0xA5, sizeof(output));
    NlSocketResult r = nl_socket_receive(s, &pair.endpoints[1], output, sizeof(output));
    CHECK(r.status == NL_SOCKET_WOULD_BLOCK && output[0] == 0xA5);
    CHECK(nl_socket_send(s, &pair.endpoints[0], NULL, 0).status == NL_SOCKET_OK);
    r = nl_socket_receive(s, &pair.endpoints[1], NULL, 0);
    CHECK(r.status == NL_SOCKET_OK && !r.eof && !r.bytes);
    CHECK(nl_socket_send(s, &pair.endpoints[0], NULL, 1).status == NL_SOCKET_ARGUMENT);
    CHECK(nl_socket_receive(s, &pair.endpoints[1], output, NL_SOCKET_IO_MAX + 1).status == NL_SOCKET_ARGUMENT);
    CHECK(nl_socket_send(s, &pair.endpoints[0], input, NL_SOCKET_IO_MAX + 1).status == NL_SOCKET_ARGUMENT);
    CHECK(nl_socket_receive(s, &pair.endpoints[1], &pair.endpoints[1], 1).status == NL_SOCKET_ARGUMENT);
    CHECK(nl_socket_send(s, &pair.endpoints[0], &pair.endpoints[0], 1).status == NL_SOCKET_ARGUMENT);
#ifdef NL_SOCKET_NETWORK_INSTRUMENT
    CHECK(send_calls == 0 && receive_calls == 1);
    send_limit = 7; receive_limit = 5;
    send_fault = EINTR;
    CHECK(nl_socket_send(s, &pair.endpoints[0], input, sizeof(input)).status == NL_SOCKET_INTERRUPTED);
    send_fault = EPIPE;
    CHECK(nl_socket_send(s, &pair.endpoints[0], input, sizeof(input)).host_errno == EPIPE);
    send_fault = 0; receive_fault = EINTR;
    CHECK(nl_socket_receive(s, &pair.endpoints[1], output, sizeof(output)).status == NL_SOCKET_INTERRUPTED);
    CHECK(output[0] == 0xA5);
    receive_fault = 0;
#endif
    size_t sent = 0, received = 0;
    for (unsigned attempt = 0; attempt < 2048 && received < sizeof(input); attempt++) {
        if (sent < sizeof(input)) {
            r = nl_socket_send(s, &pair.endpoints[0], input + sent, sizeof(input) - sent);
            CHECK(r.status == NL_SOCKET_OK && r.bytes && r.bytes <= sizeof(input) - sent);
            sent += r.bytes;
        }
        r = nl_socket_receive(s, &pair.endpoints[1], output + received, sizeof(output) - received);
        CHECK(r.status == NL_SOCKET_OK && r.bytes && r.bytes <= sizeof(output) - received);
        received += r.bytes;
    }
    CHECK(sent == sizeof(input) && received == sizeof(input) && !memcmp(input, output, sizeof(input)));
    NlSocketToken old = pair.endpoints[0];
    CHECK(nl_socket_transfer(s, &pair.endpoints[0], &pair.endpoints[0]).status == NL_SOCKET_OK);
    CHECK(nl_socket_send(s, &old, input, 1).status == NL_SOCKET_TOKEN);
    CHECK(nl_socket_receive(s, &old, output, 1).status == NL_SOCKET_TOKEN);
    CHECK(nl_socket_consume_close(s, &pair.endpoints[0]).status == NL_SOCKET_OK);
    output[0] = 0xA5;
    r = nl_socket_receive(s, &pair.endpoints[1], output, sizeof(output));
    CHECK(r.status == NL_SOCKET_EOF && r.eof && !r.bytes && output[0] == 0xA5);
    CHECK(nl_socket_consume_close(s, &pair.endpoints[1]).status == NL_SOCKET_OK);
    CHECK(nl_socket_acquire_pair(s, NL_CAP_READ, NL_CAP_WRITE, &pair).status == NL_SOCKET_OK);
    CHECK(nl_socket_send(s, &pair.endpoints[0], input, 1).status == NL_SOCKET_RIGHTS);
    CHECK(nl_socket_receive(s, &pair.endpoints[1], output, 1).status == NL_SOCKET_RIGHTS);
    CHECK(nl_socket_consume_close(s, &pair.endpoints[0]).status == NL_SOCKET_OK);
    CHECK(nl_socket_consume_close(s, &pair.endpoints[1]).status == NL_SOCKET_OK);
    puts("PASS owned buffer partial I/O, NULs, rights, stale tokens and EOF");
}

#include "socket_tcp_fixture.h"
static void resolved_tcp(NlSocketService *s) {
    NlSocketAddress address;
    int listener = tcp_listener(NL_SOCKET_IPV4, &address, true);
    NlSocketResolution resolved;
    CHECK(nl_socket_resolve_tcp(s, "localhost", 9, address.port, true, &resolved).status == NL_SOCKET_OK);
    const NlSocketAddress *v4 = NULL;
    for (size_t i = 0; i < resolved.count; i++) if (resolved.addresses[i].family == NL_SOCKET_IPV4) v4 = &resolved.addresses[i];
    CHECK(v4);
    NlSocketToken token;
    CHECK(nl_socket_acquire_tcp(s, v4, NL_CAP_READ | NL_CAP_WRITE, &token).status == NL_SOCKET_OK);
    CHECK(tcp_wait_connected(s, &token).status == NL_SOCKET_OK);
    struct pollfd ready = {.fd = listener, .events = POLLIN};
    CHECK(poll(&ready, 1, 2000) == 1);
    int peer = accept(listener, NULL, NULL);
    CHECK(peer >= 0);
    CHECK(nl_socket_send(s, &token, "hello", 5).bytes == 5);
    ready = (struct pollfd){.fd = peer, .events = POLLIN};
    CHECK(poll(&ready, 1, 2000) == 1);
    char data[5];
    CHECK(recv(peer, data, sizeof(data), MSG_WAITALL) == 5 && !memcmp(data, "hello", 5));
    CHECK(nl_socket_consume_close(s, &token).status == NL_SOCKET_OK);
    CHECK(close(peer) == 0 && close(listener) == 0);
    puts("PASS localhost resolution to owned TCP and real buffer traffic");
}
int main(void) {
    NlSocketService *s = NULL;
    CHECK(nl_socket_service_create(&s).status == NL_SOCKET_OK);
    resolution_contract(s);
    buffer_contract(s);
    resolved_tcp(s);
    /* I also keep the existing independent numeric IPv4/IPv6 integration. */
    tcp_real_connections(NL_SOCKET_IPV4);
    tcp_real_connections(NL_SOCKET_IPV6);
    CHECK(nl_socket_service_dispose(s).status == NL_SOCKET_OK);
    NlSocketResolution out, before;
    memset(&out, 0xA5, sizeof(out)); before = out;
    CHECK(nl_socket_resolve_tcp(s, "localhost", 9, 80, true, &out).status == NL_SOCKET_DISPOSED);
    CHECK(nl_socket_resolve_tcp(NULL, "localhost", 9, 80, true, &out).status == NL_SOCKET_ARGUMENT);
    CHECK(!memcmp(&out, &before, sizeof(out)));
    CHECK(nl_socket_service_destroy(s).status == NL_SOCKET_OK);
    return 0;
}

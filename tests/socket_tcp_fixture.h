#ifndef NL_TEST_SOCKET_TCP_FIXTURE_H
#define NL_TEST_SOCKET_TCP_FIXTURE_H

/* I exercise the public C adapter against real TCP peers in both linkage modes. */
#include "../src/nsi_socket.h"
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

static int tcp_listener(NlSocketFamily family, NlSocketAddress *address, bool listen_now) {
    int domain = family == NL_SOCKET_IPV4 ? AF_INET : AF_INET6;
    int fd = socket(domain, SOCK_STREAM, IPPROTO_TCP);
    CHECK(fd >= 0);
    CHECK(fcntl(fd, F_SETFL, O_NONBLOCK) == 0);
    memset(address, 0, sizeof(*address));
    address->family = family;
    if (family == NL_SOCKET_IPV4) {
        struct sockaddr_in host = {.sin_family = AF_INET};
        host.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        CHECK(bind(fd, (struct sockaddr *)&host, sizeof(host)) == 0);
        socklen_t size = sizeof(host);
        CHECK(getsockname(fd, (struct sockaddr *)&host, &size) == 0);
        memcpy(address->address, &host.sin_addr, 4);
        address->port = ntohs(host.sin_port);
    } else {
        struct sockaddr_in6 host = {.sin6_family = AF_INET6, .sin6_addr = IN6ADDR_LOOPBACK_INIT};
        CHECK(bind(fd, (struct sockaddr *)&host, sizeof(host)) == 0);
        socklen_t size = sizeof(host);
        CHECK(getsockname(fd, (struct sockaddr *)&host, &size) == 0);
        memcpy(address->address, &host.sin6_addr, 16);
        address->port = ntohs(host.sin6_port);
    }
    if (listen_now) CHECK(listen(fd, 4) == 0);
    return fd;
}

static NlSocketResult tcp_wait_connected(NlSocketService *s, const NlSocketToken *token) {
    NlSocketResult r = {0};
    for (unsigned i = 0; i < 2000; i++) {
        r = nl_socket_finish_connect(s, token);
        if (r.status != NL_SOCKET_WOULD_BLOCK && r.status != NL_SOCKET_INTERRUPTED) return r;
        usleep(1000);
    }
    CHECK(!"I exceeded my bounded connection wait");
    return r;
}

static NlSocketResult tcp_wait_byte(NlSocketService *s, const NlSocketToken *token, uint8_t *byte) {
    NlSocketResult r = {0};
    for (unsigned i = 0; i < 2000; i++) {
        r = nl_socket_receive_byte(s, token, byte);
        if (r.status != NL_SOCKET_WOULD_BLOCK && r.status != NL_SOCKET_INTERRUPTED) return r;
        usleep(1000);
    }
    CHECK(!"I exceeded my bounded byte wait");
    return r;
}

static void tcp_real_connections(NlSocketFamily family) {
    NlSocketAddress address;
    int listener = tcp_listener(family, &address, true);
    NlSocketService *s = NULL;
    CHECK(nl_socket_service_create(&s).status == NL_SOCKET_OK);
    NlSocketToken token;
    NlSocketResult r = nl_socket_acquire_tcp(s, &address,
        NL_CAP_READ | NL_CAP_WRITE | NL_CAP_TRANSFER, &token);
    CHECK(r.status == NL_SOCKET_OK && !r.consumed && !r.close_attempts);
    NlSocketToken old = token;
    CHECK(nl_socket_transfer(s, &token, &token).status == NL_SOCKET_OK);
    CHECK(nl_socket_finish_connect(s, &old).status == NL_SOCKET_TOKEN);
    r = tcp_wait_connected(s, &token);
    CHECK(r.status == NL_SOCKET_OK && !r.connect_pending);
    CHECK(nl_socket_finish_connect(s, &token).status == NL_SOCKET_OK);
    struct pollfd wait = {.fd = listener, .events = POLLIN};
    CHECK(poll(&wait, 1, 2000) == 1 && (wait.revents & POLLIN));
    int peer = accept(listener, NULL, NULL);
    CHECK(peer >= 0);
    CHECK(fcntl(peer, F_SETFL, O_NONBLOCK) == 0);
    CHECK(close(listener) == 0);
    uint8_t byte = 77;
    r = nl_socket_receive_byte(s, &token, &byte);
    CHECK(r.status == NL_SOCKET_WOULD_BLOCK && byte == 77 && !r.eof);
    r = nl_socket_send_byte(s, &token, 0);
    CHECK(r.status == NL_SOCKET_OK && r.bytes == 1);
    wait = (struct pollfd){.fd = peer, .events = POLLIN};
    CHECK(poll(&wait, 1, 2000) == 1 && (wait.revents & POLLIN));
    CHECK(recv(peer, &byte, 1, 0) == 1 && byte == 0);
    byte = 255;
    CHECK(send(peer, &byte, 1, 0) == 1);
    byte = 77;
    r = tcp_wait_byte(s, &token, &byte);
    CHECK(r.status == NL_SOCKET_OK && r.bytes == 1 && byte == 255);
    CHECK(shutdown(peer, SHUT_WR) == 0);
    byte = 77;
    r = tcp_wait_byte(s, &token, &byte);
    CHECK(r.status == NL_SOCKET_EOF && r.eof && !r.bytes && byte == 0);
    r = nl_socket_consume_close(s, &token);
    CHECK(r.status == NL_SOCKET_OK && r.consumed && r.closed_count == 1);
    CHECK(nl_socket_consume_close(s, &token).status == NL_SOCKET_TOKEN);
    CHECK(close(peer) == 0);
    CHECK(nl_socket_service_destroy(s).status == NL_SOCKET_OK);

    /* I reserve a port through initiation, then close the non-listener. Darwin
     * may leave the SYN pending while that bound socket remains open. */
    listener = tcp_listener(family, &address, false);
    CHECK(nl_socket_service_create(&s).status == NL_SOCKET_OK);
    memset(&token, 0x35, sizeof(token));
    old = token;
    r = nl_socket_acquire_tcp(s, &address, NL_CAP_READ | NL_CAP_WRITE, &token);
    CHECK(close(listener) == 0);
    if (r.status == NL_SOCKET_OK) {
        r = tcp_wait_connected(s, &token);
        CHECK(r.status == NL_SOCKET_IO && r.host_errno == ECONNREFUSED);
        r = nl_socket_finish_connect(s, &token);
        CHECK(r.status == NL_SOCKET_IO && r.host_errno == ECONNREFUSED);
        byte = 77;
        r = nl_socket_receive_byte(s, &token, &byte);
        CHECK(r.status == NL_SOCKET_IO && r.host_errno == ECONNREFUSED && byte == 77);
        CHECK(nl_socket_send_byte(s, &token, 1).status == NL_SOCKET_IO);
        CHECK(nl_socket_consume_close(s, &token).status == NL_SOCKET_OK);
    } else {
        CHECK(r.status == NL_SOCKET_IO && r.host_errno == ECONNREFUSED);
        CHECK(!memcmp(&token, &old, sizeof(token)) && r.closed_count == 1);
    }
    CHECK(nl_socket_service_destroy(s).status == NL_SOCKET_OK);
    printf("PASS real IPv%d TCP transfer/NUL/byte/EOF/refusal/close\n", (int)family);
}
#endif

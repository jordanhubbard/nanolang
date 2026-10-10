#include "../src/nsi_socket_values.h"
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

static bool vs_empty(NlSocketValue v) {
    return !v.invocation && !v.generation && !v.slot;
}
static NlSocketValues *vs_create(void) {
    NlSocketValues *s = NULL;
    CHECK(nl_socket_values_create(&s) == NL_SOCKET_VALUE_OK && s);
    return s;
}
static void vs_clean(NlSocketValues *s) {
    NlSocketValuesFinish r = nl_socket_values_destroy(s, NL_SOCKET_VALUE_OK);
    CHECK(r.execution == NL_SOCKET_VALUE_OK && !r.cleanup_failures);
}
static int vs_listener(NlSocketFamily family, NlSocketAddress *address) {
    int fd = socket(family == NL_SOCKET_IPV4 ? AF_INET : AF_INET6, SOCK_STREAM, IPPROTO_TCP);
    CHECK(fd >= 0);
    CHECK(fcntl(fd, F_SETFL, O_NONBLOCK) == 0);
    *address = (NlSocketAddress){.family = family};
    if (family == NL_SOCKET_IPV4) {
        struct sockaddr_in host = {.sin_family = AF_INET, .sin_addr = {.s_addr = htonl(INADDR_LOOPBACK)}};
        CHECK(bind(fd, (struct sockaddr *)&host, sizeof(host)) == 0);
        socklen_t length = sizeof(host);
        CHECK(getsockname(fd, (struct sockaddr *)&host, &length) == 0);
        memcpy(address->address, &host.sin_addr, 4);
        address->port = ntohs(host.sin_port);
    } else {
        struct sockaddr_in6 host = {.sin6_family = AF_INET6, .sin6_addr = IN6ADDR_LOOPBACK_INIT};
        CHECK(bind(fd, (struct sockaddr *)&host, sizeof(host)) == 0);
        socklen_t length = sizeof(host);
        CHECK(getsockname(fd, (struct sockaddr *)&host, &length) == 0);
        memcpy(address->address, &host.sin6_addr, 16);
        address->port = ntohs(host.sin6_port);
    }
    CHECK(listen(fd, 4) == 0);
    return fd;
}
static NlSocketValue vs_connection(NlSocketValues *s, const NlSocketAddress *address) {
    NlSocketValue result = {0}, connection = {0};
    CHECK(nl_socket_values_connect(s, address, &result) == NL_SOCKET_VALUE_OK);
    CHECK(nl_socket_value_validate(s, &result, true) == NL_SOCKET_VALUE_OK);
    CHECK(nl_socket_value_validate(s, &result, false) == NL_SOCKET_VALUE_TYPE);
    NlSocketConnectView view;
    CHECK(nl_socket_connect_view(s, &result, &view) == NL_SOCKET_VALUE_OK && view.ok);
    NlSocketValueBorrow premature = {0};
    CHECK(nl_socket_value_borrow(s, &result, &premature) == NL_SOCKET_VALUE_TYPE);
    NlSocketResult wrong = {.bytes = 888};
    CHECK(nl_socket_connect_take_error(s, &result, &wrong) == NL_SOCKET_VALUE_TYPE && wrong.bytes == 888);
    NlSocketValue old = result;
    CHECK(nl_socket_connect_take_ok(s, &result, &connection) == NL_SOCKET_VALUE_OK && vs_empty(result));
    CHECK(nl_socket_value_drop(s, &old) == NL_SOCKET_VALUE_STALE);
    CHECK(nl_socket_value_validate(s, &connection, false) == NL_SOCKET_VALUE_OK);
    return connection;
}
static NlSocketScalarResult vs_wait(NlSocketValues *s, NlSocketValueBorrow *borrow, bool completion) {
    NlSocketScalarResult r = {0};
    for (unsigned i = 0; i < 2000; i++) {
        CHECK((completion ? nl_socket_value_finish_connect(s, borrow, &r) :
                            nl_socket_value_receive_byte(s, borrow, &r)) == NL_SOCKET_VALUE_OK);
        if (r.detail.status != NL_SOCKET_WOULD_BLOCK && r.detail.status != NL_SOCKET_INTERRUPTED) return r;
        usleep(1000);
    }
    CHECK(!"I exceeded my bounded value-operation wait");
    return r;
}
static void vs_real_lifetime(NlSocketFamily family) {
    NlSocketAddress address;
    int listener = vs_listener(family, &address);
    NlSocketValues *s = vs_create(), *other = vs_create();
    NlSocketValue connection = vs_connection(s, &address), moved = {0}, old = connection;
    CHECK(nl_socket_value_move(s, &connection, &moved) == NL_SOCKET_VALUE_OK && vs_empty(connection));
    CHECK(moved.generation > old.generation && moved.slot == old.slot);
    CHECK(nl_socket_value_drop(s, &old) == NL_SOCKET_VALUE_STALE);
    CHECK(nl_socket_value_drop(other, &moved) == NL_SOCKET_VALUE_STALE);
    CHECK(nl_socket_value_move(s, &moved, &moved) == NL_SOCKET_VALUE_ARGUMENT);
    NlSocketValueBorrow borrow = {0}, second = {0};
    CHECK(nl_socket_value_borrow(s, &moved, &borrow) == NL_SOCKET_VALUE_OK);
    CHECK(nl_socket_value_borrow_validate(s, &borrow) == NL_SOCKET_VALUE_OK);
    CHECK(nl_socket_value_borrow(s, &moved, &second) == NL_SOCKET_VALUE_BORROWED && !second.epoch);
    CHECK(nl_socket_value_move(s, &moved, &connection) == NL_SOCKET_VALUE_BORROWED && vs_empty(connection));
    CHECK(nl_socket_value_drop(s, &moved) == NL_SOCKET_VALUE_BORROWED);
    NlSocketScalarResult r = {.value = 777};
    CHECK(nl_socket_value_close(s, &moved, &r) == NL_SOCKET_VALUE_BORROWED && r.value == 777);
    uint64_t owners = 0, borrowed = 0;
    CHECK(nl_socket_values_live_slots(s, &owners, &borrowed) == NL_SOCKET_VALUE_OK);
    CHECK(owners == (UINT64_C(1) << moved.slot) && borrowed == owners);
    CHECK(nl_socket_values_live_slots(s, &owners, &owners) == NL_SOCKET_VALUE_ARGUMENT);
    r = vs_wait(s, &borrow, true);
    CHECK(r.ok && r.kind == NL_SOCKET_VALUE_CONNECT && !r.value && !r.eof);
    struct pollfd wait = {.fd = listener, .events = POLLIN};
    CHECK(poll(&wait, 1, 2000) == 1 && (wait.revents & POLLIN));
    int peer = accept(listener, NULL, NULL);
    CHECK(peer >= 0 && fcntl(peer, F_SETFL, O_NONBLOCK) == 0);
    CHECK(nl_socket_value_receive_byte(s, &borrow, &r) == NL_SOCKET_VALUE_OK);
    CHECK(!r.ok && !r.value && !r.eof && r.detail.status == NL_SOCKET_WOULD_BLOCK);
    CHECK(nl_socket_value_send_byte(s, &borrow, -1, &r) == NL_SOCKET_VALUE_OK && !r.ok && !r.value && !r.eof);
    CHECK(r.detail.status == NL_SOCKET_ARGUMENT && r.kind == NL_SOCKET_VALUE_SEND);
    CHECK(nl_socket_value_send_byte(s, &borrow, 256, &r) == NL_SOCKET_VALUE_OK && !r.ok);
    CHECK(nl_socket_value_send_byte(s, &borrow, 0, &r) == NL_SOCKET_VALUE_OK && r.ok && r.value == 1);
    wait = (struct pollfd){.fd = peer, .events = POLLIN};
    CHECK(poll(&wait, 1, 2000) == 1 && (wait.revents & POLLIN));
    uint8_t byte = 77;
    CHECK(recv(peer, &byte, 1, 0) == 1 && byte == 0);
    byte = 255;
    CHECK(send(peer, &byte, 1, 0) == 1);
    r = vs_wait(s, &borrow, false);
    CHECK(r.ok && r.kind == NL_SOCKET_VALUE_RECEIVE && r.value == 255 && !r.eof);
    CHECK(shutdown(peer, SHUT_WR) == 0);
    r = vs_wait(s, &borrow, false);
    CHECK(r.ok && r.eof && !r.value && r.detail.status == NL_SOCKET_EOF);
    NlSocketValueBorrow stale = borrow;
    CHECK(nl_socket_value_end_borrow(s, &borrow) == NL_SOCKET_VALUE_OK && !borrow.epoch);
    r.value = 777;
    CHECK(nl_socket_value_receive_byte(s, &stale, &r) == NL_SOCKET_VALUE_STALE && r.value == 777);
    CHECK(nl_socket_value_borrow(s, &moved, &borrow) == NL_SOCKET_VALUE_OK && borrow.epoch > stale.epoch);
    CHECK(nl_socket_value_end_borrow(s, &stale) == NL_SOCKET_VALUE_STALE);
    CHECK(nl_socket_value_end_borrow(s, &borrow) == NL_SOCKET_VALUE_OK);
    old = moved;
    CHECK(nl_socket_value_close(s, &moved, &r) == NL_SOCKET_VALUE_OK && vs_empty(moved));
    CHECK(r.ok && r.kind == NL_SOCKET_VALUE_CLOSE && !r.value && !r.eof && r.detail.consumed);
    CHECK(nl_socket_value_close(s, &old, &r) == NL_SOCKET_VALUE_STALE);
    CHECK(nl_socket_value_drop(s, &moved) == NL_SOCKET_VALUE_OK);
    CHECK(close(peer) == 0);
    NlSocketValue unhandled = {0};
    CHECK(nl_socket_values_connect(s, &address, &unhandled) == NL_SOCKET_VALUE_OK);
    CHECK(nl_socket_value_drop(s, &unhandled) == NL_SOCKET_VALUE_OK && vs_empty(unhandled));
    connection = vs_connection(s, &address);
    CHECK(nl_socket_value_borrow(s, &connection, &borrow) == NL_SOCKET_VALUE_OK);
    NlSocketValuesFinish report = nl_socket_values_finish(s, NL_SOCKET_VALUE_TYPE);
    CHECK(report.execution == NL_SOCKET_VALUE_TYPE && !report.cleanup_failures);
    CHECK(nl_socket_values_finish(s, NL_SOCKET_VALUE_OK).execution == NL_SOCKET_VALUE_TYPE);
    CHECK(nl_socket_value_end_borrow(s, &borrow) == NL_SOCKET_VALUE_DISPOSED);
    CHECK(nl_socket_value_drop(s, &connection) == NL_SOCKET_VALUE_DISPOSED);
    CHECK(nl_socket_values_destroy(s, NL_SOCKET_VALUE_OK).execution == NL_SOCKET_VALUE_TYPE);
    CHECK(nl_socket_value_drop(other, &connection) == NL_SOCKET_VALUE_STALE);
    vs_clean(other);
    CHECK(close(listener) == 0);
    printf("PASS IPv%d affine TCP Result/borrow/move/NUL/byte/EOF/cleanup\n", (int)family);
}
static void vs_errors_and_capacity(void) {
    NlSocketValues *s = vs_create();
    NlSocketAddress invalid = {.family = NL_SOCKET_IPV4};
    NlSocketValue error = {0}, moved = {0};
    CHECK(nl_socket_values_connect(s, NULL, &error) == NL_SOCKET_VALUE_ARGUMENT && vs_empty(error));
    CHECK(nl_socket_values_connect(s, &invalid, &error) == NL_SOCKET_VALUE_OK);
    NlSocketConnectView view;
    CHECK(nl_socket_connect_view(s, &error, &view) == NL_SOCKET_VALUE_OK && !view.ok && !view.pending);
    CHECK(view.error.status == NL_SOCKET_ARGUMENT);
    CHECK(nl_socket_connect_take_ok(s, &error, &moved) == NL_SOCKET_VALUE_TYPE && vs_empty(moved));
    CHECK(nl_socket_value_move(s, &error, &moved) == NL_SOCKET_VALUE_OK && vs_empty(error));
    NlSocketResult detail;
    NlSocketValue stale = moved;
    CHECK(nl_socket_connect_take_error(s, &moved, &detail) == NL_SOCKET_VALUE_OK && vs_empty(moved));
    CHECK(detail.status == NL_SOCKET_ARGUMENT);
    CHECK(nl_socket_value_drop(s, &stale) == NL_SOCKET_VALUE_STALE);
    NlSocketValue all[NL_SOCKET_VALUE_SLOTS] = {0};
    for (unsigned i = 0; i < NL_SOCKET_VALUE_SLOTS; i++)
        CHECK(nl_socket_values_connect(s, &invalid, &all[i]) == NL_SOCKET_VALUE_OK);
    CHECK(nl_socket_values_connect(s, &invalid, &error) == NL_SOCKET_VALUE_LIMIT && vs_empty(error));
    CHECK(nl_socket_value_move(s, &all[0], &all[1]) == NL_SOCKET_VALUE_ARGUMENT);
    CHECK(nl_socket_value_move(s, &all[0], &error) == NL_SOCKET_VALUE_OK && vs_empty(all[0]));
    CHECK(nl_socket_value_drop(s, &error) == NL_SOCKET_VALUE_OK);
    CHECK(nl_socket_values_connect(s, &invalid, &error) == NL_SOCKET_VALUE_OK);
    uint64_t owners = 0, borrowed = 17;
    CHECK(nl_socket_values_live_slots(s, &owners, &borrowed) == NL_SOCKET_VALUE_OK && owners == UINT64_MAX && !borrowed);
    NlSocketValuesFinish report;
    CHECK(nl_socket_values_report(s, &report) && !report.cleanup_failures);
    CHECK(nl_socket_values_destroy(s, NL_SOCKET_VALUE_STATUS_COUNT).execution == NL_SOCKET_VALUE_ARGUMENT);
    CHECK(nl_socket_value_drop(s, &error) == NL_SOCKET_VALUE_OK);
    vs_clean(s);
    size_t bound = 0, service = 0;
    CHECK(nl_socket_values_storage_bound(&bound) && nl_socket_service_storage_bound(&service) && bound > service);
    CHECK(!nl_socket_values_storage_bound(NULL) && !nl_socket_service_storage_bound(NULL));
}
static void vs_overlap_controls(void) {
    NlSocketValues *s = vs_create();
    NlSocketAddress invalid = {.family = NL_SOCKET_IPV4};
    union { NlSocketValue value; NlSocketConnectView view; NlSocketResult error;
            NlSocketValueBorrow borrow; NlSocketScalarResult scalar; NlSocketAddress address; } overlap;
    memset(&overlap, 0, sizeof(overlap));
    overlap.address = invalid;
    CHECK(nl_socket_values_connect(s, &overlap.address, &overlap.value) == NL_SOCKET_VALUE_ARGUMENT);
    overlap.value = (NlSocketValue){0};
    CHECK(nl_socket_values_connect(s, &invalid, &overlap.value) == NL_SOCKET_VALUE_OK);
    NlSocketValue saved = overlap.value;
    CHECK(nl_socket_connect_view(s, &overlap.value, &overlap.view) == NL_SOCKET_VALUE_ARGUMENT);
    CHECK(nl_socket_connect_take_error(s, &overlap.value, &overlap.error) == NL_SOCKET_VALUE_ARGUMENT);
    CHECK(nl_socket_value_borrow(s, &overlap.value, &overlap.borrow) == NL_SOCKET_VALUE_ARGUMENT);
    CHECK(nl_socket_value_close(s, &overlap.value, &overlap.scalar) == NL_SOCKET_VALUE_ARGUMENT);
    CHECK(!memcmp(&saved, &overlap.value, sizeof(saved)));
    CHECK(nl_socket_value_drop(s, &overlap.value) == NL_SOCKET_VALUE_OK);
    memset(&overlap, 0, sizeof(overlap));
    CHECK(nl_socket_value_finish_connect(s, &overlap.borrow, &overlap.scalar) == NL_SOCKET_VALUE_ARGUMENT);
    CHECK(nl_socket_value_send_byte(s, &overlap.borrow, 0, &overlap.scalar) == NL_SOCKET_VALUE_ARGUMENT);
    CHECK(nl_socket_value_receive_byte(s, &overlap.borrow, &overlap.scalar) == NL_SOCKET_VALUE_ARGUMENT);
    vs_clean(s);
}

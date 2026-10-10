/* I retain the real-host interception harness without executing its old main. */
#define main socket_adapter_fixture_main
#include "test_nsi_socket.c"
#undef main
#define calloc checked_calloc
#define free checked_free
#include "../src/nsi_socket_values.c"
#undef calloc
#undef free
#include "socket_values_cases.h"

static NlSocketValue vs_pending(NlSocketValues *s) {
    NlSocketAddress address = tcp_address();
    pending_connect = 1;
    return vs_connection(s, &address);
}
static void vs_faults(void) {
    for (long budget = 0; budget < 4; budget++) {
        allocation_budget = budget;
        NlSocketValues *s = NULL;
        NlSocketValueStatus status = nl_socket_values_create(&s);
        allocation_budget = -1;
        if (budget < 3) CHECK(status == NL_SOCKET_VALUE_MEMORY && !s);
        else { CHECK(status == NL_SOCKET_VALUE_OK); vs_clean(s); }
        empty();
    }
    uint64_t identity = sv_identity;
    sv_identity = UINT64_MAX;
    NlSocketValues *s = NULL;
    CHECK(nl_socket_values_create(&s) == NL_SOCKET_VALUE_LIMIT && !s);
    sv_identity = identity;
    identity = socket_context_counter; socket_context_counter = UINT64_MAX;
    CHECK(nl_socket_values_create(&s) == NL_SOCKET_VALUE_LIMIT && !s);
    socket_context_counter = identity; empty();
    s = vs_create();
    NlSocketAddress address = tcp_address();
    NlSocketValue result = {0};
    fail_connect = ECONNREFUSED;
    faults(EINTR, 0);
    CHECK(nl_socket_values_connect(s, &address, &result) == NL_SOCKET_VALUE_OK);
    fail_connect = 0; faults(0, 0);
    NlSocketConnectView view;
    CHECK(nl_socket_connect_view(s, &result, &view) == NL_SOCKET_VALUE_OK && !view.ok);
    CHECK(view.error.host_errno == ECONNREFUSED && view.error.cleanup_errno == EINTR && view.error.closure_unknown);
    CHECK(nl_socket_value_drop(s, &result) == NL_SOCKET_VALUE_OK);
    NlSocketValuesFinish report = nl_socket_values_destroy(s, NL_SOCKET_VALUE_MEMORY);
    CHECK(report.execution == NL_SOCKET_VALUE_MEMORY && report.cleanup_failures == 2);
    CHECK(report.first_cleanup.host_errno == ECONNREFUSED && report.next_cleanup.host_errno == EINTR);
    empty();
}
static void vs_pending_failures(void) {
    NlSocketValues *s = vs_create();
    NlSocketValue connection = vs_pending(s), moved = {0};
    NlSocketValueBorrow borrow = {0};
    CHECK(nl_socket_value_move(s, &connection, &moved) == NL_SOCKET_VALUE_OK);
    CHECK(nl_socket_value_borrow(s, &moved, &borrow) == NL_SOCKET_VALUE_OK);
    NlSocketScalarResult out = {.value = 888, .eof = true};
    unsigned calls = io_calls;
    CHECK(nl_socket_value_send_byte(s, &borrow, 7, &out) == NL_SOCKET_VALUE_OK);
    CHECK(!out.ok && !out.value && !out.eof && out.detail.connect_pending && out.detail.status == NL_SOCKET_WOULD_BLOCK);
    CHECK(nl_socket_value_receive_byte(s, &borrow, &out) == NL_SOCKET_VALUE_OK && !out.ok && !out.value && !out.eof);
    CHECK(io_calls == calls);
    poll_override = 0;
    CHECK(nl_socket_value_finish_connect(s, &borrow, &out) == NL_SOCKET_VALUE_OK && !out.ok && out.detail.connect_pending);
    fail_poll = EINTR;
    CHECK(nl_socket_value_finish_connect(s, &borrow, &out) == NL_SOCKET_VALUE_OK && !out.ok && out.detail.status == NL_SOCKET_INTERRUPTED);
    fail_poll = 0; poll_override = POLLOUT; socket_error = ECONNREFUSED;
    CHECK(nl_socket_value_finish_connect(s, &borrow, &out) == NL_SOCKET_VALUE_OK && !out.ok && out.detail.host_errno == ECONNREFUSED);
    socket_error = 0;
    CHECK(nl_socket_value_end_borrow(s, &borrow) == NL_SOCKET_VALUE_OK);
    CHECK(nl_socket_value_move(s, &moved, &connection) == NL_SOCKET_VALUE_OK);
    CHECK(nl_socket_value_borrow(s, &connection, &borrow) == NL_SOCKET_VALUE_OK);
    unsigned polls = poll_calls;
    CHECK(nl_socket_value_finish_connect(s, &borrow, &out) == NL_SOCKET_VALUE_OK && !out.ok && out.detail.host_errno == ECONNREFUSED);
    CHECK(nl_socket_value_send_byte(s, &borrow, 3, &out) == NL_SOCKET_VALUE_OK && !out.ok && out.detail.host_errno == ECONNREFUSED);
    CHECK(nl_socket_value_receive_byte(s, &borrow, &out) == NL_SOCKET_VALUE_OK && !out.ok && out.detail.host_errno == ECONNREFUSED);
    CHECK(io_calls == calls && poll_calls == polls);
    vs_clean(s);
    pending_connect = 0; poll_override = -1; empty();
}
static void vs_host_error_results(void) {
    NlSocketValues *s = vs_create();
    NlSocketValue connection = vs_pending(s);
    NlSocketValueBorrow borrow = {0};
    CHECK(nl_socket_value_borrow(s, &connection, &borrow) == NL_SOCKET_VALUE_OK);
    poll_override = POLLOUT;
    NlSocketScalarResult out;
    CHECK(nl_socket_value_finish_connect(s, &borrow, &out) == NL_SOCKET_VALUE_OK && out.ok);
    int errors[] = {EINTR, EAGAIN, EIO};
    NlSocketStatus expected[] = {NL_SOCKET_INTERRUPTED, NL_SOCKET_WOULD_BLOCK, NL_SOCKET_IO};
    for (unsigned i = 0; i < 3; i++) {
        fail_send = errors[i]; out.value = 888; out.eof = true;
        CHECK(nl_socket_value_send_byte(s, &borrow, 0, &out) == NL_SOCKET_VALUE_OK);
        CHECK(!out.ok && !out.value && !out.eof && out.detail.status == expected[i] && out.detail.host_errno == errors[i]);
        fail_send = 0; fail_recv = errors[i]; out.value = 888; out.eof = true;
        CHECK(nl_socket_value_receive_byte(s, &borrow, &out) == NL_SOCKET_VALUE_OK);
        CHECK(!out.ok && !out.value && !out.eof && out.detail.status == expected[i] && out.detail.host_errno == errors[i]);
        fail_recv = 0;
    }
    SvSlot *slot = &s->slots[connection.slot];
    NlCapSlot *cap = &s->service->caps->slots[slot->token.cap.slot];
    uint32_t rights = cap->rights;
    cap->rights = NL_CAP_READ;
    unsigned calls = io_calls;
    CHECK(nl_socket_value_send_byte(s, &borrow, 0, &out) == NL_SOCKET_VALUE_OK && !out.ok);
    CHECK(out.detail.status == NL_SOCKET_RIGHTS && io_calls == calls);
    cap->rights = rights;
    CHECK(nl_socket_value_end_borrow(s, &borrow) == NL_SOCKET_VALUE_OK);
    uint64_t secret = slot->token.cap.secret;
    slot->token.cap.secret ^= 1;
    out.value = 888;
    CHECK(nl_socket_value_close(s, &connection, &out) == NL_SOCKET_VALUE_STATE && out.value == 888);
    CHECK(!vs_empty(connection));
    slot->token.cap.secret = secret;
    NlSocketValuesFinish report = nl_socket_values_destroy(s, NL_SOCKET_VALUE_STATE);
    CHECK(report.execution == NL_SOCKET_VALUE_STATE && report.cleanup_failures == 1 && report.first_cleanup.status == NL_SOCKET_TOKEN);
    pending_connect = 0; poll_override = -1; empty();
}
static void vs_limits_and_cleanup(void) {
    NlSocketValues *s = vs_create();
    NlSocketValue connection = vs_pending(s), out = {0};
    s->slots[connection.slot].generation = connection.generation = UINT64_MAX;
    CHECK(nl_socket_value_move(s, &connection, &out) == NL_SOCKET_VALUE_LIMIT && vs_empty(out));
    CHECK(nl_socket_value_drop(s, &connection) == NL_SOCKET_VALUE_OK);
    connection = vs_pending(s); CHECK(connection.slot != 0);
    s->slots[connection.slot].borrow_epoch = UINT64_MAX;
    NlSocketValueBorrow borrow = {0};
    CHECK(nl_socket_value_borrow(s, &connection, &borrow) == NL_SOCKET_VALUE_LIMIT && !borrow.epoch);
    unsigned retired = connection.slot;
    CHECK(nl_socket_value_drop(s, &connection) == NL_SOCKET_VALUE_OK);
    connection = vs_pending(s); CHECK(connection.slot != retired);
    vs_clean(s); pending_connect = 0; empty();

    s = vs_create();
    NlSocketValue all[NL_SOCKET_VALUE_SLOTS];
    for (unsigned i = 0; i < NL_SOCKET_VALUE_SLOTS; i++) all[i] = vs_pending(s);
    unsigned opens = host_opens;
    NlSocketAddress address = tcp_address();
    CHECK(nl_socket_values_connect(s, &address, &out) == NL_SOCKET_VALUE_LIMIT && host_opens == opens);
    CHECK(nl_socket_value_move(s, &all[0], &out) == NL_SOCKET_VALUE_OK);
    CHECK(nl_socket_value_drop(s, &out) == NL_SOCKET_VALUE_OK);
    out = vs_pending(s);
    vs_clean(s); pending_connect = 0; empty();

    s = vs_create();
    NlSocketValue a = vs_pending(s), b = vs_pending(s), c = vs_pending(s);
    CHECK(nl_socket_value_borrow(s, &c, &borrow) == NL_SOCKET_VALUE_OK);
    faults(EIO, ENOSPC); close_faults[2] = EPIPE;
    NlSocketScalarResult scalar;
    CHECK(nl_socket_value_close(s, &a, &scalar) == NL_SOCKET_VALUE_OK && !scalar.ok && scalar.detail.consumed && vs_empty(a));
    CHECK(nl_socket_value_drop(s, &b) == NL_SOCKET_VALUE_OK);
    NlSocketValuesFinish report = nl_socket_values_finish(s, NL_SOCKET_VALUE_TYPE);
    CHECK(report.execution == NL_SOCKET_VALUE_TYPE && report.cleanup_failures == 4);
    CHECK(report.first_cleanup.host_errno == EIO && report.next_cleanup.host_errno == ENOSPC);
    CHECK(live_allocations == 1);
    unsigned closes = close_attempts;
    CHECK(nl_socket_values_finish(s, NL_SOCKET_VALUE_STATUS_COUNT).cleanup_failures == 4);
    CHECK(nl_socket_values_destroy(s, NL_SOCKET_VALUE_OK).execution == NL_SOCKET_VALUE_TYPE && close_attempts == closes);
    faults(0, 0); pending_connect = 0; empty();

    s = vs_create(); a = vs_pending(s);
    int fd = s->service->sockets[s->slots[a.slot].token.cap.slot].fd;
    faults(-EINTR, 0);
    CHECK(nl_socket_value_drop(s, &a) == NL_SOCKET_VALUE_OK && vs_empty(a));
    faults(0, 0);
    report = nl_socket_values_destroy(s, NL_SOCKET_VALUE_OK);
    CHECK(report.cleanup_failures == 2 && report.first_cleanup.closure_unknown);
    CHECK(fcntl(fd, F_GETFD) >= 0); real_close(fd, true);
    pending_connect = 0; empty();
}
int main(void) {
    for (unsigned i = 0; i < 128; i++) descriptors[i] = -1;
    vs_real_lifetime(NL_SOCKET_IPV4); empty();
    vs_real_lifetime(NL_SOCKET_IPV6); empty();
    vs_errors_and_capacity(); empty(); vs_overlap_controls(); empty();
    vs_faults(); vs_pending_failures(); vs_host_error_results(); vs_limits_and_cleanup();
    printf("PASS %u Socket value checks; tracked real opens=%u closes=%u; harness recovery closes=%u\n",
           checks, host_opens, host_closes, manual_closes);
    return 0;
}

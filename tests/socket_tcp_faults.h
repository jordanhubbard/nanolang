/* I include these controls after the instrumented adapter and pair fixtures. */
static unsigned tcp_live_caps(NlSocketService *s) {
    unsigned count = 0;
    for (unsigned i = 0; i < NL_CAP_PRIVATE_SLOTS; i++) count += s->caps->slots[i].used != 0;
    return count;
}

static NlSocketAddress tcp_address(void) {
    return (NlSocketAddress){.family = NL_SOCKET_IPV4, .address = {127, 0, 0, 1}, .port = 1};
}

static NlSocketToken tcp_pending(NlSocketService *s, uint32_t rights) {
    NlSocketAddress address = tcp_address();
    NlSocketToken token;
    pending_connect = 1;
    NlSocketResult r = nl_socket_acquire_tcp(s, &address, rights, &token);
    CHECK(r.status == NL_SOCKET_OK && r.connect_pending && !r.consumed);
    return token;
}

static void tcp_setup_faults(void) {
    puts("I check TCP setup rollback and unchanged output/adjacent owners.");
    unsigned steps = 4;
#ifdef __APPLE__
    steps = 5;
#endif
    for (unsigned step = 0; step <= steps + 3; step++) {
        NlSocketService *s = create();
        NlSocketPair live = pair(s, SOCKET_RIGHTS, SOCKET_RIGHTS);
        NlSocketToken out = live.endpoints[0], saved = out;
        NlSocketAddress address = tcp_address();
        config_calls = 0;
        fail_socket = step == 0 ? EMFILE : 0;
        fail_config = step > 0 && step <= steps ? (int)step : 0;
        int errors[] = {ECONNREFUSED, EINTR, EAGAIN};
        fail_connect = step > steps ? errors[step - steps - 1] : 0;
        int expected_errno = fail_socket ? fail_socket : (fail_config ? EACCES : fail_connect);
        unsigned opens = host_opens, closes = host_closes;
        unsigned calls = connect_calls;
        NlSocketResult r = nl_socket_acquire_tcp(s, &address, SOCKET_RIGHTS, &out);
        fail_socket = fail_config = fail_connect = 0;
        CHECK(r.status == (expected_errno == EINTR ? NL_SOCKET_INTERRUPTED :
                          expected_errno == EAGAIN ? NL_SOCKET_WOULD_BLOCK : NL_SOCKET_IO));
        CHECK(r.host_errno == expected_errno && !r.connect_pending && !r.consumed);
        CHECK(!memcmp(&out, &saved, sizeof(out)) && tcp_live_caps(s) == 2);
        CHECK(r.close_attempts == (step ? 1u : 0u) && r.closed_count == r.close_attempts);
        CHECK(host_opens == opens + (step ? 1u : 0u) && host_closes == closes + (step ? 1u : 0u));
        CHECK(connect_calls == calls + (step > steps ? 1u : 0u));
        send_receive(s, &live.endpoints[0], &live.endpoints[1], 37);
        NlSocketToken fresh = tcp_pending(s, SOCKET_RIGHTS);
        pending_connect = 0;
        CHECK(nl_socket_consume_close(s, &fresh).status == NL_SOCKET_OK);
        CHECK(nl_socket_service_destroy(s).status == NL_SOCKET_OK);
        empty();
    }
    for (unsigned mode = 0; mode < 2; mode++) {
        NlSocketService *s = create();
        NlSocketAddress address = tcp_address();
        NlSocketToken out = {0}, saved = out;
        fail_connect = ECONNREFUSED;
        faults(mode ? -EINTR : EINTR, 0);
        unsigned before = close_attempts;
        NlSocketResult r = nl_socket_acquire_tcp(s, &address, SOCKET_RIGHTS, &out);
        CHECK(r.status == NL_SOCKET_IO && r.host_errno == ECONNREFUSED);
        CHECK(r.cleanup_failed && r.cleanup_errno == EINTR && r.closure_unknown);
        CHECK(r.close_attempts == 1 && !r.closed_count && close_attempts == before + 1);
        CHECK(!memcmp(&out, &saved, sizeof(out)) && !tcp_live_caps(s));
        fail_connect = 0;
        faults(0, 0);
        r = nl_socket_service_destroy(s);
        CHECK(r.status == NL_SOCKET_IO && r.closure_unknown && !r.close_attempts);
        if (mode) {
            for (unsigned i = 0; i < 128; i++)
                if (descriptors[i] >= 0) real_close(descriptors[i], true);
        }
        empty();
    }
}

static void tcp_arguments_and_limits(void) {
    puts("I refuse invalid TCP addresses and exhausted authority before host calls.");
    NlSocketService *s = create();
    NlSocketToken out = {0}, saved = out;
    NlSocketAddress address = tcp_address();
    unsigned before = socket_calls;
    CHECK(nl_socket_acquire_tcp(s, NULL, SOCKET_RIGHTS, &out).status == NL_SOCKET_ARGUMENT);
    CHECK(nl_socket_acquire_tcp(s, &address, SOCKET_RIGHTS, NULL).status == NL_SOCKET_ARGUMENT);
    CHECK(nl_socket_acquire_tcp(s, &address, UINT32_MAX, &out).status == NL_SOCKET_ARGUMENT);
    address.family = (NlSocketFamily)5;
    CHECK(nl_socket_acquire_tcp(s, &address, 0, &out).status == NL_SOCKET_ARGUMENT);
    address = tcp_address(); address.port = 0;
    CHECK(nl_socket_acquire_tcp(s, &address, 0, &out).status == NL_SOCKET_ARGUMENT);
    address = tcp_address(); address.scope_id = 1;
    CHECK(nl_socket_acquire_tcp(s, &address, 0, &out).status == NL_SOCKET_ARGUMENT);
    address = tcp_address(); address.address[15] = 1;
    CHECK(nl_socket_acquire_tcp(s, &address, 0, &out).status == NL_SOCKET_ARGUMENT);
    union { NlSocketAddress address; NlSocketToken token; } overlap;
    overlap.address = tcp_address();
    CHECK(nl_socket_acquire_tcp(s, &overlap.address, 0, &overlap.token).status == NL_SOCKET_ARGUMENT);
    address = tcp_address();
    CHECK(!memcmp(&out, &saved, sizeof(out)) && socket_calls == before && !tcp_live_caps(s));
    for (unsigned i = 0; i < NL_CAP_PRIVATE_SLOTS / 2; i++) (void)pair(s, SOCKET_RIGHTS, SOCKET_RIGHTS);
    CHECK(nl_socket_acquire_tcp(s, &address, SOCKET_RIGHTS, &out).status == NL_SOCKET_CAPACITY);
    CHECK(!memcmp(&out, &saved, sizeof(out)) && socket_calls == before);
    CHECK(nl_socket_service_destroy(s).status == NL_SOCKET_OK); empty();
    s = create(); s->caps->next_generation = UINT32_MAX;
    CHECK(nl_socket_acquire_tcp(s, &address, SOCKET_RIGHTS, &out).status == NL_SOCKET_LIMIT);
    CHECK(!memcmp(&out, &saved, sizeof(out)) && socket_calls == before && !tcp_live_caps(s));
    CHECK(nl_socket_service_dispose(s).status == NL_SOCKET_OK);
    CHECK(nl_socket_acquire_tcp(s, &address, 0, &out).status == NL_SOCKET_DISPOSED);
    CHECK(nl_socket_service_destroy(s).status == NL_SOCKET_OK); empty();
}

static void tcp_publication_and_transfer_limits(void) {
    puts("I check immediate publication and full-capacity pending TCP transfer rollback.");
    NlSocketService *s = create();
    NlSocketAddress address = tcp_address();
    NlSocketToken token;
    pending_connect = -1;
    NlSocketResult r = nl_socket_acquire_tcp(s, &address, SOCKET_RIGHTS, &token);
    CHECK(r.status == NL_SOCKET_OK && !r.connect_pending);
    unsigned polls = poll_calls, reads = get_error_calls;
    CHECK(nl_socket_finish_connect(s, &token).status == NL_SOCKET_OK);
    CHECK(poll_calls == polls && get_error_calls == reads);
    CHECK(nl_socket_consume_close(s, &token).status == NL_SOCKET_OK);
    pending_connect = 0;
    NlSocketToken all[NL_CAP_PRIVATE_SLOTS];
    for (unsigned i = 0; i < NL_CAP_PRIVATE_SLOTS; i++) all[i] = tcp_pending(s, SOCKET_RIGHTS);
    NlSocketToken saved = all[0];
    CHECK(nl_socket_transfer(s, &all[0], &all[0]).status == NL_SOCKET_CAPACITY);
    CHECK(!memcmp(&all[0], &saved, sizeof(saved)));
    poll_override = 0;
    CHECK(nl_socket_finish_connect(s, &all[0]).status == NL_SOCKET_WOULD_BLOCK);
    CHECK(nl_socket_consume_close(s, &all[1]).status == NL_SOCKET_OK);
    CHECK(nl_socket_transfer(s, &all[0], &all[0]).status == NL_SOCKET_OK);
    CHECK(nl_socket_finish_connect(s, &saved).status == NL_SOCKET_TOKEN);
    CHECK(nl_socket_finish_connect(s, &all[0]).status == NL_SOCKET_WOULD_BLOCK);
    saved = all[0]; s->caps->next_generation = UINT32_MAX;
    CHECK(nl_socket_transfer(s, &all[0], &all[0]).status == NL_SOCKET_LIMIT);
    CHECK(!memcmp(&all[0], &saved, sizeof(saved)));
    r = nl_socket_service_dispose(s);
    CHECK(r.status == NL_SOCKET_OK && r.closed_count == NL_CAP_PRIVATE_SLOTS - 1);
    CHECK(nl_socket_finish_connect(s, &all[0]).status == NL_SOCKET_DISPOSED);
    CHECK(nl_socket_service_destroy(s).status == NL_SOCKET_OK);
    pending_connect = 0; poll_override = -1; empty();
}

static void tcp_pending_and_completion(void) {
    puts("I preserve pending/failed state through transfer and never reread a consumed error.");
    NlSocketService *s = create(), *other = create();
    NlSocketToken token = tcp_pending(s, SOCKET_RIGHTS), old = token;
    uint8_t byte = 93;
    unsigned calls = io_calls, polls = poll_calls, reads = get_error_calls;
    CHECK(nl_socket_send_byte(s, &token, 3).status == NL_SOCKET_WOULD_BLOCK);
    CHECK(nl_socket_receive_byte(s, &token, &byte).status == NL_SOCKET_WOULD_BLOCK && byte == 93);
    CHECK(io_calls == calls && poll_calls == polls);
    CHECK(nl_socket_finish_connect(other, &token).status == NL_SOCKET_TOKEN);
    NlCapSlot *slot = &s->caps->slots[token.cap.slot];
    char saved = slot->service_id[0]; slot->service_id[0] = 'X';
    CHECK(nl_socket_finish_connect(s, &token).status == NL_SOCKET_TOKEN);
    slot->service_id[0] = saved;
    CHECK(poll_calls == polls);
    CHECK(nl_socket_transfer(s, &token, &token).status == NL_SOCKET_OK);
    CHECK(nl_socket_finish_connect(s, &old).status == NL_SOCKET_TOKEN);
    poll_override = 0;
    NlSocketResult r = nl_socket_finish_connect(s, &token);
    CHECK(r.status == NL_SOCKET_WOULD_BLOCK && r.connect_pending && get_error_calls == reads);
    poll_override = POLLIN;
    CHECK(nl_socket_finish_connect(s, &token).status == NL_SOCKET_WOULD_BLOCK && get_error_calls == reads);
    int transient[] = {EINTR, EAGAIN};
    for (unsigned i = 0; i < 2; i++) {
        fail_poll = transient[i];
        r = nl_socket_finish_connect(s, &token);
        CHECK(r.host_errno == transient[i] && r.connect_pending);
        CHECK(r.status == (i ? NL_SOCKET_WOULD_BLOCK : NL_SOCKET_INTERRUPTED));
        CHECK(get_error_calls == reads);
        fail_poll = 0; poll_override = POLLOUT; fail_get_error = transient[i];
        r = nl_socket_finish_connect(s, &token);
        CHECK(r.host_errno == transient[i] && r.connect_pending);
        CHECK(r.status == (i ? NL_SOCKET_WOULD_BLOCK : NL_SOCKET_INTERRUPTED));
        fail_get_error = 0; reads++;
    }
    socket_error = 0;
    r = nl_socket_finish_connect(s, &token);
    CHECK(r.status == NL_SOCKET_OK && !r.connect_pending);
    polls = poll_calls; reads = get_error_calls;
    CHECK(nl_socket_finish_connect(s, &token).status == NL_SOCKET_OK);
    CHECK(poll_calls == polls && get_error_calls == reads);
    /* This ready path uses an injected completion, so I make no traffic claim. */
    CHECK(nl_socket_consume_close(s, &token).status == NL_SOCKET_OK);
    CHECK(nl_socket_service_destroy(other).status == NL_SOCKET_OK);
    CHECK(nl_socket_service_destroy(s).status == NL_SOCKET_OK);
    pending_connect = 0; poll_override = -1; empty();

    for (unsigned mode = 0; mode < 8; mode++) {
        s = create(); token = tcp_pending(s, SOCKET_RIGHTS);
        poll_override = POLLOUT;
        int expected = EIO;
        if (mode == 0) fail_poll = EIO;
        if (mode == 1) fail_get_error = EIO;
        if (mode == 2) { poll_override = POLLNVAL; expected = EBADF; }
        if (mode == 3) short_error = 1;
        if (mode == 4) { socket_error = ECONNREFUSED; expected = ECONNREFUSED; }
        if (mode == 5) { socket_error = EINTR; expected = EINTR; }
        if (mode == 6) { poll_override = POLLHUP; expected = ECONNABORTED; }
        if (mode == 7) { poll_override = POLLOUT | POLLERR; expected = ECONNABORTED; }
        r = nl_socket_finish_connect(s, &token);
        CHECK(r.status == NL_SOCKET_IO && r.host_errno == expected && !r.connect_pending);
        fail_poll = fail_get_error = short_error = socket_error = 0;
        polls = poll_calls; reads = get_error_calls; calls = io_calls;
        old = token;
        CHECK(nl_socket_transfer(s, &token, &token).status == NL_SOCKET_OK);
        CHECK(nl_socket_finish_connect(s, &old).status == NL_SOCKET_TOKEN);
        r = nl_socket_finish_connect(s, &token);
        CHECK(r.status == NL_SOCKET_IO && r.host_errno == expected);
        r = nl_socket_send_byte(s, &token, 3);
        CHECK(r.status == NL_SOCKET_IO && r.host_errno == expected);
        r = nl_socket_receive_byte(s, &token, &byte);
        CHECK(r.status == NL_SOCKET_IO && r.host_errno == expected && byte == 93);
        CHECK(poll_calls == polls && get_error_calls == reads && io_calls == calls);
        r = nl_socket_service_destroy(s);
        CHECK(r.status == NL_SOCKET_OK && r.close_attempts == 1 && r.closed_count == 1);
        pending_connect = 0; poll_override = -1; empty();
    }
    s = create(); token = tcp_pending(s, 0); calls = io_calls; polls = poll_calls;
    CHECK(nl_socket_send_byte(s, &token, 1).status == NL_SOCKET_RIGHTS);
    CHECK(nl_socket_receive_byte(s, &token, &byte).status == NL_SOCKET_RIGHTS && byte == 93);
    CHECK(nl_socket_transfer(s, &token, &token).status == NL_SOCKET_RIGHTS);
    CHECK(io_calls == calls && poll_calls == polls);
    poll_override = 0;
    CHECK(nl_socket_finish_connect(s, &token).status == NL_SOCKET_WOULD_BLOCK);
    CHECK(nl_socket_consume_close(s, &token).status == NL_SOCKET_OK);
    CHECK(nl_socket_service_destroy(s).status == NL_SOCKET_OK);
    pending_connect = 0; poll_override = -1; empty();
}

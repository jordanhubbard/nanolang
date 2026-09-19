#include "nsi_socket.h"
#include "nsi_cap_private.h"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#if !defined(__linux__) && !defined(__APPLE__)
#error "I require the reviewed Linux or Darwin local-socket policy"
#endif

#define SOCKET_TYPE "nsi:nanolang/net#Socket"
#define SOCKET_SERVICE "nsi:nanolang/net/local-socket"
#define SOCKET_RIGHTS (NL_CAP_READ | NL_CAP_WRITE | NL_CAP_TRANSFER)

typedef struct { int fd; bool live; NlCap token; } SocketEntry;
struct NlSocketService {
    NlCapTable *caps;
    uint64_t identity;
    bool disposed, closure_unknown;
    int first_unknown_errno;
    SocketEntry sockets[NL_CAP_PRIVATE_SLOTS];
};
static uint64_t socket_context_counter;

static NlSocketResult socket_result(NlSocketStatus status) {
    NlSocketResult result = {0};
    result.status = status;
    return result;
}
static NlSocketStatus socket_cap_status(int status) {
    switch (status) {
    case NL_CAP_OK: return NL_SOCKET_OK;
    case NL_CAP_ERR_RIGHTS: case NL_CAP_ERR_TRANSFER: return NL_SOCKET_RIGHTS;
    case NL_CAP_ERR_FULL: return NL_SOCKET_CAPACITY;
    case NL_CAP_PRIVATE_ERR_GENERATION: return NL_SOCKET_LIMIT;
    default: return NL_SOCKET_TOKEN;
    }
}
static NlSocketStatus socket_context(const NlSocketService *service) {
    if (!service) return NL_SOCKET_ARGUMENT;
    return service->disposed ? NL_SOCKET_DISPOSED : NL_SOCKET_OK;
}
static bool socket_same_cap(NlCap a, NlCap b) {
    return a.slot == b.slot && a.generation == b.generation && a.secret == b.secret;
}
static NlSocketStatus socket_resolve(NlSocketService *service,
                                     const NlSocketToken *token, uint32_t rights,
                                     SocketEntry **out) {
    NlSocketStatus status = socket_context(service);
    if (status != NL_SOCKET_OK) return status;
    if (!token) return NL_SOCKET_ARGUMENT;
    if (token->context_id != service->identity || token->cap.slot >= NL_CAP_PRIVATE_SLOTS)
        return NL_SOCKET_TOKEN;
    int rc = nl_cap_check(service->caps, &token->cap, rights);
    if (rc != NL_CAP_OK) return socket_cap_status(rc);
    SocketEntry *entry = &service->sockets[token->cap.slot];
    if (!entry->live || !socket_same_cap(entry->token, token->cap) ||
        strcmp(nl_cap_type_id(service->caps, &token->cap), SOCKET_TYPE) ||
        strcmp(nl_cap_service_id(service->caps, &token->cap), SOCKET_SERVICE))
        return NL_SOCKET_TOKEN;
    *out = entry;
    return NL_SOCKET_OK;
}
/* I compare unsigned addresses without pointer ordering or end-address addition.
 * Valid C objects cannot wrap the address space; subtraction is ordered first. */
static bool socket_overlap(const void *a, size_t na, const void *b, size_t nb) {
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x <= y ? y - x < na : x - y < nb;
}
static void socket_cleanup_error(NlSocketResult *result, int saved) {
    if (!result->cleanup_failed) result->cleanup_errno = saved;
    result->cleanup_failed = true;
}
static void socket_close_once(NlSocketService *service, int fd,
                               NlSocketResult *result, bool cleanup) {
    result->close_attempts++;
    errno = 0;
    int rc = close(fd), saved = errno;
    if (rc == 0) {
        result->closed_count++;
        return;
    }
    result->closure_unknown = true;
    if (!service->closure_unknown) service->first_unknown_errno = saved;
    service->closure_unknown = true;
    if (cleanup || result->status != NL_SOCKET_OK) {
        socket_cleanup_error(result, saved);
    } else {
        result->status = NL_SOCKET_IO;
        result->host_errno = saved;
    }
}
static void socket_rollback(NlSocketService *service, const NlCap *caps,
                            unsigned count, const int descriptors[2],
                            NlSocketResult *result) {
    for (unsigned i = 0; i < count; i++) {
        if (nl_cap_private_consume(service->caps, &caps[i]) != NL_CAP_OK)
            socket_cleanup_error(result, 0);
    }
    for (unsigned i = 0; i < 2; i++)
        if (descriptors[i] >= 0) socket_close_once(service, descriptors[i], result, true);
}
static bool socket_configure(int fd) {
    int flags = fcntl(fd, F_GETFL);
    if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) return false;
    flags = fcntl(fd, F_GETFD);
    if (flags < 0 || fcntl(fd, F_SETFD, flags | FD_CLOEXEC) < 0) return false;
#ifdef __APPLE__
    int enabled = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &enabled, sizeof(enabled)) != 0)
        return false;
#endif
    return true;
}
static NlSocketResult socket_io_error(int saved) {
    NlSocketResult result = socket_result(saved == EINTR ? NL_SOCKET_INTERRUPTED :
        ((saved == EAGAIN || saved == EWOULDBLOCK) ? NL_SOCKET_WOULD_BLOCK : NL_SOCKET_IO));
    result.host_errno = saved;
    return result;
}

NlSocketResult nl_socket_service_create(NlSocketService **out) {
    if (!out) return socket_result(NL_SOCKET_ARGUMENT);
    if (socket_context_counter == UINT64_MAX) return socket_result(NL_SOCKET_LIMIT);
    NlSocketService *service = calloc(1, sizeof(*service));
    if (!service) return socket_result(NL_SOCKET_MEMORY);
    service->caps = nl_cap_table_create();
    if (!service->caps) {
        free(service);
        return socket_result(NL_SOCKET_MEMORY);
    }
    for (unsigned i = 0; i < NL_CAP_PRIVATE_SLOTS; i++) service->sockets[i].fd = -1;
    service->identity = ++socket_context_counter;
    *out = service;
    return socket_result(NL_SOCKET_OK);
}

NlSocketResult nl_socket_acquire_pair(NlSocketService *service, uint32_t left_rights,
                                     uint32_t right_rights, NlSocketPair *out) {
    NlSocketStatus status = socket_context(service);
    if (status != NL_SOCKET_OK) return socket_result(status);
    if (!out || ((left_rights | right_rights) & ~SOCKET_RIGHTS))
        return socket_result(NL_SOCKET_ARGUMENT);
    unsigned available = 0;
    for (unsigned i = 0; i < NL_CAP_PRIVATE_SLOTS; i++) available += !service->sockets[i].live;
    if (available < 2) return socket_result(NL_SOCKET_CAPACITY);
    NlCap caps[2];
    const uint32_t rights[2] = {left_rights, right_rights};
    int descriptors[2] = {-1, -1};
    /* I prepare both unpublished identities before acquiring host state. */
    for (unsigned i = 0; i < 2; i++) {
        int rc = nl_cap_private_mint(service->caps, SOCKET_TYPE, SOCKET_SERVICE, rights[i], &caps[i]);
        if (rc != NL_CAP_OK) {
            NlSocketResult result = socket_result(socket_cap_status(rc));
            socket_rollback(service, caps, i, descriptors, &result);
            return result;
        }
    }
    errno = 0;
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, descriptors) != 0) {
        NlSocketResult result = socket_io_error(errno);
        socket_rollback(service, caps, 2, descriptors, &result);
        return result;
    }
    for (unsigned i = 0; i < 2; i++) {
        errno = 0;
        if (!socket_configure(descriptors[i])) {
            NlSocketResult result = socket_io_error(errno);
            socket_rollback(service, caps, 2, descriptors, &result);
            return result;
        }
    }
    NlSocketPair pair;
    for (unsigned i = 0; i < 2; i++) {
        service->sockets[caps[i].slot] = (SocketEntry){descriptors[i], true, caps[i]};
        pair.endpoints[i] = (NlSocketToken){service->identity, caps[i]};
    }
    *out = pair;
    return socket_result(NL_SOCKET_OK);
}

NlSocketResult nl_socket_send_byte(NlSocketService *service,
                                  const NlSocketToken *token, uint8_t byte) {
    SocketEntry *entry = NULL;
    NlSocketStatus status = socket_resolve(service, token, NL_CAP_WRITE, &entry);
    if (status != NL_SOCKET_OK) return socket_result(status);
#ifdef __linux__
    const int flags = MSG_NOSIGNAL;
#else
    const int flags = 0;
#endif
    errno = 0;
    ssize_t sent = send(entry->fd, &byte, 1, flags);
    if (sent < 0) return socket_io_error(errno);
    if (sent != 1) return socket_result(NL_SOCKET_IO);
    NlSocketResult result = socket_result(NL_SOCKET_OK);
    result.bytes = 1;
    return result;
}

NlSocketResult nl_socket_receive_byte(NlSocketService *service,
                                     const NlSocketToken *token, uint8_t *out) {
    if (!out || !token || socket_overlap(out, sizeof(*out), token, sizeof(*token)))
        return socket_result(NL_SOCKET_ARGUMENT);
    SocketEntry *entry = NULL;
    NlSocketStatus status = socket_resolve(service, token, NL_CAP_READ, &entry);
    if (status != NL_SOCKET_OK) return socket_result(status);
    uint8_t byte = 0;
    errno = 0;
    ssize_t received = recv(entry->fd, &byte, 1, 0);
    if (received < 0) return socket_io_error(errno);
    if (received > 1) return socket_result(NL_SOCKET_IO);
    NlSocketResult result = socket_result(received ? NL_SOCKET_OK : NL_SOCKET_EOF);
    result.bytes = (size_t)received;
    result.eof = received == 0;
    *out = received ? byte : 0;
    return result;
}

NlSocketResult nl_socket_transfer(NlSocketService *service,
                                  const NlSocketToken *token, NlSocketToken *out) {
    if (!out) return socket_result(NL_SOCKET_ARGUMENT);
    SocketEntry *entry = NULL;
    NlSocketStatus status = socket_resolve(service, token, NL_CAP_TRANSFER, &entry);
    if (status != NL_SOCKET_OK) return socket_result(status);
    SocketEntry original = *entry;
    *entry = (SocketEntry){.fd = -1};
    NlCap next;
    int rc = nl_cap_private_transfer(service->caps, &original.token, &next);
    if (rc != NL_CAP_OK) {
        *entry = original;
        return socket_result(socket_cap_status(rc));
    }
    original.token = next;
    service->sockets[next.slot] = original;
    *out = (NlSocketToken){service->identity, next};
    NlSocketResult result = socket_result(NL_SOCKET_OK);
    result.consumed = true;
    return result;
}

NlSocketResult nl_socket_consume_close(NlSocketService *service, const NlSocketToken *token) {
    SocketEntry *entry = NULL;
    NlSocketStatus status = socket_resolve(service, token, 0, &entry);
    if (status != NL_SOCKET_OK) return socket_result(status);
    SocketEntry original = *entry;
    *entry = (SocketEntry){.fd = -1};
    int rc = nl_cap_private_consume(service->caps, &original.token);
    if (rc != NL_CAP_OK) {
        *entry = original;
        return socket_result(socket_cap_status(rc));
    }
    NlSocketResult result = socket_result(NL_SOCKET_OK);
    result.consumed = true;
    socket_close_once(service, original.fd, &result, false);
    return result;
}

NlSocketResult nl_socket_service_dispose(NlSocketService *service) {
    if (!service) return socket_result(NL_SOCKET_ARGUMENT);
    NlSocketResult result = socket_result(NL_SOCKET_OK);
    if (!service->disposed) {
        service->disposed = true;
        result.consumed = true;
        for (unsigned i = 0; i < NL_CAP_PRIVATE_SLOTS; i++) {
            SocketEntry entry = service->sockets[i];
            if (!entry.live) continue;
            service->sockets[i] = (SocketEntry){.fd = -1};
            int rc = nl_cap_private_consume(service->caps, &entry.token);
            if (rc != NL_CAP_OK) {
                if (result.status == NL_SOCKET_OK) result.status = socket_cap_status(rc);
                else socket_cleanup_error(&result, 0);
            }
            socket_close_once(service, entry.fd, &result, false);
        }
    }
    /* I keep an earlier ambiguous host closure visible even after the token was
     * retired. Repeated disposal is idempotent, not evidence that it closed. */
    if (service->closure_unknown) {
        result.closure_unknown = true;
        if (result.status == NL_SOCKET_OK) {
            result.status = NL_SOCKET_IO;
            result.host_errno = service->first_unknown_errno;
        }
    }
    return result;
}

NlSocketResult nl_socket_service_destroy(NlSocketService *service) {
    if (!service) return socket_result(NL_SOCKET_ARGUMENT);
    NlSocketResult result = nl_socket_service_dispose(service);
    nl_cap_table_destroy(service->caps);
    free(service);
    return result;
}

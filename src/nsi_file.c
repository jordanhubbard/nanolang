#include "nsi_file.h"
#include "nsi_cap_private.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FILE_TYPE "nsi:nanolang/filesystem#File"
#define FILE_SERVICE "nsi:nanolang/filesystem"
#define FILE_RIGHTS (NL_CAP_READ | NL_CAP_WRITE | NL_CAP_TRANSFER)

typedef enum { FILE_NEUTRAL, FILE_READING, FILE_WRITING } FileDirection;
typedef struct {
    FILE *stream;
    NlCap token;
    FileDirection direction;
} FileEntry;
struct NlFileService {
    NlCapTable *caps;
    uint64_t identity;
    bool disposed;
    FileEntry files[NL_CAP_PRIVATE_SLOTS];
};

/* Serialized private context creation, not a process-shared/remote token ABI.
 * I never reset this identity when a context is freed or its address is reused. */
static uint64_t file_context_counter;

static NlFileResult file_result(NlFileStatus status) {
    NlFileResult r = {0};
    r.status = status;
    return r;
}
static NlFileStatus file_cap_status(int status) {
    switch (status) {
    case NL_CAP_OK: return NL_FILE_OK;
    case NL_CAP_ERR_RIGHTS: case NL_CAP_ERR_TRANSFER: return NL_FILE_RIGHTS;
    case NL_CAP_ERR_FULL: return NL_FILE_CAPACITY;
    case NL_CAP_PRIVATE_ERR_GENERATION: return NL_FILE_LIMIT;
    default: return NL_FILE_TOKEN;
    }
}
static bool file_same_cap(NlCap a, NlCap b) {
    return a.slot == b.slot && a.generation == b.generation && a.secret == b.secret;
}
static NlFileStatus file_context(const NlFileService *s) {
    if (!s) return NL_FILE_ARGUMENT;
    return s->disposed ? NL_FILE_DISPOSED : NL_FILE_OK;
}
static NlFileStatus file_resolve(NlFileService *s, const NlFileToken *token,
                                 uint32_t rights, FileEntry **out) {
    NlFileStatus status = file_context(s);
    if (status != NL_FILE_OK) return status;
    if (!token) return NL_FILE_ARGUMENT;
    if (token->context_id != s->identity || token->cap.slot >= NL_CAP_PRIVATE_SLOTS)
        return NL_FILE_TOKEN;
    int rc = nl_cap_check(s->caps, &token->cap, rights);
    if (rc != NL_CAP_OK) return file_cap_status(rc);
    FileEntry *entry = &s->files[token->cap.slot];
    if (!entry->stream || !file_same_cap(entry->token, token->cap) ||
        strcmp(nl_cap_type_id(s->caps, &token->cap), FILE_TYPE) ||
        strcmp(nl_cap_service_id(s->caps, &token->cap), FILE_SERVICE))
        return NL_FILE_TOKEN;
    *out = entry;
    return NL_FILE_OK;
}
static void file_rollback_stream(FILE *stream, NlFileResult *result) {
    errno = 0;
    int rc = fclose(stream);
    int saved = errno;
    if (rc != 0) {
        result->cleanup_failed = true;
        result->cleanup_errno = saved;
    }
}

NlFileResult nl_file_service_create(NlFileService **out) {
    if (!out) return file_result(NL_FILE_ARGUMENT);
    if (file_context_counter == UINT64_MAX) return file_result(NL_FILE_LIMIT);
    NlFileService *s = calloc(1, sizeof(*s));
    if (!s) return file_result(NL_FILE_MEMORY);
    s->caps = nl_cap_table_create();
    if (!s->caps) {
        free(s);
        return file_result(NL_FILE_MEMORY);
    }
    s->identity = ++file_context_counter;
    *out = s;
    return file_result(NL_FILE_OK);
}

NlFileResult nl_file_acquire_temp(NlFileService *s, uint32_t rights, NlFileToken *out) {
    NlFileStatus status = file_context(s);
    if (status != NL_FILE_OK) return file_result(status);
    if (!out || (rights & ~FILE_RIGHTS)) return file_result(NL_FILE_ARGUMENT);
    uint32_t slot = 0;
    while (slot < NL_CAP_PRIVATE_SLOTS && s->files[slot].stream) slot++;
    if (slot == NL_CAP_PRIVATE_SLOTS) return file_result(NL_FILE_CAPACITY);
    /* Registry storage is already reserved. Only the host acquisition and
     * private capability publication can fail after this point. */
    errno = 0;
    FILE *stream = tmpfile();
    int saved = errno;
    if (!stream) {
        NlFileResult result = file_result(NL_FILE_IO);
        result.host_errno = saved;
        return result;
    }
    NlCap cap;
    int rc = nl_cap_private_mint(s->caps, FILE_TYPE, FILE_SERVICE, rights, &cap);
    if (rc != NL_CAP_OK) {
        NlFileResult result = file_result(file_cap_status(rc));
        file_rollback_stream(stream, &result);
        return result;
    }
    /* The table and registry are exclusively owned by this context. Mint only
     * chooses a vacant slot, and every other operation maintains this pairing. */
    s->files[cap.slot] = (FileEntry){stream, cap, FILE_NEUTRAL};
    *out = (NlFileToken){s->identity, cap};
    return file_result(NL_FILE_OK);
}

NlFileResult nl_file_read(NlFileService *s, const NlFileToken *token,
                          void *buffer, size_t capacity) {
    if ((!buffer && capacity) || capacity > (size_t)PTRDIFF_MAX)
        return file_result(NL_FILE_ARGUMENT);
    FileEntry *entry = NULL;
    NlFileStatus status = file_resolve(s, token, NL_CAP_READ, &entry);
    if (status != NL_FILE_OK) return file_result(status);
    NlFileResult result = file_result(NL_FILE_OK);
    if (!capacity) return result;
    if (entry->direction == FILE_WRITING) return file_result(NL_FILE_DIRECTION);
    errno = 0;
    result.bytes = fread(buffer, 1, capacity, entry->stream);
    int saved = errno;
    entry->direction = FILE_READING;
    result.eof = feof(entry->stream) != 0;
    if (ferror(entry->stream)) {
        result.status = NL_FILE_IO;
        result.host_errno = saved;
    }
    return result;
}

NlFileResult nl_file_write(NlFileService *s, const NlFileToken *token,
                           const void *bytes, size_t length) {
    if ((!bytes && length) || length > (size_t)PTRDIFF_MAX)
        return file_result(NL_FILE_ARGUMENT);
    FileEntry *entry = NULL;
    NlFileStatus status = file_resolve(s, token, NL_CAP_WRITE, &entry);
    if (status != NL_FILE_OK) return file_result(status);
    NlFileResult result = file_result(NL_FILE_OK);
    if (!length) return result;
    if (entry->direction == FILE_READING) return file_result(NL_FILE_DIRECTION);
    errno = 0;
    result.bytes = fwrite(bytes, 1, length, entry->stream);
    int saved = errno;
    entry->direction = FILE_WRITING;
    if (result.bytes != length || ferror(entry->stream)) {
        result.status = NL_FILE_IO;
        result.host_errno = saved;
    }
    return result;
}

NlFileResult nl_file_rewind(NlFileService *s, const NlFileToken *token) {
    FileEntry *entry = NULL;
    NlFileStatus status = file_resolve(s, token, NL_CAP_READ, &entry);
    if (status != NL_FILE_OK) return file_result(status);
    errno = 0;
    int rc = fseek(entry->stream, 0, SEEK_SET);
    int saved = errno;
    if (rc != 0) {
        NlFileResult result = file_result(NL_FILE_IO);
        result.host_errno = saved;
        return result;
    }
    clearerr(entry->stream);
    entry->direction = FILE_NEUTRAL;
    return file_result(NL_FILE_OK);
}

NlFileResult nl_file_transfer(NlFileService *s, const NlFileToken *token,
                              NlFileToken *out) {
    if (!out) return file_result(NL_FILE_ARGUMENT);
    FileEntry *entry = NULL;
    NlFileStatus status = file_resolve(s, token, NL_CAP_TRANSFER, &entry);
    if (status != NL_FILE_OK) return file_result(status);
    FileEntry original = *entry;
    NlCap next;
    /* Detach before capability retirement, restoring on preparation failure.
     * No host operation or fallible allocation follows a successful transfer. */
    memset(entry, 0, sizeof(*entry));
    int rc = nl_cap_private_transfer(s->caps, &original.token, &next);
    if (rc != NL_CAP_OK) {
        *entry = original;
        return file_result(file_cap_status(rc));
    }
    original.token = next;
    s->files[next.slot] = original;
    *out = (NlFileToken){s->identity, next};
    NlFileResult result = file_result(NL_FILE_OK);
    result.consumed = true;
    return result;
}

NlFileResult nl_file_consume_close(NlFileService *s, const NlFileToken *token) {
    FileEntry *entry = NULL;
    NlFileStatus status = file_resolve(s, token, 0, &entry);
    if (status != NL_FILE_OK) return file_result(status);
    FileEntry original = *entry;
    memset(entry, 0, sizeof(*entry));
    int rc = nl_cap_private_consume(s->caps, &original.token);
    if (rc != NL_CAP_OK) {
        *entry = original;
        return file_result(file_cap_status(rc));
    }
    errno = 0;
    rc = fclose(original.stream);
    int saved = errno;
    NlFileResult result = file_result(rc == 0 ? NL_FILE_OK : NL_FILE_IO);
    result.consumed = true;
    if (rc != 0) result.host_errno = saved;
    return result;
}

NlFileResult nl_file_service_dispose(NlFileService *s) {
    if (!s) return file_result(NL_FILE_ARGUMENT);
    NlFileResult first = file_result(NL_FILE_OK);
    if (s->disposed) return first;
    s->disposed = true;
    first.consumed = true;
    for (uint32_t i = 0; i < NL_CAP_PRIVATE_SLOTS; i++) {
        FileEntry entry = s->files[i];
        if (!entry.stream) continue;
        memset(&s->files[i], 0, sizeof(s->files[i]));
        int cap_rc = nl_cap_private_consume(s->caps, &entry.token);
        errno = 0;
        int close_rc = fclose(entry.stream);
        int saved = errno;
        if (first.status == NL_FILE_OK && (cap_rc != NL_CAP_OK || close_rc != 0)) {
            first.status = cap_rc != NL_CAP_OK ? file_cap_status(cap_rc) : NL_FILE_IO;
            if (close_rc != 0) {
                if (cap_rc == NL_CAP_OK) first.host_errno = saved;
                else { first.cleanup_failed = true; first.cleanup_errno = saved; }
            }
        }
    }
    return first;
}

NlFileResult nl_file_service_destroy(NlFileService *s) {
    if (!s) return file_result(NL_FILE_ARGUMENT);
    NlFileResult result = nl_file_service_dispose(s);
    nl_cap_table_destroy(s->caps);
    free(s);
    return result;
}

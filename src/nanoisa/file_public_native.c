#include "file_native_public.h"
#include "file_native_emit.inc"

static bool file_public_identifier(const char *name) {
    if (!name) return false;
    for (size_t i = 0; i < 64; ++i) {
        unsigned char ch = (unsigned char)name[i];
        if (!ch) return i != 0;
        bool letter = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z');
        if (!letter && (!i || !((ch >= '0' && ch <= '9') || ch == '_')))
            return false;
    }
    return false;
}

NvmFileRuntimeStatus nvm2c_emit_file_bytes(const uint8_t *bytes, size_t size,
    const char *entry_identifier, char **out, char *diagnostic, size_t diagnostic_size) {
    NvmFileHostStatus entered = nvm_file_host_enter_query();
    if (entered != NVM_FILE_HOST_OK) return nvm_file_public_grant_status(entered);
    NvmFileRuntimeStatus status;
    if (!out || !bytes || !size || (diagnostic_size && !diagnostic) ||
        !file_public_identifier(entry_identifier)) {
        status = NVM_FILE_RUNTIME_INVALID;
        if (diagnostic && diagnostic_size)
            (void)snprintf(diagnostic, diagnostic_size, "I require exact File bytes and a valid entry identifier");
    } else status = file_native_emit_serialized(bytes, size, entry_identifier,
                                               out, diagnostic, diagnostic_size);
    nvm_file_host_leave();
    return status;
}

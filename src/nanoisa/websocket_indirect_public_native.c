#include "websocket_indirect_native_public.h"
#include "websocket_dispatch_config.h"
#include "service_indirect_native_emit.inc"

static bool websocket_public_identifier(const char *name) {
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

NvmWebSocketRuntimeStatus nvm2c_emit_websocket_indirect_bytes(const uint8_t *bytes, size_t size,
    const char *entry_identifier, char **out, char *diagnostic, size_t diagnostic_size) {
    NvmWebSocketHostStatus entered = nvm_websocket_host_enter_query();
    if (entered != NVM_WEBSOCKET_HOST_OK) return nvm_websocket_public_grant_status(entered);
    NvmWebSocketRuntimeStatus status;
    if (!out || !bytes || !size || (diagnostic_size && !diagnostic) ||
        !websocket_public_identifier(entry_identifier)) {
        status = NVM_WEBSOCKET_RUNTIME_INVALID;
        if (diagnostic && diagnostic_size)
            (void)snprintf(diagnostic, diagnostic_size, "I require exact WebSocket bytes and a valid entry identifier");
    } else status = file_indirect_native_emit_serialized(bytes, size, FNE_PUBLIC, entry_identifier,
                                               out, diagnostic, diagnostic_size);
    nvm_websocket_host_leave();
    return status;
}

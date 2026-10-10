/* I instantiate only this immutable catalog's nominal and flow identities. */
#include "socket_flow.h"
#include "service_socket_nominal_config.h"
#define FLOW_FN(name) nvm_socket_flow_##name
#define FLOW_TYPE(name) NvmSocketFlow##name
#define FLOW_CONST(name) NVM_SOCKET_FLOW_##name
#define FLOW_ENDPOINT_TYPE 8u
#define FLOW_ENDPOINT_CHECK NVM_SOCKET_FLOW_CHECK_ENDPOINT
#define CODE_FN(name) nvm_socket_code_##name
#define CODE_TYPE(name) NvmSocketCode##name
#define CODE_CONST(name) NVM_SOCKET_CODE_##name
#define BODY_FN(name) nvm_socket_body_##name
#define BODY_TYPE(name) NvmSocketBody##name
#define BODY_CONST(name) NVM_SOCKET_BODY_##name
#define SERVICE_FN(name) nvm_socket_##name
#define SERVICE_TYPE(name) NvmSocket##name
#define SERVICE_CONST(name) NVM_SOCKET_##name

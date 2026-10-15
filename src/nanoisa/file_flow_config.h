/* I instantiate only this immutable catalog's nominal and flow identities. */
#include "file_flow.h"
#include "service_file_nominal_config.h"
#define FLOW_FN(name) nvm_file_flow_##name
#define FLOW_TYPE(name) NvmFileFlow##name
#define FLOW_CONST(name) NVM_FILE_FLOW_##name
#define FLOW_ENDPOINT_TYPE UINT32_MAX
#define FLOW_ENDPOINT_CHECK 0u
#define CODE_FN(name) nvm_file_code_##name
#define CODE_TYPE(name) NvmFileCode##name
#define CODE_CONST(name) NVM_FILE_CODE_##name
#define BODY_FN(name) nvm_file_body_##name
#define BODY_TYPE(name) NvmFileBody##name
#define BODY_CONST(name) NVM_FILE_BODY_##name
#define SERVICE_FN(name) nvm_file_##name
#define SERVICE_TYPE(name) NvmFile##name
#define SERVICE_CONST(name) NVM_FILE_##name

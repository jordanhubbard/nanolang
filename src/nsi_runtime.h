#ifndef NL_NSI_RUNTIME_H
#define NL_NSI_RUNTIME_H

#include "nsi.h"

#include <stdint.h>
#include <stddef.h>

#define NL_NSI_OK 0
#define NL_NSI_ERR_MALFORMED 1
#define NL_NSI_ERR_UNAUTHORIZED 2
#define NL_NSI_ERR_BACKPRESSURE 3
#define NL_NSI_ERR_BREAKING 4
#define NL_NSI_ERR_CANCELLED 5
#define NL_NSI_ERR_DEADLINE 6
#define NL_NSI_ERR_UNSUPPORTED 7
#define NL_NSI_ERR_IO 8

#define NL_NSI_RIGHT_READ 1u
#define NL_NSI_RIGHT_WRITE 2u

typedef enum {
    NL_NSI_ADAPTER_INPROC = 0,
    NL_NSI_ADAPTER_MOCK,
    NL_NSI_ADAPTER_LOCAL
} NlNsiAdapterKind;

typedef struct {
    char type_id[128];
    char service_id[128];
    uint64_t generation;
    uint32_t rights;
    int live;
} NlNsiHandle;

typedef struct NlNsiSession NlNsiSession;

typedef struct {
    const char *method_id;
    const char *payload_json;
    const char *request_id;
    const char *capability;
    const char *auth;
    int async;
    int timeout_ms;
} NlNsiCall;

typedef struct {
    int status;
    char *payload_json;
    char *error_id;
    char *call_id;
} NlNsiResult;

void nl_nsi_result_free(NlNsiResult *r);

NlNsiSession *nl_nsi_session_open(const NlNsi *nsi, NlNsiAdapterKind kind,
                                  const char *schema_path,
                                  const char *auth,
                                  const char **caps, size_t cap_count,
                                  int queue_bound);
void nl_nsi_session_close(NlNsiSession *s);

int nl_nsi_session_hello(NlNsiSession *s, const NlNsi *client_nsi);
int nl_nsi_session_negotiated(const NlNsiSession *s);

int nl_nsi_invoke(NlNsiSession *s, const NlNsiCall *call, NlNsiResult *out);
int nl_nsi_cancel(NlNsiSession *s, const char *call_id);
int nl_nsi_take_async(NlNsiSession *s, NlNsiResult *out);

int nl_nsi_queue_len(const NlNsiSession *s);
const char *nl_nsi_last_log(const NlNsiSession *s);

int nl_nsi_handle_count(const NlNsiSession *s);
const NlNsiHandle *nl_nsi_handle_at(const NlNsiSession *s, int i);

char *nl_nsi_frame_request(const char *iface, const char *method,
                           const char *call_id, const char *payload,
                           const char *request_id, const char *cap,
                           const char *auth, int deadline_ms, int async);
int nl_nsi_frame_kind(const char *json, char *kind_out, size_t kind_n);

#endif

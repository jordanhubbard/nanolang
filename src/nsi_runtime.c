#include "nsi_runtime.h"

#include "cJSON.h"

#include <arpa/inet.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#define NL_NSI_MAX_HANDLES 32
#define NL_NSI_MAX_Q 8
#define NL_NSI_MAX_CAPS 16
#define NL_NSI_MAX_IDEMP 16

typedef struct {
    char method_id[192];
    char payload[1024];
    char request_id[64];
    char capability[128];
    char auth[64];
    char call_id[32];
    int timeout_ms;
    int cancelled;
    int done;
    int status;
    char result_payload[1024];
    char error_id[128];
} NlNsiQItem;

typedef struct {
    char request_id[64];
    char payload[1024];
    int status;
    int used;
} NlNsiIdemp;

struct NlNsiSession {
    const NlNsi *nsi;
    NlNsiAdapterKind kind;
    char schema_path[512];
    char auth[64];
    char caps[NL_NSI_MAX_CAPS][128];
    size_t cap_count;
    int negotiated;
    int fds[2];
    pid_t child;
    int owns_nsi;
    NlNsi *loaded;
    int qbound;
    int qcount;
    int qhead;
    int qtail;
    NlNsiQItem q[NL_NSI_MAX_Q];
    uint64_t next_call;
    uint64_t next_generation;
    NlNsiHandle handles[NL_NSI_MAX_HANDLES];
    int handle_count;
    char last_log[512];
    NlNsiIdemp idemp[NL_NSI_MAX_IDEMP];
};

void nl_nsi_result_free(NlNsiResult *r) {
    if (!r) return;
    free(r->payload_json);
    free(r->error_id);
    free(r->call_id);
    r->payload_json = NULL;
    r->error_id = NULL;
    r->call_id = NULL;
    r->status = 0;
}

static char *xstrdup(const char *s) {
    size_t n;
    char *d;
    if (!s) return NULL;
    n = strlen(s);
    d = malloc(n + 1);
    if (!d) return NULL;
    memcpy(d, s, n + 1);
    return d;
}

static int cap_granted(const NlNsiSession *s, const char *cap) {
    size_t i;
    if (!cap || !cap[0]) return s->cap_count == 0;
    for (i = 0; i < s->cap_count; i++) {
        if (strcmp(s->caps[i], cap) == 0) return 1;
    }
    return 0;
}

static int method_needs_cap(const NlNsi *nsi, const char **out) {
    if (!nsi || nsi->capability_count == 0) {
        if (out) *out = NULL;
        return 0;
    }
    if (out) *out = nsi->capabilities[0].id;
    return 1;
}

static int payload_keys_ok(const NlNsiMethod *m, cJSON *obj) {
    cJSON *child;
    size_t j;
    if (!m || !obj || !cJSON_IsObject(obj)) return 0;
    cJSON_ArrayForEach(child, obj) {
        int ok = 0;
        if (!child->string) return 0;
        for (j = 0; j < m->param_count; j++) {
            const NlNsiParam *p = &m->params[j];
            if (p->direction != NL_NSI_DIR_IN && p->direction != NL_NSI_DIR_INOUT)
                continue;
            if (p->name && strcmp(p->name, child->string) == 0) {
                ok = 1;
                break;
            }
        }
        if (!ok) return 0;
    }
    return 1;
}

static int json_int(cJSON *obj, const char *key, int *out) {
    cJSON *it;
    if (!obj || !key || !out) return 0;
    it = cJSON_GetObjectItemCaseSensitive(obj, key);
    if (!it || !cJSON_IsNumber(it)) return 0;
    *out = (int)it->valuedouble;
    return 1;
}

static char *json_str_dup(cJSON *obj, const char *key) {
    cJSON *it = cJSON_GetObjectItemCaseSensitive(obj, key);
    if (!it || !cJSON_IsString(it) || !it->valuestring) return NULL;
    return xstrdup(it->valuestring);
}

char *nl_nsi_frame_request(const char *iface, const char *method,
                           const char *call_id, const char *payload,
                           const char *request_id, const char *cap,
                           const char *auth, int deadline_ms, int async) {
    cJSON *o = cJSON_CreateObject();
    char *s;
    if (!o) return NULL;
    cJSON_AddNumberToObject(o, "nsi_version", 0);
    cJSON_AddStringToObject(o, "frame", "request");
    cJSON_AddStringToObject(o, "interface", iface ? iface : "");
    cJSON_AddStringToObject(o, "method", method ? method : "");
    cJSON_AddStringToObject(o, "call_id", call_id ? call_id : "");
    cJSON_AddRawToObject(o, "payload", payload && payload[0] ? payload : "{}");
    if (request_id && request_id[0])
        cJSON_AddStringToObject(o, "request_id", request_id);
    if (cap && cap[0])
        cJSON_AddStringToObject(o, "capability", cap);
    if (auth && auth[0])
        cJSON_AddStringToObject(o, "auth", auth);
    cJSON_AddNumberToObject(o, "deadline_ms", deadline_ms);
    cJSON_AddBoolToObject(o, "async", async ? 1 : 0);
    s = cJSON_PrintUnformatted(o);
    cJSON_Delete(o);
    return s;
}

int nl_nsi_frame_kind(const char *json, char *kind_out, size_t kind_n) {
    cJSON *o;
    cJSON *frame;
    if (!json || !kind_out || kind_n == 0) return -1;
    o = cJSON_Parse(json);
    if (!o) return -1;
    frame = cJSON_GetObjectItemCaseSensitive(o, "frame");
    if (!frame || !cJSON_IsString(frame) || !frame->valuestring) {
        cJSON_Delete(o);
        return -1;
    }
    strncpy(kind_out, frame->valuestring, kind_n - 1);
    kind_out[kind_n - 1] = '\0';
    cJSON_Delete(o);
    return 0;
}

static int write_full(int fd, const void *buf, size_t n) {
    const char *p = buf;
    size_t off = 0;
    while (off < n) {
        ssize_t w = write(fd, p + off, n - off);
        if (w < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (w == 0) return -1;
        off += (size_t)w;
    }
    return 0;
}

static int read_full(int fd, void *buf, size_t n) {
    char *p = buf;
    size_t off = 0;
    while (off < n) {
        ssize_t r = read(fd, p + off, n - off);
        if (r < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (r == 0) return -1;
        off += (size_t)r;
    }
    return 0;
}

static int write_msg(int fd, const char *json) {
    uint32_t n;
    uint32_t be;
    if (!json) return -1;
    n = (uint32_t)strlen(json);
    be = htonl(n);
    if (write_full(fd, &be, 4) != 0) return -1;
    return write_full(fd, json, n);
}

static char *read_msg(int fd) {
    uint32_t be;
    uint32_t n;
    char *buf;
    if (read_full(fd, &be, 4) != 0) return NULL;
    n = ntohl(be);
    if (n == 0 || n > 65536) return NULL;
    buf = malloc(n + 1);
    if (!buf) return NULL;
    if (read_full(fd, buf, n) != 0) {
        free(buf);
        return NULL;
    }
    buf[n] = '\0';
    return buf;
}

static NlNsiHandle *mint_handle(NlNsiSession *s, const char *type_id, uint32_t rights) {
    NlNsiHandle *h;
    if (!s || s->handle_count >= NL_NSI_MAX_HANDLES) return NULL;
    h = &s->handles[s->handle_count++];
    memset(h, 0, sizeof(*h));
    strncpy(h->type_id, type_id ? type_id : "", sizeof(h->type_id) - 1);
    strncpy(h->service_id, nl_nsi_interface_id(s->nsi), sizeof(h->service_id) - 1);
    s->next_generation++;
    h->generation = s->next_generation;
    h->rights = rights;
    h->live = 1;
    return h;
}

static int handler_inproc(NlNsiSession *s, const char *method, const char *payload,
                          char *out, size_t outn) {
    cJSON *obj = cJSON_Parse(payload ? payload : "{}");
    if (!obj || !cJSON_IsObject(obj)) {
        cJSON_Delete(obj);
        return NL_NSI_ERR_MALFORMED;
    }
    if (strcmp(method, "nsi:nanolang/log#write") == 0) {
        char *msg = json_str_dup(obj, "message");
        snprintf(s->last_log, sizeof(s->last_log), "%s", msg ? msg : "");
        free(msg);
        snprintf(out, outn, "{\"ok\":true}");
        cJSON_Delete(obj);
        return NL_NSI_OK;
    }
    if (strcmp(method, "nsi:nanolang/log#write_event") == 0) {
        snprintf(s->last_log, sizeof(s->last_log), "event");
        snprintf(out, outn, "{\"ok\":true}");
        cJSON_Delete(obj);
        return NL_NSI_OK;
    }
    if (strcmp(method, "nsi:nanolang/vector2d#add") == 0) {
        int x1, y1, x2, y2;
        if (!json_int(obj, "x1", &x1) || !json_int(obj, "y1", &y1) ||
            !json_int(obj, "x2", &x2) || !json_int(obj, "y2", &y2)) {
            cJSON_Delete(obj);
            return NL_NSI_ERR_MALFORMED;
        }
        snprintf(out, outn, "{\"x\":%d,\"y\":%d}", x1 + x2, y1 + y2);
        cJSON_Delete(obj);
        return NL_NSI_OK;
    }
    if (strcmp(method, "nsi:nanolang/filesystem#open") == 0 ||
        strcmp(method, "nsi:nanolang/process#spawn") == 0 ||
        strcmp(method, "nsi:nanolang/net#connect") == 0 ||
        strcmp(method, "nsi:nanolang/audio#write_frame") == 0 ||
        strcmp(method, "nsi:nanolang/graphics#present") == 0 ||
        strcmp(method, "nsi:nanolang/gpu#submit") == 0 ||
        strcmp(method, "nsi:nanolang/python#eval") == 0) {
        const char *tid = "nsi:core/resource";
        const NlNsiMethod *m = nl_nsi_find_method(s->nsi, method);
        NlNsiHandle *h;
        size_t i;
        if (m) {
            for (i = 0; i < m->param_count; i++) {
                if (m->params[i].direction == NL_NSI_DIR_RETURN && m->params[i].type_id)
                    tid = m->params[i].type_id;
            }
        }
        h = mint_handle(s, tid, NL_NSI_RIGHT_READ | NL_NSI_RIGHT_WRITE);
        if (!h) {
            cJSON_Delete(obj);
            return NL_NSI_ERR_IO;
        }
        snprintf(out, outn,
                 "{\"generation\":%llu,\"rights\":%u,\"type\":\"%s\",\"service\":\"%s\"}",
                 (unsigned long long)h->generation, h->rights, h->type_id, h->service_id);
        cJSON_Delete(obj);
        return NL_NSI_OK;
    }
    snprintf(out, outn, "{\"ok\":true}");
    cJSON_Delete(obj);
    return NL_NSI_OK;
}

static int lookup_idemp(NlNsiSession *s, const char *rid, char *out, size_t n, int *status) {
    int i;
    if (!rid || !rid[0]) return 0;
    for (i = 0; i < NL_NSI_MAX_IDEMP; i++) {
        if (s->idemp[i].used && strcmp(s->idemp[i].request_id, rid) == 0) {
            snprintf(out, n, "%s", s->idemp[i].payload);
            *status = s->idemp[i].status;
            return 1;
        }
    }
    return 0;
}

static void store_idemp(NlNsiSession *s, const char *rid, const char *payload, int status) {
    int i;
    if (!rid || !rid[0]) return;
    for (i = 0; i < NL_NSI_MAX_IDEMP; i++) {
        if (!s->idemp[i].used) {
            s->idemp[i].used = 1;
            strncpy(s->idemp[i].request_id, rid, sizeof(s->idemp[i].request_id) - 1);
            strncpy(s->idemp[i].payload, payload ? payload : "", sizeof(s->idemp[i].payload) - 1);
            s->idemp[i].status = status;
            return;
        }
        if (strcmp(s->idemp[i].request_id, rid) == 0) {
            strncpy(s->idemp[i].payload, payload ? payload : "", sizeof(s->idemp[i].payload) - 1);
            s->idemp[i].status = status;
            return;
        }
    }
}

static int dispatch_local(NlNsiSession *s, const char *method, const char *payload,
                          const char *request_id, char *out, size_t outn) {
    const NlNsiMethod *m = nl_nsi_find_method(s->nsi, method);
    int status;
    cJSON *obj;
    if (!m) return NL_NSI_ERR_UNSUPPORTED;
    obj = cJSON_Parse(payload ? payload : "{}");
    if (!obj || !cJSON_IsObject(obj) || !payload_keys_ok(m, obj)) {
        cJSON_Delete(obj);
        return NL_NSI_ERR_MALFORMED;
    }
    cJSON_Delete(obj);
    if (m->idempotent && lookup_idemp(s, request_id, out, outn, &status))
        return status;
    status = handler_inproc(s, method, payload, out, outn);
    if (m->idempotent)
        store_idemp(s, request_id, out, status);
    return status;
}

static const char *err_id(int status) {
    switch (status) {
    case NL_NSI_ERR_MALFORMED: return "nsi:core/malformed";
    case NL_NSI_ERR_UNAUTHORIZED: return "nsi:core/unauthorized";
    case NL_NSI_ERR_BACKPRESSURE: return "nsi:core/backpressure";
    case NL_NSI_ERR_BREAKING: return "nsi:core/breaking";
    case NL_NSI_ERR_CANCELLED: return "nsi:core/cancelled";
    case NL_NSI_ERR_DEADLINE: return "nsi:core/deadline";
    case NL_NSI_ERR_UNSUPPORTED: return "nsi:core/unsupported";
    case NL_NSI_ERR_IO: return "nsi:core/io";
    default: return "nsi:core/ok";
    }
}

static void fill_result(NlNsiResult *out, int status, const char *payload,
                        const char *call_id) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    out->status = status;
    out->payload_json = xstrdup(payload ? payload : "");
    out->call_id = xstrdup(call_id ? call_id : "");
    if (status != NL_NSI_OK)
        out->error_id = xstrdup(err_id(status));
}

static int authorize(NlNsiSession *s, const NlNsiCall *call) {
    const char *need = NULL;
    if (!s->negotiated) return NL_NSI_ERR_BREAKING;
    if (s->auth[0]) {
        if (!call->auth || strcmp(call->auth, s->auth) != 0)
            return NL_NSI_ERR_UNAUTHORIZED;
    }
    if (method_needs_cap(s->nsi, &need) && need) {
        const char *cap = call->capability ? call->capability : need;
        if (!cap_granted(s, cap))
            return NL_NSI_ERR_UNAUTHORIZED;
    }
    return NL_NSI_OK;
}

static int run_call(NlNsiSession *s, const NlNsiCall *call, const char *call_id,
                    NlNsiResult *out) {
    char payload[1024];
    int status;
    const NlNsiMethod *m;
    if (call->timeout_ms == 0) {
        fill_result(out, NL_NSI_ERR_DEADLINE, "", call_id);
        return NL_NSI_ERR_DEADLINE;
    }
    m = nl_nsi_find_method(s->nsi, call->method_id ? call->method_id : "");
    if (!m) {
        fill_result(out, NL_NSI_ERR_UNSUPPORTED, "", call_id);
        return NL_NSI_ERR_UNSUPPORTED;
    }
    if (s->kind == NL_NSI_ADAPTER_LOCAL && s->fds[0] >= 0) {
        char *req = nl_nsi_frame_request(nl_nsi_interface_id(s->nsi),
                                         call->method_id, call_id,
                                         call->payload_json, call->request_id,
                                         call->capability, call->auth,
                                         call->timeout_ms, call->async);
        char *resp;
        cJSON *o;
        if (!req) {
            fill_result(out, NL_NSI_ERR_IO, "", call_id);
            return NL_NSI_ERR_IO;
        }
        if (write_msg(s->fds[0], req) != 0) {
            free(req);
            fill_result(out, NL_NSI_ERR_IO, "", call_id);
            return NL_NSI_ERR_IO;
        }
        free(req);
        resp = read_msg(s->fds[0]);
        if (!resp) {
            fill_result(out, NL_NSI_ERR_IO, "", call_id);
            return NL_NSI_ERR_IO;
        }
        o = cJSON_Parse(resp);
        free(resp);
        if (!o) {
            fill_result(out, NL_NSI_ERR_MALFORMED, "", call_id);
            return NL_NSI_ERR_MALFORMED;
        }
        {
            cJSON *st = cJSON_GetObjectItemCaseSensitive(o, "status");
            cJSON *pl = cJSON_GetObjectItemCaseSensitive(o, "payload");
            char *ps = NULL;
            status = (st && cJSON_IsNumber(st)) ? (int)st->valuedouble : NL_NSI_ERR_MALFORMED;
            if (pl && cJSON_IsString(pl) && pl->valuestring)
                ps = pl->valuestring;
            else if (pl && !cJSON_IsString(pl)) {
                ps = cJSON_PrintUnformatted(pl);
                fill_result(out, status, ps ? ps : "", call_id);
                free(ps);
                cJSON_Delete(o);
                return status;
            }
            fill_result(out, status, ps ? ps : "", call_id);
        }
        cJSON_Delete(o);
        return status;
    }
    status = dispatch_local(s, call->method_id, call->payload_json,
                            call->request_id, payload, sizeof(payload));
    fill_result(out, status, payload, call_id);
    return status;
}

static void child_loop(int fd, const char *path) {
    NlNsi *nsi = nl_nsi_load_path(path);
    NlNsiSession *s;
    const char *caps[NL_NSI_MAX_CAPS];
    size_t i;
    size_t nc = 0;
    if (!nsi) _exit(1);
    for (i = 0; i < nsi->capability_count && nc < NL_NSI_MAX_CAPS; i++)
        caps[nc++] = nsi->capabilities[i].id;
    s = nl_nsi_session_open(nsi, NL_NSI_ADAPTER_INPROC, path, "", caps, nc, 8);
    if (!s) _exit(1);
    s->negotiated = 1;
    for (;;) {
        char *msg = read_msg(fd);
        cJSON *o;
        cJSON *method;
        cJSON *payload;
        cJSON *rid;
        cJSON *cid;
        char *pl = NULL;
        char outp[1024];
        int status;
        cJSON *resp;
        char *rs;
        if (!msg) break;
        o = cJSON_Parse(msg);
        free(msg);
        if (!o) {
            write_msg(fd, "{\"nsi_version\":0,\"frame\":\"error\",\"status\":1,\"payload\":\"\"}");
            continue;
        }
        method = cJSON_GetObjectItemCaseSensitive(o, "method");
        payload = cJSON_GetObjectItemCaseSensitive(o, "payload");
        rid = cJSON_GetObjectItemCaseSensitive(o, "request_id");
        cid = cJSON_GetObjectItemCaseSensitive(o, "call_id");
        if (payload && cJSON_IsObject(payload))
            pl = cJSON_PrintUnformatted(payload);
        else if (payload && cJSON_IsString(payload))
            pl = xstrdup(payload->valuestring);
        else
            pl = xstrdup("{}");
        status = dispatch_local(s,
                                (method && cJSON_IsString(method)) ? method->valuestring : "",
                                pl ? pl : "{}",
                                (rid && cJSON_IsString(rid)) ? rid->valuestring : "",
                                outp, sizeof(outp));
        free(pl);
        resp = cJSON_CreateObject();
        cJSON_AddNumberToObject(resp, "nsi_version", 0);
        cJSON_AddStringToObject(resp, "frame", status == NL_NSI_OK ? "response" : "error");
        cJSON_AddNumberToObject(resp, "status", status);
        cJSON_AddStringToObject(resp, "payload", outp);
        if (cid && cJSON_IsString(cid))
            cJSON_AddStringToObject(resp, "call_id", cid->valuestring);
        rs = cJSON_PrintUnformatted(resp);
        cJSON_Delete(resp);
        cJSON_Delete(o);
        if (rs) {
            write_msg(fd, rs);
            free(rs);
        }
    }
    nl_nsi_session_close(s);
    nl_nsi_free(nsi);
    _exit(0);
}

static int spawn_local(NlNsiSession *s) {
    int sp[2];
    pid_t pid;
    if (!s->schema_path[0]) return -1;
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sp) != 0) return -1;
    pid = fork();
    if (pid < 0) {
        close(sp[0]);
        close(sp[1]);
        return -1;
    }
    if (pid == 0) {
        close(sp[0]);
        signal(SIGPIPE, SIG_IGN);
        child_loop(sp[1], s->schema_path);
        _exit(1);
    }
    close(sp[1]);
    s->fds[0] = sp[0];
    s->fds[1] = -1;
    s->child = pid;
    return 0;
}

NlNsiSession *nl_nsi_session_open(const NlNsi *nsi, NlNsiAdapterKind kind,
                                  const char *schema_path,
                                  const char *auth,
                                  const char **caps, size_t cap_count,
                                  int queue_bound) {
    NlNsiSession *s;
    size_t i;
    if (!nsi) return NULL;
    if (queue_bound < 1) queue_bound = 1;
    if (queue_bound > NL_NSI_MAX_Q) queue_bound = NL_NSI_MAX_Q;
    if (cap_count > NL_NSI_MAX_CAPS) return NULL;
    s = calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->nsi = nsi;
    s->kind = kind;
    s->fds[0] = -1;
    s->fds[1] = -1;
    s->child = -1;
    s->qbound = queue_bound;
    if (schema_path)
        strncpy(s->schema_path, schema_path, sizeof(s->schema_path) - 1);
    if (auth)
        strncpy(s->auth, auth, sizeof(s->auth) - 1);
    for (i = 0; i < cap_count; i++) {
        if (!caps[i]) {
            nl_nsi_session_close(s);
            return NULL;
        }
        strncpy(s->caps[i], caps[i], sizeof(s->caps[i]) - 1);
    }
    s->cap_count = cap_count;
    if (kind == NL_NSI_ADAPTER_LOCAL) {
        if (spawn_local(s) != 0) {
            nl_nsi_session_close(s);
            return NULL;
        }
    }
    return s;
}

void nl_nsi_session_close(NlNsiSession *s) {
    if (!s) return;
    if (s->fds[0] >= 0) close(s->fds[0]);
    if (s->fds[1] >= 0) close(s->fds[1]);
    if (s->child > 0) {
        int st = 0;
        kill(s->child, SIGTERM);
        waitpid(s->child, &st, 0);
    }
    if (s->owns_nsi && s->loaded)
        nl_nsi_free(s->loaded);
    free(s);
}

int nl_nsi_session_hello(NlNsiSession *s, const NlNsi *client_nsi) {
    if (!s || !client_nsi) return NL_NSI_ERR_MALFORMED;
    if (nl_nsi_compat(client_nsi, s->nsi) != NL_NSI_COMPAT_OK)
        return NL_NSI_ERR_BREAKING;
    s->negotiated = 1;
    return NL_NSI_OK;
}

int nl_nsi_session_negotiated(const NlNsiSession *s) {
    return s && s->negotiated;
}

static void copy_call(NlNsiQItem *q, const NlNsiCall *call, const char *cid) {
    memset(q, 0, sizeof(*q));
    strncpy(q->method_id, call->method_id ? call->method_id : "", sizeof(q->method_id) - 1);
    strncpy(q->payload, call->payload_json ? call->payload_json : "{}", sizeof(q->payload) - 1);
    strncpy(q->request_id, call->request_id ? call->request_id : "", sizeof(q->request_id) - 1);
    strncpy(q->capability, call->capability ? call->capability : "", sizeof(q->capability) - 1);
    strncpy(q->auth, call->auth ? call->auth : "", sizeof(q->auth) - 1);
    strncpy(q->call_id, cid, sizeof(q->call_id) - 1);
    q->timeout_ms = call->timeout_ms;
}

int nl_nsi_invoke(NlNsiSession *s, const NlNsiCall *call, NlNsiResult *out) {
    char cid[32];
    int authz;
    if (!s || !call || !out) return NL_NSI_ERR_MALFORMED;
    memset(out, 0, sizeof(*out));
    s->next_call++;
    snprintf(cid, sizeof(cid), "%llu", (unsigned long long)s->next_call);
    authz = authorize(s, call);
    if (authz != NL_NSI_OK) {
        fill_result(out, authz, "", cid);
        return authz;
    }
    if (call->async) {
        if (s->qcount >= s->qbound) {
            fill_result(out, NL_NSI_ERR_BACKPRESSURE, "", cid);
            return NL_NSI_ERR_BACKPRESSURE;
        }
        copy_call(&s->q[s->qtail], call, cid);
        s->qtail = (s->qtail + 1) % NL_NSI_MAX_Q;
        s->qcount++;
        fill_result(out, NL_NSI_OK, "", cid);
        return NL_NSI_OK;
    }
    return run_call(s, call, cid, out);
}

int nl_nsi_cancel(NlNsiSession *s, const char *call_id) {
    int i;
    int n;
    if (!s || !call_id) return NL_NSI_ERR_MALFORMED;
    n = s->qcount;
    i = s->qhead;
    while (n > 0) {
        if (strcmp(s->q[i].call_id, call_id) == 0) {
            s->q[i].cancelled = 1;
            return NL_NSI_OK;
        }
        i = (i + 1) % NL_NSI_MAX_Q;
        n--;
    }
    return NL_NSI_ERR_UNSUPPORTED;
}

int nl_nsi_take_async(NlNsiSession *s, NlNsiResult *out) {
    NlNsiQItem *item;
    NlNsiCall call;
    if (!s || !out) return NL_NSI_ERR_MALFORMED;
    memset(out, 0, sizeof(*out));
    if (s->qcount == 0) return NL_NSI_ERR_UNSUPPORTED;
    item = &s->q[s->qhead];
    s->qhead = (s->qhead + 1) % NL_NSI_MAX_Q;
    s->qcount--;
    if (item->cancelled) {
        fill_result(out, NL_NSI_ERR_CANCELLED, "", item->call_id);
        return NL_NSI_ERR_CANCELLED;
    }
    memset(&call, 0, sizeof(call));
    call.method_id = item->method_id;
    call.payload_json = item->payload;
    call.request_id = item->request_id;
    call.capability = item->capability;
    call.auth = item->auth;
    call.timeout_ms = item->timeout_ms;
    return run_call(s, &call, item->call_id, out);
}

int nl_nsi_queue_len(const NlNsiSession *s) {
    return s ? s->qcount : 0;
}

const char *nl_nsi_last_log(const NlNsiSession *s) {
    return s ? s->last_log : "";
}

int nl_nsi_handle_count(const NlNsiSession *s) {
    return s ? s->handle_count : 0;
}

const NlNsiHandle *nl_nsi_handle_at(const NlNsiSession *s, int i) {
    if (!s || i < 0 || i >= s->handle_count) return NULL;
    return &s->handles[i];
}

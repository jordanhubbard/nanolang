#include "nsi_fabric.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#ifndef NL_FAB_MAX
#define NL_FAB_MAX 24
#endif

typedef struct {
    NlServiceSpec spec;
    int registered;
    int started;
    int ready;
    int alive;
    int hung;
    int cancelled;
    int generation;
    int preserve;
    int mem_used;
    int cpu_used;
    int handles_used;
    int queue_used;
    int files_used;
    int net_used;
    int devices_used;
    int ipc[2];
    char last_request[64];
    char last_trace[64];
    char last_audit[64];
    char state[64];
    int fail_next;
    NlFailClass last_fail;
} NlFabSlot;

struct NlFabric {
    NlHost host;
    NlCapTable *caps;
    NlFabSlot slot[NL_FAB_MAX];
    int n;
    int started;
    int walker_idx;
    int freeze_idx;
};

static int posix_ipc(int fd[2]) {
    return socketpair(AF_UNIX, SOCK_STREAM, 0, fd);
}

static uint64_t posix_clock(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static int posix_entropy(void *buf, size_t n) {
#ifdef __APPLE__
    arc4random_buf(buf, n);
    return 0;
#else
    return getentropy(buf, n);
#endif
}

static void *posix_shm(size_t n, int *copy) {
    void *p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_ANON | MAP_SHARED, -1, 0);
    if (p != MAP_FAILED) {
        if (copy) *copy = 0;
        return p;
    }
    if (copy) *copy = 1;
    return calloc(1, n);
}

static void posix_shm_free(void *p, size_t n, int copy) {
    if (!p) return;
    if (copy)
        free(p);
    else
        munmap(p, n);
}

static void *inproc_shm(size_t n, int *copy) {
    if (copy) *copy = 1;
    return calloc(1, n);
}

static int posix_spawn(const char *path, char *const argv[], int *pid_out) {
    pid_t pid = fork();
    (void)path;
    (void)argv;
    if (pid < 0) return -1;
    if (pid == 0)
        _exit(0);
    if (pid_out) *pid_out = (int)pid;
    return 0;
}

static void *thread_identity(void *arg) {
    return arg;
}

static int posix_thread(void *(*fn)(void *), void *arg) {
    pthread_t t;
    if (pthread_create(&t, NULL, fn ? fn : thread_identity, arg) != 0)
        return -1;
    pthread_join(t, NULL);
    return 0;
}

static int posix_file_open(const char *path, int flags) {
    if (!path) return -1;
    return open(path, flags);
}

static int posix_net_socket(int domain, int type, int protocol) {
    return socket(domain, type, protocol);
}

static int posix_device_open(const char *path) {
    if (!path) {
        errno = EINVAL;
        return -1;
    }
    return open(path, O_RDONLY);
}

static uint32_t posix_uid(void) {
    return (uint32_t)getuid();
}

static void host_fill(NlHost *h, const char *name,
                      void *(*shm)(size_t, int *)) {
    memset(h, 0, sizeof(*h));
    h->spawn = posix_spawn;
    h->thread_run = posix_thread;
    h->ipc_pair = posix_ipc;
    h->clock_ns = posix_clock;
    h->entropy = posix_entropy;
    h->shm_alloc = shm;
    h->shm_free = posix_shm_free;
    h->file_open = posix_file_open;
    h->net_socket = posix_net_socket;
    h->device_open = posix_device_open;
    h->credential_uid = posix_uid;
    h->name = name;
}

NlHost nl_host_posix(void) {
    NlHost h;
    host_fill(&h, "posix", posix_shm);
    return h;
}

NlHost nl_host_inproc(void) {
    NlHost h;
    host_fill(&h, "inproc", inproc_shm);
    return h;
}

NlFabric *nl_fabric_create(const NlHost *host) {
    NlFabric *f = calloc(1, sizeof(*f));
    if (!f) return NULL;
    f->host = host ? *host : nl_host_posix();
    f->caps = nl_cap_table_create();
    f->walker_idx = -1;
    f->freeze_idx = -1;
    if (!f->caps) {
        free(f);
        return NULL;
    }
    return f;
}

void nl_fabric_destroy(NlFabric *f) {
    int i;
    if (!f) return;
    for (i = 0; i < f->n; i++) {
        if (f->slot[i].ipc[0] >= 0) close(f->slot[i].ipc[0]);
        if (f->slot[i].ipc[1] >= 0) close(f->slot[i].ipc[1]);
    }
    nl_cap_table_destroy(f->caps);
    free(f);
}

static NlFabSlot *find(NlFabric *f, const char *name) {
    int i;
    if (!f || !name) return NULL;
    for (i = 0; i < f->n; i++)
        if (strcmp(f->slot[i].spec.name, name) == 0) return &f->slot[i];
    return NULL;
}

int nl_fabric_register(NlFabric *f, const NlServiceSpec *spec) {
    NlFabSlot *s;
    if (!f || !spec || !spec->name[0] || f->n >= NL_FAB_MAX) return NL_FAB_ERR;
    if (find(f, spec->name)) return NL_FAB_ERR;
    s = &f->slot[f->n++];
    memset(s, 0, sizeof(*s));
    s->spec = *spec;
    s->registered = 1;
    s->ipc[0] = -1;
    s->ipc[1] = -1;
    s->generation = 1;
    if (strcmp(spec->name, "editor.walker") == 0) f->walker_idx = f->n - 1;
    if (strcmp(spec->name, "editor.freeze") == 0) f->freeze_idx = f->n - 1;
    return NL_FAB_OK;
}

static int dep_started(NlFabric *f, const char *dep) {
    NlFabSlot *s;
    if (!dep || !dep[0]) return 1;
    s = find(f, dep);
    return s && s->started && s->ready;
}

int nl_fabric_start(NlFabric *f) {
    int progress, i, guard = 0;
    if (!f) return NL_FAB_ERR;
    do {
        progress = 0;
        for (i = 0; i < f->n; i++) {
            NlFabSlot *s = &f->slot[i];
            if (s->started) continue;
            if (!dep_started(f, s->spec.dep)) continue;
            if (s->spec.isolated || s->spec.remote) {
                if (f->host.ipc_pair(s->ipc) != 0) return NL_FAB_ERR;
            }
            s->started = 1;
            s->ready = 1;
            s->alive = 1;
            progress = 1;
        }
        guard++;
    } while (progress && guard < NL_FAB_MAX + 2);
    for (i = 0; i < f->n; i++)
        if (!f->slot[i].started) return NL_FAB_ERR_HEALTH;
    f->started = 1;
    return NL_FAB_OK;
}

int nl_fabric_shutdown(NlFabric *f) {
    int i;
    if (!f) return NL_FAB_ERR;
    for (i = f->n - 1; i >= 0; i--) {
        f->slot[i].ready = 0;
        f->slot[i].alive = 0;
        f->slot[i].started = 0;
        if (f->slot[i].ipc[0] >= 0) {
            close(f->slot[i].ipc[0]);
            f->slot[i].ipc[0] = -1;
        }
        if (f->slot[i].ipc[1] >= 0) {
            close(f->slot[i].ipc[1]);
            f->slot[i].ipc[1] = -1;
        }
    }
    f->started = 0;
    return NL_FAB_OK;
}

int nl_fabric_ready(NlFabric *f, const char *name) {
    NlFabSlot *s = find(f, name);
    return s && s->ready ? 1 : 0;
}

int nl_fabric_health(NlFabric *f, const char *name) {
    NlFabSlot *s = find(f, name);
    return s && s->alive && !s->hung ? 1 : 0;
}

const char *nl_fabric_discover(NlFabric *f, const char *interface_id) {
    int i;
    if (!f || !interface_id) return NULL;
    for (i = 0; i < f->n; i++)
        if (strcmp(f->slot[i].spec.interface_id, interface_id) == 0)
            return f->slot[i].spec.name;
    return NULL;
}

int nl_fabric_startup_index(NlFabric *f, const char *name) {
    int i;
    if (!f) return -1;
    for (i = 0; i < f->n; i++)
        if (strcmp(f->slot[i].spec.name, name) == 0) return i;
    return -1;
}

int nl_fabric_has_ipc(NlFabric *f, const char *name) {
    NlFabSlot *s = find(f, name);
    return s && s->ipc[0] >= 0 ? 1 : 0;
}

const NlHost *nl_fabric_host(const NlFabric *f) {
    return f ? &f->host : NULL;
}

static int over_quota(NlFabSlot *s) {
    if (s->spec.mem_budget && s->mem_used > (int)s->spec.mem_budget) return 1;
    if (s->spec.cpu_ms_budget && s->cpu_used > (int)s->spec.cpu_ms_budget) return 1;
    if (s->spec.handle_budget && s->handles_used > s->spec.handle_budget) return 1;
    if (s->spec.queue_budget && s->queue_used > s->spec.queue_budget) return 1;
    if (s->spec.file_budget && s->files_used > s->spec.file_budget) return 1;
    if (s->spec.net_budget && s->net_used > s->spec.net_budget) return 1;
    if (s->spec.device_budget && s->devices_used > s->spec.device_budget) return 1;
    return 0;
}

static int cap_fail(NlFabric *f, const NlCap *cap, uint32_t need) {
    int rc;
    if (!cap) return NL_FAB_ERR_POLICY;
    rc = nl_cap_check(f->caps, cap, need);
    if (rc == NL_CAP_OK) return NL_FAB_OK;
    if (rc == NL_CAP_ERR_STALE || rc == NL_CAP_ERR_REVOKED || rc == NL_CAP_ERR_FORGED)
        return NL_FAB_ERR_STALE;
    return NL_FAB_ERR_POLICY;
}

static int dispatch(NlFabric *f, NlFabSlot *s, const char *method, const char *payload,
                    const NlCap *cap, char *out, size_t outn) {
    const char *scope;
    if (!method) method = "";
    if (!payload) payload = "";
    if (strcmp(s->spec.name, "log") == 0) {
        snprintf(out, outn, "logged:%s", payload);
        return NL_FAB_OK;
    }
    if (strcmp(s->spec.name, "fs") == 0) {
        int rc = cap_fail(f, cap, NL_CAP_READ);
        if (rc != NL_FAB_OK) return rc;
        scope = nl_cap_scope(f->caps, cap);
        if (scope[0] && strncmp(payload, scope, strlen(scope)) != 0)
            return NL_FAB_ERR_POLICY;
        s->files_used++;
        snprintf(out, outn, "path-scoped:%s", payload);
        return NL_FAB_OK;
    }
    if (strcmp(s->spec.name, "process") == 0) {
        int rc = cap_fail(f, cap, NL_CAP_TRANSFER);
        if (rc != NL_FAB_OK) return rc;
        scope = nl_cap_scope(f->caps, cap);
        if (scope[0] && strcmp(scope, payload) != 0 &&
            strstr(scope, payload) == NULL)
            return NL_FAB_ERR_POLICY;
        s->handles_used++;
        snprintf(out, outn, "proc:%s", payload[0] ? payload : method);
        return NL_FAB_OK;
    }
    if (strcmp(s->spec.name, "net") == 0) {
        int rc = cap_fail(f, cap, NL_CAP_WRITE);
        if (rc != NL_FAB_OK) return rc;
        scope = nl_cap_scope(f->caps, cap);
        if (scope[0] && strcmp(scope, payload) != 0)
            return NL_FAB_ERR_POLICY;
        s->net_used++;
        snprintf(out, outn, "endpoint-scoped:%s", payload);
        return NL_FAB_OK;
    }
    if (strcmp(s->spec.name, "audio") == 0) {
        int rc = cap_fail(f, cap, NL_CAP_MAP);
        if (rc != NL_FAB_OK) return rc;
        s->devices_used++;
        snprintf(out, outn, "stream+shm");
        return NL_FAB_OK;
    }
    if (strcmp(s->spec.name, "graphics") == 0) {
        int rc = cap_fail(f, cap, NL_CAP_BUFFER);
        if (rc != NL_FAB_OK) return rc;
        s->devices_used++;
        snprintf(out, outn, "surface+input");
        return NL_FAB_OK;
    }
    if (strcmp(s->spec.name, "gpu") == 0) {
        int rc = cap_fail(f, cap, NL_CAP_MAP);
        if (rc != NL_FAB_OK) return rc;
        s->devices_used++;
        snprintf(out, outn, "device/queue/memory/shader/sync");
        return NL_FAB_OK;
    }
    if (strcmp(s->spec.name, "python") == 0) {
        int rc = cap_fail(f, cap, NL_CAP_EVAL);
        if (rc != NL_FAB_OK) return rc;
        if (strstr(payload, "PyObject") || strstr(payload, "0x"))
            return NL_FAB_ERR_POLICY;
        snprintf(out, outn, "typed-adapter");
        return NL_FAB_OK;
    }
    if (strcmp(s->spec.name, "editor.walker") == 0) {
        if (strcmp(method, "eval") == 0) {
            int rc = cap_fail(f, cap, NL_CAP_EVAL);
            if (rc != NL_FAB_OK) return rc;
            snprintf(out, outn, "walker-eval:%s", payload);
            return NL_FAB_OK;
        }
        if (strcmp(method, "bind") == 0) {
            int rc = cap_fail(f, cap, NL_CAP_BUFFER);
            if (rc != NL_FAB_OK) return rc;
            snprintf(out, outn, "bound:%zu", strlen(payload));
            return NL_FAB_OK;
        }
        if (strcmp(method, "echo") == 0) {
            int rc = cap_fail(f, cap, NL_CAP_ECHO);
            if (rc != NL_FAB_OK) return rc;
            snprintf(out, outn, "%s", payload);
            return NL_FAB_OK;
        }
        if (strcmp(method, "chrome") == 0) {
            int rc = cap_fail(f, cap, NL_CAP_CHROME);
            if (rc != NL_FAB_OK) return rc;
            s->queue_used++;
            snprintf(out, outn, "chrome-queued");
            return NL_FAB_OK;
        }
        return NL_FAB_ERR_POLICY;
    }
    if (strcmp(s->spec.name, "editor.freeze") == 0) {
        int rc;
        if (strncmp(method, "ed_", 3) == 0) return NL_FAB_ERR_POLICY;
        rc = cap_fail(f, cap, NL_CAP_EVAL);
        if (rc != NL_FAB_OK) return rc;
        snprintf(out, outn, "ok:%s", payload[0] ? payload : "module");
        return NL_FAB_OK;
    }
    snprintf(out, outn, "ok");
    return NL_FAB_OK;
}

int nl_fabric_call(NlFabric *f, const char *name, const char *method,
                   const char *payload, const NlCap *cap,
                   int timeout_ms, const char *request_id, int idempotent,
                   const char *trace_id, const char *audit_id,
                   char *out, size_t outn, NlFailClass *cls) {
    NlFabSlot *s;
    int rc;
    if (cls) *cls = NL_FAIL_IMPLEMENTATION;
    if (!f || !name || !out || outn == 0) return NL_FAB_ERR;
    if (payload && strlen(payload) > NL_FAB_MAX_PAYLOAD) {
        if (cls) *cls = NL_FAIL_PROTOCOL;
        return NL_FAB_ERR;
    }
    s = find(f, name);
    if (!s || !s->started) return NL_FAB_ERR_HEALTH;
    if (trace_id)
        snprintf(s->last_trace, sizeof(s->last_trace), "%s", trace_id);
    if (audit_id)
        snprintf(s->last_audit, sizeof(s->last_audit), "%s", audit_id);
    if (method && strcmp(method, "fail-permanent") == 0) {
        if (cls) *cls = NL_FAIL_PERMANENT;
        return NL_FAB_ERR;
    }
    if (method && strcmp(method, "fail-protocol") == 0) {
        if (cls) *cls = NL_FAIL_PROTOCOL;
        return NL_FAB_ERR;
    }
    if (!s->alive) {
        if (s->spec.restart == NL_RESTART_FAIL_APPLICATION) {
            if (cls) *cls = NL_FAIL_PERMANENT;
            return NL_FAB_ERR_HEALTH;
        }
        if (s->spec.restart == NL_RESTART_FAIL_REQUEST ||
            s->spec.restart == NL_RESTART_NEVER) {
            if (cls) *cls = NL_FAIL_TRANSIENT;
            return NL_FAB_ERR_RETRY;
        }
        s->alive = 1;
        s->ready = 1;
        s->generation++;
        s->cancelled = 0;
        nl_cap_invalidate_service(f->caps, s->spec.interface_id);
        if (!s->preserve)
            s->state[0] = 0;
        if (cls) *cls = NL_FAIL_TRANSIENT;
    }
    if (s->cancelled) {
        if (cls) *cls = NL_FAIL_TRANSIENT;
        s->cancelled = 0;
        return NL_FAB_ERR_CANCEL;
    }
    if (s->hung) {
        if (timeout_ms >= 0 && timeout_ms < 1000) {
            if (cls) *cls = NL_FAIL_TRANSIENT;
            return NL_FAB_ERR_DEADLINE;
        }
        if (cls) *cls = NL_FAIL_TRANSIENT;
        return NL_FAB_ERR_CANCEL;
    }
    if (over_quota(s)) {
        if (cls) *cls = NL_FAIL_QUOTA;
        return NL_FAB_ERR_QUOTA;
    }
    s->queue_used++;
    if (over_quota(s)) {
        s->queue_used--;
        if (cls) *cls = NL_FAIL_QUOTA;
        return NL_FAB_ERR_QUOTA;
    }
    if (s->fail_next) {
        s->fail_next = 0;
        if (idempotent && request_id && request_id[0] &&
            strcmp(s->last_request, request_id) == 0 && s->state[0]) {
            snprintf(out, outn, "%s", s->state);
            if (cls) *cls = NL_FAIL_TRANSIENT;
            return NL_FAB_OK;
        }
        if (cls) *cls = NL_FAIL_TRANSIENT;
        return NL_FAB_ERR_RETRY;
    }
    rc = dispatch(f, s, method, payload, cap, out, outn);
    if (rc == NL_FAB_OK) {
        snprintf(s->state, sizeof(s->state), "%s", out);
        if (request_id)
            snprintf(s->last_request, sizeof(s->last_request), "%s", request_id);
        if (cls) *cls = NL_FAIL_TRANSIENT;
        s->cpu_used++;
        s->mem_used += 64;
    } else if (cls) {
        if (rc == NL_FAB_ERR_POLICY) *cls = NL_FAIL_AUTHORIZATION;
        else if (rc == NL_FAB_ERR_QUOTA) *cls = NL_FAIL_QUOTA;
        else if (rc == NL_FAB_ERR_STALE) *cls = NL_FAIL_AUTHORIZATION;
        else *cls = NL_FAIL_PROTOCOL;
    }
    return rc;
}

int nl_fabric_cancel(NlFabric *f, const char *name) {
    NlFabSlot *s = find(f, name);
    if (!s) return NL_FAB_ERR;
    s->cancelled = 1;
    s->hung = 0;
    return NL_FAB_OK;
}

int nl_fabric_crash(NlFabric *f, const char *name) {
    NlFabSlot *s = find(f, name);
    if (!s) return NL_FAB_ERR;
    s->alive = 0;
    s->ready = 0;
    s->hung = 0;
    s->cancelled = 0;
    return NL_FAB_OK;
}

int nl_fabric_replace(NlFabric *f, const char *name, const NlNsi *older,
                      const NlNsi *newer) {
    NlFabSlot *s = find(f, name);
    if (!s || !older || !newer) return NL_FAB_ERR;
    if (nl_nsi_compat(older, newer) != NL_NSI_COMPAT_OK)
        return NL_FAB_ERR_POLICY;
    s->generation++;
    s->alive = 1;
    s->ready = 1;
    return NL_FAB_OK;
}

int nl_fabric_preserve_state(NlFabric *f, const char *name, int preserve) {
    NlFabSlot *s = find(f, name);
    if (!s) return NL_FAB_ERR;
    s->preserve = preserve ? 1 : 0;
    return NL_FAB_OK;
}

int nl_fabric_generation(NlFabric *f, const char *name) {
    NlFabSlot *s = find(f, name);
    return s ? s->generation : 0;
}

int nl_fabric_fail_next(NlFabric *f, const char *name) {
    NlFabSlot *s = find(f, name);
    if (!s) return NL_FAB_ERR;
    s->fail_next = 1;
    return NL_FAB_OK;
}

int nl_fabric_set_budget(NlFabric *f, const char *name, const char *kind, int value) {
    NlFabSlot *s = find(f, name);
    if (!s || !kind || value < 0) return NL_FAB_ERR;
    if (strcmp(kind, "mem") == 0) s->spec.mem_budget = (size_t)value;
    else if (strcmp(kind, "cpu") == 0) s->spec.cpu_ms_budget = (size_t)value;
    else if (strcmp(kind, "handle") == 0) s->spec.handle_budget = value;
    else if (strcmp(kind, "queue") == 0) s->spec.queue_budget = value;
    else if (strcmp(kind, "file") == 0) s->spec.file_budget = value;
    else if (strcmp(kind, "net") == 0) s->spec.net_budget = value;
    else if (strcmp(kind, "device") == 0) s->spec.device_budget = value;
    else return NL_FAB_ERR;
    return NL_FAB_OK;
}

int nl_fabric_send_cap_remote(NlFabric *f, const char *name, const NlCap *cap) {
    NlFabSlot *s = find(f, name);
    (void)cap;
    if (!s) return NL_FAB_ERR;
    if (s->spec.remote) return NL_FAB_ERR_REMOTE;
    return NL_FAB_OK;
}

NlCapTable *nl_fabric_caps(NlFabric *f) {
    return f ? f->caps : NULL;
}

int nl_fabric_accounting(NlFabric *f, const char *name, NlAccounting *out) {
    NlFabSlot *s = find(f, name);
    if (!s || !out) return NL_FAB_ERR;
    out->mem = s->mem_used;
    out->cpu_ms = s->cpu_used;
    out->handles = s->handles_used;
    out->queue = s->queue_used;
    out->files = s->files_used;
    out->net = s->net_used;
    out->devices = s->devices_used;
    return NL_FAB_OK;
}

static void fill_spec(NlServiceSpec *sp, const char *name, const char *iface,
                      const char *schema, const char *dep) {
    memset(sp, 0, sizeof(*sp));
    snprintf(sp->name, sizeof(sp->name), "%s", name);
    snprintf(sp->interface_id, sizeof(sp->interface_id), "%s", iface);
    snprintf(sp->schema, sizeof(sp->schema), "%s", schema);
    if (dep) snprintf(sp->dep, sizeof(sp->dep), "%s", dep);
    sp->restart = NL_RESTART_ON_FAILURE;
    sp->mem_budget = 1024 * 1024;
    sp->cpu_ms_budget = 10000;
    sp->handle_budget = 64;
    sp->queue_budget = 128;
    sp->file_budget = 16;
    sp->net_budget = 16;
    sp->device_budget = 8;
}

int nl_fabric_register_core_services(NlFabric *f) {
    NlServiceSpec sp;
    if (!f) return NL_FAB_ERR;
    fill_spec(&sp, "log", "nsi:nanolang/log", "schema/nsi/examples/log.nsi.json", NULL);
    if (nl_fabric_register(f, &sp) != 0) return NL_FAB_ERR;
    fill_spec(&sp, "fs", "nsi:nanolang/fs", "schema/nsi/modules/filesystem.nsi.json", "log");
    if (nl_fabric_register(f, &sp) != 0) return NL_FAB_ERR;
    fill_spec(&sp, "process", "nsi:nanolang/process", "schema/nsi/modules/process.nsi.json", "log");
    if (nl_fabric_register(f, &sp) != 0) return NL_FAB_ERR;
    fill_spec(&sp, "net", "nsi:nanolang/net", "schema/nsi/modules/net.nsi.json", "log");
    if (nl_fabric_register(f, &sp) != 0) return NL_FAB_ERR;
    fill_spec(&sp, "audio", "nsi:nanolang/audio", "schema/nsi/modules/audio.nsi.json", "log");
    if (nl_fabric_register(f, &sp) != 0) return NL_FAB_ERR;
    fill_spec(&sp, "graphics", "nsi:nanolang/graphics", "schema/nsi/modules/graphics.nsi.json", "log");
    if (nl_fabric_register(f, &sp) != 0) return NL_FAB_ERR;
    fill_spec(&sp, "gpu", "nsi:nanolang/gpu", "schema/nsi/modules/gpu.nsi.json", "log");
    if (nl_fabric_register(f, &sp) != 0) return NL_FAB_ERR;
    fill_spec(&sp, "python", "nsi:nanolang/python", "schema/nsi/modules/python.nsi.json", "log");
    if (nl_fabric_register(f, &sp) != 0) return NL_FAB_ERR;
    fill_spec(&sp, "remote.diag", "nsi:nanolang/remote", "schema/nsi/examples/log.nsi.json", "log");
    sp.remote = 1;
    if (nl_fabric_register(f, &sp) != 0) return NL_FAB_ERR;
    return NL_FAB_OK;
}

int nl_fabric_register_editor(NlFabric *f) {
    NlServiceSpec sp;
    if (!f) return NL_FAB_ERR;
    fill_spec(&sp, "editor.walker", "nsi:nanolang/editor.walker",
              "schema/nsi/modules/eval.json", "log");
    sp.isolated = 1;
    sp.cpu_ms_budget = 200;
    sp.mem_budget = 65536;
    sp.queue_budget = 32;
    sp.handle_budget = 32;
    if (nl_fabric_register(f, &sp) != 0) return NL_FAB_ERR;
    fill_spec(&sp, "editor.freeze", "nsi:nanolang/editor.freeze",
              "schema/nsi/modules/eval.json", "editor.walker");
    sp.isolated = 1;
    sp.cpu_ms_budget = 200;
    if (nl_fabric_register(f, &sp) != 0) return NL_FAB_ERR;
    return NL_FAB_OK;
}

int nl_fabric_eval(NlFabric *f, const NlCap *cap, const char *src,
                  int timeout_ms, char *out, size_t outn) {
    NlFailClass cls;
    return nl_fabric_call(f, "editor.walker", "eval", src, cap, timeout_ms,
                          "eval-1", 1, "trace-eval", "audit-eval", out, outn, &cls);
}

int nl_fabric_bind_buffer(NlFabric *f, const NlCap *cap, const char *text) {
    char out[64];
    NlFailClass cls;
    return nl_fabric_call(f, "editor.walker", "bind", text ? text : "", cap, 100,
                          "bind-1", 1, "trace-bind", "audit-bind", out, sizeof(out), &cls);
}

int nl_fabric_bind_large(NlFabric *f, const NlCap *cap, const void *data, size_t n,
                         int force_copy, NlShm **region, size_t *bound) {
    NlShm *r;
    char tiny[32];
    int rc;
    if (!f || !cap || !data || n == 0) return NL_FAB_ERR;
    r = nl_shm_create(f->caps, cap, n, NL_SHM_FILE, force_copy);
    if (!r) return NL_FAB_ERR;
    if (nl_shm_write(r, 0, n, data) != NL_SHM_OK) {
        nl_shm_destroy(r);
        return NL_FAB_ERR;
    }
    if (nl_shm_transfer(r) != NL_SHM_OK) {
        nl_shm_destroy(r);
        return NL_FAB_ERR;
    }
    snprintf(tiny, sizeof(tiny), "%zu", n);
    rc = nl_fabric_bind_buffer(f, cap, tiny);
    if (rc != NL_FAB_OK) {
        nl_shm_destroy(r);
        return rc;
    }
    if (nl_shm_return(r) != NL_SHM_OK) {
        nl_shm_destroy(r);
        return NL_FAB_ERR;
    }
    if (bound) *bound = n;
    if (region)
        *region = r;
    else
        nl_shm_destroy(r);
    return NL_FAB_OK;
}

int nl_fabric_freeze(NlFabric *f, const NlCap *cap, const char *src,
                    char *out, size_t outn) {
    NlFailClass cls;
    return nl_fabric_call(f, "editor.freeze", "run", src, cap, 100,
                          "freeze-1", 1, "trace-freeze", "audit-freeze", out, outn, &cls);
}

int nl_fabric_set_hung(NlFabric *f, const char *name, int hung) {
    NlFabSlot *s = find(f, name);
    if (!s) return NL_FAB_ERR;
    s->hung = hung ? 1 : 0;
    return NL_FAB_OK;
}

int nl_fabric_walker_alive(NlFabric *f) {
    return f && f->walker_idx >= 0 && f->slot[f->walker_idx].alive ? 1 : 0;
}

int nl_fabric_freeze_alive(NlFabric *f) {
    return f && f->freeze_idx >= 0 && f->slot[f->freeze_idx].alive ? 1 : 0;
}

const char *nl_fabric_last_trace(NlFabric *f, const char *name) {
    NlFabSlot *s = find(f, name);
    return s ? s->last_trace : "";
}

const char *nl_fabric_last_audit(NlFabric *f, const char *name) {
    NlFabSlot *s = find(f, name);
    return s ? s->last_audit : "";
}

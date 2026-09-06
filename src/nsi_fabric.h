#ifndef NL_NSI_FABRIC_H
#define NL_NSI_FABRIC_H

#include "nsi.h"
#include "nsi_cap.h"
#include "nsi_shm.h"

#include <stddef.h>
#include <stdint.h>

#define NL_FAB_OK 0
#define NL_FAB_ERR 1
#define NL_FAB_ERR_POLICY 2
#define NL_FAB_ERR_QUOTA 3
#define NL_FAB_ERR_REMOTE 4
#define NL_FAB_ERR_DEADLINE 5
#define NL_FAB_ERR_CANCEL 6
#define NL_FAB_ERR_STALE 7
#define NL_FAB_ERR_HEALTH 8
#define NL_FAB_ERR_RETRY 9

#define NL_FAB_MAX_PAYLOAD 4096

typedef enum {
    NL_FAIL_TRANSIENT = 0,
    NL_FAIL_PERMANENT,
    NL_FAIL_PROTOCOL,
    NL_FAIL_AUTHORIZATION,
    NL_FAIL_QUOTA,
    NL_FAIL_IMPLEMENTATION
} NlFailClass;

typedef enum {
    NL_RESTART_NEVER = 0,
    NL_RESTART_ON_FAILURE,
    NL_RESTART_ALWAYS,
    NL_RESTART_FAIL_REQUEST,
    NL_RESTART_FAIL_APPLICATION,
    NL_RESTART_REPLACE
} NlRestartPolicy;

typedef struct {
    int (*spawn)(const char *path, char *const argv[], int *pid_out);
    int (*thread_run)(void *(*fn)(void *), void *arg);
    int (*ipc_pair)(int fd[2]);
    uint64_t (*clock_ns)(void);
    int (*entropy)(void *buf, size_t n);
    void *(*shm_alloc)(size_t n, int *copy_fallback);
    void (*shm_free)(void *p, size_t n, int copy_fallback);
    int (*file_open)(const char *path, int flags);
    int (*net_socket)(int domain, int type, int protocol);
    int (*device_open)(const char *path);
    uint32_t (*credential_uid)(void);
    const char *name;
} NlHost;

typedef struct {
    char name[64];
    char interface_id[128];
    char schema[256];
    char dep[64];
    NlRestartPolicy restart;
    int isolated;
    int remote;
    size_t mem_budget;
    size_t cpu_ms_budget;
    int handle_budget;
    int queue_budget;
    int file_budget;
    int net_budget;
    int device_budget;
} NlServiceSpec;

typedef struct {
    int mem;
    int cpu_ms;
    int handles;
    int queue;
    int files;
    int net;
    int devices;
} NlAccounting;

typedef struct NlFabric NlFabric;

NlHost nl_host_posix(void);
NlHost nl_host_inproc(void);

NlFabric *nl_fabric_create(const NlHost *host);
void nl_fabric_destroy(NlFabric *f);

int nl_fabric_register(NlFabric *f, const NlServiceSpec *spec);
int nl_fabric_start(NlFabric *f);
int nl_fabric_shutdown(NlFabric *f);
int nl_fabric_ready(NlFabric *f, const char *name);
int nl_fabric_health(NlFabric *f, const char *name);
const char *nl_fabric_discover(NlFabric *f, const char *interface_id);
int nl_fabric_startup_index(NlFabric *f, const char *name);
int nl_fabric_has_ipc(NlFabric *f, const char *name);
const NlHost *nl_fabric_host(const NlFabric *f);

int nl_fabric_call(NlFabric *f, const char *name, const char *method,
                   const char *payload, const NlCap *cap,
                   int timeout_ms, const char *request_id, int idempotent,
                   const char *trace_id, const char *audit_id,
                   char *out, size_t outn, NlFailClass *cls);

int nl_fabric_cancel(NlFabric *f, const char *name);
int nl_fabric_crash(NlFabric *f, const char *name);
int nl_fabric_replace(NlFabric *f, const char *name, const NlNsi *older,
                      const NlNsi *newer);
int nl_fabric_preserve_state(NlFabric *f, const char *name, int preserve);
int nl_fabric_generation(NlFabric *f, const char *name);
int nl_fabric_fail_next(NlFabric *f, const char *name);
int nl_fabric_set_budget(NlFabric *f, const char *name, const char *kind, int value);

int nl_fabric_send_cap_remote(NlFabric *f, const char *name, const NlCap *cap);

NlCapTable *nl_fabric_caps(NlFabric *f);
int nl_fabric_accounting(NlFabric *f, const char *name, NlAccounting *out);

int nl_fabric_register_core_services(NlFabric *f);
int nl_fabric_register_editor(NlFabric *f);

int nl_fabric_eval(NlFabric *f, const NlCap *cap, const char *src,
                   int timeout_ms, char *out, size_t outn);
int nl_fabric_bind_buffer(NlFabric *f, const NlCap *cap, const char *text);
int nl_fabric_bind_large(NlFabric *f, const NlCap *cap, const void *data, size_t n,
                         int force_copy, NlShm **region, size_t *bound);
int nl_fabric_freeze(NlFabric *f, const NlCap *cap, const char *src,
                     char *out, size_t outn);
int nl_fabric_set_hung(NlFabric *f, const char *name, int hung);
int nl_fabric_walker_alive(NlFabric *f);
int nl_fabric_freeze_alive(NlFabric *f);

#endif

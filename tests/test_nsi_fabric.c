#include "nsi.h"
#include "nsi_cap.h"
#include "nsi_fabric.h"
#include "nsi_shm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

static int g_pass = 0;
static int g_fail = 0;

#define PASS(name) do { g_pass++; printf("  %-60s PASS\n", (name)); } while (0)
#define FAIL(name, msg) do { g_fail++; printf("  %-60s FAIL: %s\n", (name), (msg)); } while (0)

static uint32_t walker_rights(void) {
    return NL_CAP_READ | NL_CAP_WRITE | NL_CAP_MAP | NL_CAP_SEAL | NL_CAP_TRANSFER |
           NL_CAP_BORROW | NL_CAP_RETURN | NL_CAP_REVOKE | NL_CAP_DELEGATE |
           NL_CAP_EVAL | NL_CAP_BUFFER | NL_CAP_ECHO | NL_CAP_CHROME;
}

static NlFabric *boot(const NlHost *host) {
    NlFabric *f = nl_fabric_create(host);
    if (!f) return NULL;
    if (nl_fabric_register_core_services(f) != 0 ||
        nl_fabric_register_editor(f) != 0 ||
        nl_fabric_start(f) != 0) {
        nl_fabric_destroy(f);
        return NULL;
    }
    return f;
}

static int mint(NlFabric *f, const char *type_id, const char *svc,
                uint32_t rights, const char *scope, NlCap *out) {
    return nl_cap_mint(nl_fabric_caps(f), type_id, svc, rights, 1, scope, out);
}

static void test_supervisor(void) {
    const char *name = "fabric: discover, order, health, ready, shutdown";
    NlFabric *f = boot(NULL);
    if (!f) { FAIL(name, "boot"); return; }
    if (!nl_fabric_ready(f, "log") || !nl_fabric_health(f, "fs")) {
        FAIL(name, "ready"); nl_fabric_destroy(f); return;
    }
    if (strcmp(nl_fabric_discover(f, "nsi:nanolang/log"), "log") != 0) {
        FAIL(name, "discover"); nl_fabric_destroy(f); return;
    }
    if (nl_fabric_startup_index(f, "log") >= nl_fabric_startup_index(f, "fs") ||
        nl_fabric_startup_index(f, "editor.walker") >=
            nl_fabric_startup_index(f, "editor.freeze")) {
        FAIL(name, "order"); nl_fabric_destroy(f); return;
    }
    if (nl_fabric_has_ipc(f, "log") || !nl_fabric_has_ipc(f, "editor.walker") ||
        !nl_fabric_has_ipc(f, "remote.diag")) {
        FAIL(name, "ipc"); nl_fabric_destroy(f); return;
    }
    if (nl_fabric_shutdown(f) != 0 || nl_fabric_ready(f, "log")) {
        FAIL(name, "shutdown"); nl_fabric_destroy(f); return;
    }
    PASS(name);
    nl_fabric_destroy(f);
}

static void test_restart_policies(void) {
    const char *name = "fabric: restart, fail-request, fail-application";
    NlFabric *f = nl_fabric_create(NULL);
    NlServiceSpec sp;
    char out[64];
    NlFailClass cls;
    memset(&sp, 0, sizeof(sp));
    snprintf(sp.name, sizeof(sp.name), "ok");
    snprintf(sp.interface_id, sizeof(sp.interface_id), "nsi:nanolang/ok");
    sp.restart = NL_RESTART_ON_FAILURE;
    sp.queue_budget = 32;
    sp.mem_budget = 1024;
    sp.cpu_ms_budget = 100;
    sp.handle_budget = 16;
    if (!f || nl_fabric_register(f, &sp) != 0) {
        FAIL(name, "reg"); nl_fabric_destroy(f); return;
    }
    memset(&sp, 0, sizeof(sp));
    snprintf(sp.name, sizeof(sp.name), "req");
    snprintf(sp.interface_id, sizeof(sp.interface_id), "nsi:nanolang/req");
    sp.restart = NL_RESTART_FAIL_REQUEST;
    sp.queue_budget = 32;
    sp.mem_budget = 1024;
    sp.cpu_ms_budget = 100;
    if (nl_fabric_register(f, &sp) != 0) {
        FAIL(name, "req"); nl_fabric_destroy(f); return;
    }
    memset(&sp, 0, sizeof(sp));
    snprintf(sp.name, sizeof(sp.name), "app");
    snprintf(sp.interface_id, sizeof(sp.interface_id), "nsi:nanolang/app");
    sp.restart = NL_RESTART_FAIL_APPLICATION;
    sp.queue_budget = 32;
    sp.mem_budget = 1024;
    sp.cpu_ms_budget = 100;
    if (nl_fabric_register(f, &sp) != 0 || nl_fabric_start(f) != 0) {
        FAIL(name, "start"); nl_fabric_destroy(f); return;
    }
    nl_fabric_crash(f, "ok");
    if (nl_fabric_call(f, "ok", "x", "", NULL, 10, "a", 0, "t", "u",
                       out, sizeof(out), &cls) != 0) {
        FAIL(name, "restart"); nl_fabric_destroy(f); return;
    }
    nl_fabric_crash(f, "req");
    if (nl_fabric_call(f, "req", "x", "", NULL, 10, "a", 0, "t", "u",
                       out, sizeof(out), &cls) != NL_FAB_ERR_RETRY) {
        FAIL(name, "fail-request"); nl_fabric_destroy(f); return;
    }
    nl_fabric_crash(f, "app");
    if (nl_fabric_call(f, "app", "x", "", NULL, 10, "a", 0, "t", "u",
                       out, sizeof(out), &cls) != NL_FAB_ERR_HEALTH ||
        cls != NL_FAIL_PERMANENT) {
        FAIL(name, "fail-app"); nl_fabric_destroy(f); return;
    }
    PASS(name);
    nl_fabric_destroy(f);
}

static void test_fail_classes_retry(void) {
    const char *name = "fabric: failure classes, idempotent retry, context";
    NlFabric *f = boot(NULL);
    char out[64];
    NlFailClass cls;
    char big[NL_FAB_MAX_PAYLOAD + 2];
    if (!f) { FAIL(name, "boot"); return; }
    memset(big, 'x', sizeof(big));
    big[sizeof(big) - 1] = 0;
    if (nl_fabric_call(f, "log", "fail-permanent", "", NULL, 10, "p", 0, "tr", "au",
                       out, sizeof(out), &cls) == 0 || cls != NL_FAIL_PERMANENT) {
        FAIL(name, "perm"); nl_fabric_destroy(f); return;
    }
    if (nl_fabric_call(f, "log", "fail-protocol", "", NULL, 10, "p", 0, "tr", "au",
                       out, sizeof(out), &cls) == 0 || cls != NL_FAIL_PROTOCOL) {
        FAIL(name, "proto"); nl_fabric_destroy(f); return;
    }
    if (nl_fabric_call(f, "log", "write", big, NULL, 10, "p", 0, "tr", "au",
                       out, sizeof(out), &cls) == 0 || cls != NL_FAIL_PROTOCOL) {
        FAIL(name, "oversize"); nl_fabric_destroy(f); return;
    }
    if (nl_fabric_call(f, "log", "write", "once", NULL, 10, "rid-1", 1, "tr", "au",
                       out, sizeof(out), &cls) != 0) {
        FAIL(name, "first"); nl_fabric_destroy(f); return;
    }
    nl_fabric_fail_next(f, "log");
    if (nl_fabric_call(f, "log", "write", "once", NULL, 10, "rid-1", 1, "tr", "au",
                       out, sizeof(out), &cls) != 0 ||
        strstr(out, "logged:once") == NULL) {
        FAIL(name, "idemp"); nl_fabric_destroy(f); return;
    }
    nl_fabric_fail_next(f, "log");
    if (nl_fabric_call(f, "log", "write", "once", NULL, 10, "other", 0, "tr", "au",
                       out, sizeof(out), &cls) != NL_FAB_ERR_RETRY ||
        cls != NL_FAIL_TRANSIENT) {
        FAIL(name, "no-idemp"); nl_fabric_destroy(f); return;
    }
    PASS(name);
    nl_fabric_destroy(f);
}

static void test_preserve_replace(void) {
    const char *name = "fabric: preserve state and rolling replace";
    NlFabric *f = boot(NULL);
    NlNsi *nsi;
    NlNsi *vec;
    char out[64];
    NlFailClass cls;
    if (!f) { FAIL(name, "boot"); return; }
    nsi = nl_nsi_load_path("schema/nsi/examples/log.nsi.json");
    vec = nl_nsi_load_path("schema/nsi/modules/vector2d.nsi.json");
    if (!nsi || !vec) {
        FAIL(name, "load"); nl_nsi_free(nsi); nl_nsi_free(vec);
        nl_fabric_destroy(f); return;
    }
    if (nl_fabric_call(f, "log", "write", "keep", NULL, 10, "rid-p", 1, "t", "a",
                       out, sizeof(out), &cls) != 0) {
        FAIL(name, "seed"); nl_nsi_free(nsi); nl_nsi_free(vec);
        nl_fabric_destroy(f); return;
    }
    nl_fabric_preserve_state(f, "log", 1);
    nl_fabric_crash(f, "log");
    nl_fabric_fail_next(f, "log");
    if (nl_fabric_call(f, "log", "write", "keep", NULL, 10, "rid-p", 1, "t", "a",
                       out, sizeof(out), &cls) != 0 ||
        strstr(out, "logged:keep") == NULL) {
        FAIL(name, "preserve"); nl_nsi_free(nsi); nl_nsi_free(vec);
        nl_fabric_destroy(f); return;
    }
    if (nl_fabric_replace(f, "log", nsi, nsi) != 0) {
        FAIL(name, "compat"); nl_nsi_free(nsi); nl_nsi_free(vec);
        nl_fabric_destroy(f); return;
    }
    if (nl_fabric_replace(f, "log", nsi, vec) != NL_FAB_ERR_POLICY) {
        FAIL(name, "breaking"); nl_nsi_free(nsi); nl_nsi_free(vec);
        nl_fabric_destroy(f); return;
    }
    PASS(name);
    nl_nsi_free(nsi);
    nl_nsi_free(vec);
    nl_fabric_destroy(f);
}

static void test_host_and_conformance(void) {
    const char *name = "fabric: posix host, inproc policy match, remote no caps";
    NlHost posix = nl_host_posix();
    NlHost inproc = nl_host_inproc();
    NlFabric *a;
    NlFabric *b;
    char oa[64];
    char ob[64];
    NlFailClass cls;
    int pid = 0;
    int fd;
    unsigned char ent[8];
    NlCap cap;
    if (strcmp(posix.name, "posix") != 0 || strcmp(inproc.name, "inproc") != 0) {
        FAIL(name, "names"); return;
    }
    if (!posix.spawn || !posix.thread_run || !posix.ipc_pair || !posix.clock_ns ||
        !posix.entropy || !posix.shm_alloc || !posix.file_open ||
        !posix.net_socket || !posix.device_open || !posix.credential_uid) {
        FAIL(name, "vtable"); return;
    }
    if (posix.spawn(NULL, NULL, &pid) != 0 || pid <= 0) {
        FAIL(name, "spawn"); return;
    }
    waitpid(pid, NULL, 0);
    if (posix.thread_run(NULL, NULL) != 0) {
        FAIL(name, "thread"); return;
    }
    if (posix.entropy(ent, sizeof(ent)) != 0 || posix.clock_ns() == 0) {
        FAIL(name, "clock"); return;
    }
    fd = posix.file_open("/dev/null", 0);
    if (fd < 0) { FAIL(name, "file"); return; }
    close(fd);
    fd = posix.net_socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) { FAIL(name, "net"); return; }
    close(fd);
    fd = posix.device_open("/dev/null");
    if (fd < 0) { FAIL(name, "device"); return; }
    close(fd);
    (void)posix.credential_uid();
    a = boot(&posix);
    b = boot(&inproc);
    if (!a || !b) {
        FAIL(name, "boot"); nl_fabric_destroy(a); nl_fabric_destroy(b); return;
    }
    if (nl_fabric_call(a, "log", "write", "hi", NULL, 10, "c", 1, "t", "a",
                       oa, sizeof(oa), &cls) != 0 ||
        nl_fabric_call(b, "log", "write", "hi", NULL, 10, "c", 1, "t", "a",
                       ob, sizeof(ob), &cls) != 0 ||
        strcmp(oa, ob) != 0) {
        FAIL(name, "policy"); nl_fabric_destroy(a); nl_fabric_destroy(b); return;
    }
    if (mint(a, "t", "nsi:nanolang/log", walker_rights(), NULL, &cap) != 0) {
        FAIL(name, "mint"); nl_fabric_destroy(a); nl_fabric_destroy(b); return;
    }
    if (nl_fabric_send_cap_remote(a, "remote.diag", &cap) != NL_FAB_ERR_REMOTE ||
        nl_fabric_send_cap_remote(a, "log", &cap) != 0) {
        FAIL(name, "remote"); nl_fabric_destroy(a); nl_fabric_destroy(b); return;
    }
    if (nl_fabric_has_ipc(a, "editor.walker") == nl_fabric_has_ipc(a, "log")) {
        FAIL(name, "isolated"); nl_fabric_destroy(a); nl_fabric_destroy(b); return;
    }
    PASS(name);
    nl_fabric_destroy(a);
    nl_fabric_destroy(b);
}

static void test_migrated_services(void) {
    const char *name = "fabric: migrated services with scoped caps";
    NlFabric *f = boot(NULL);
    NlCap fs, proc, net, audio, gfx, gpu, py, bad;
    char out[80];
    NlFailClass cls;
    if (!f) { FAIL(name, "boot"); return; }
    if (mint(f, "file", "nsi:nanolang/fs", walker_rights(), "/tmp", &fs) != 0 ||
        mint(f, "proc", "nsi:nanolang/process", walker_rights(), "exec", &proc) != 0 ||
        mint(f, "ep", "nsi:nanolang/net", walker_rights(), "tcp:53", &net) != 0 ||
        mint(f, "aud", "nsi:nanolang/audio", walker_rights(), "stream", &audio) != 0 ||
        mint(f, "gfx", "nsi:nanolang/graphics", walker_rights(), "surface", &gfx) != 0 ||
        mint(f, "gpu", "nsi:nanolang/gpu", walker_rights(), "queue", &gpu) != 0 ||
        mint(f, "py", "nsi:nanolang/python", walker_rights(), NULL, &py) != 0) {
        FAIL(name, "mint"); nl_fabric_destroy(f); return;
    }
    if (nl_fabric_call(f, "log", "write", "boot", NULL, 10, "l", 1, "t", "a",
                       out, sizeof(out), &cls) != 0 ||
        nl_fabric_call(f, "fs", "open", "/tmp/x", &fs, 10, "f", 1, "t", "a",
                       out, sizeof(out), &cls) != 0 ||
        nl_fabric_call(f, "fs", "open", "/etc/passwd", &fs, 10, "f2", 1, "t", "a",
                       out, sizeof(out), &cls) == 0 ||
        nl_fabric_call(f, "process", "spawn", "exec", &proc, 10, "p", 1, "t", "a",
                       out, sizeof(out), &cls) != 0 ||
        nl_fabric_call(f, "net", "dial", "tcp:53", &net, 10, "n", 1, "t", "a",
                       out, sizeof(out), &cls) != 0 ||
        nl_fabric_call(f, "net", "dial", "tcp:22", &net, 10, "n2", 1, "t", "a",
                       out, sizeof(out), &cls) == 0 ||
        nl_fabric_call(f, "audio", "play", "", &audio, 10, "a", 1, "t", "a",
                       out, sizeof(out), &cls) != 0 ||
        nl_fabric_call(f, "graphics", "surface", "", &gfx, 10, "g", 1, "t", "a",
                       out, sizeof(out), &cls) != 0 ||
        nl_fabric_call(f, "gpu", "submit", "", &gpu, 10, "u", 1, "t", "a",
                       out, sizeof(out), &cls) != 0 ||
        nl_fabric_call(f, "python", "eval", "1+1", &py, 10, "y", 1, "t", "a",
                       out, sizeof(out), &cls) != 0 ||
        nl_fabric_call(f, "python", "eval", "PyObject* 0x1", &py, 10, "y2", 1, "t", "a",
                       out, sizeof(out), &cls) == 0) {
        FAIL(name, "calls"); nl_fabric_destroy(f); return;
    }
    if (nl_cap_from_integer(nl_fabric_caps(f), 99, &bad) != NL_CAP_ERR_FORGED) {
        FAIL(name, "forge"); nl_fabric_destroy(f); return;
    }
    PASS(name);
    nl_fabric_destroy(f);
}

static void test_budgets_cancel(void) {
    const char *name = "fabric: quotas, cancel, accounting, flood";
    NlFabric *f = boot(NULL);
    char out[64];
    NlFailClass cls;
    NlAccounting acc;
    int i;
    int hit = 0;
    if (!f) { FAIL(name, "boot"); return; }
    if (nl_fabric_set_budget(f, "log", "queue", 3) != 0) {
        FAIL(name, "budget"); nl_fabric_destroy(f); return;
    }
    for (i = 0; i < 8; i++) {
        if (nl_fabric_call(f, "log", "write", "q", NULL, 10, "q", 0, "t", "a",
                           out, sizeof(out), &cls) == NL_FAB_ERR_QUOTA) {
            hit = 1;
            if (cls != NL_FAIL_QUOTA) {
                FAIL(name, "class"); nl_fabric_destroy(f); return;
            }
            break;
        }
    }
    if (!hit) {
        FAIL(name, "flood"); nl_fabric_destroy(f); return;
    }
    nl_fabric_cancel(f, "log");
    if (nl_fabric_call(f, "log", "write", "x", NULL, 10, "c", 0, "t", "a",
                       out, sizeof(out), &cls) != NL_FAB_ERR_CANCEL) {
        FAIL(name, "cancel"); nl_fabric_destroy(f); return;
    }
    if (nl_fabric_accounting(f, "log", &acc) != 0 || acc.queue < 1) {
        FAIL(name, "acct"); nl_fabric_destroy(f); return;
    }
    PASS(name);
    nl_fabric_destroy(f);
}

static void test_editor_client(void) {
    const char *name = "fabric: editor walker/freeze caps, quotas, kill, stale";
    NlFabric *f = boot(NULL);
    NlCap all, echo, freeze, stale;
    char out[80];
    NlFailClass cls;
    NlShm *region = NULL;
    size_t bound = 0;
    unsigned char blob[256];
    int gen;
    if (!f) { FAIL(name, "boot"); return; }
    memset(blob, 7, sizeof(blob));
    if (mint(f, "walk", "nsi:nanolang/editor.walker", walker_rights(), NULL, &all) != 0 ||
        mint(f, "echo", "nsi:nanolang/editor.walker", NL_CAP_ECHO | NL_CAP_DELEGATE,
             NULL, &echo) != 0 ||
        mint(f, "frz", "nsi:nanolang/editor.freeze", NL_CAP_EVAL | NL_CAP_DELEGATE,
             NULL, &freeze) != 0) {
        FAIL(name, "mint"); nl_fabric_destroy(f); return;
    }
    if (nl_fabric_eval(f, &all, "1", 100, out, sizeof(out)) != 0 ||
        nl_fabric_bind_buffer(f, &all, "buf") != 0 ||
        nl_fabric_call(f, "editor.walker", "echo", "ping", &all, 10, "e", 1, "t", "a",
                       out, sizeof(out), &cls) != 0 ||
        nl_fabric_call(f, "editor.walker", "chrome", "mode", &all, 10, "ch", 1, "t", "a",
                       out, sizeof(out), &cls) != 0) {
        FAIL(name, "walker"); nl_fabric_destroy(f); return;
    }
    if (nl_fabric_eval(f, &echo, "1", 100, out, sizeof(out)) == 0) {
        FAIL(name, "eval-rights"); nl_fabric_destroy(f); return;
    }
    if (nl_fabric_freeze(f, &freeze, "module", out, sizeof(out)) != 0) {
        FAIL(name, "freeze"); nl_fabric_destroy(f); return;
    }
    if (nl_fabric_call(f, "editor.freeze", "ed_bind", "x", &freeze, 10, "ed", 0, "t", "a",
                       out, sizeof(out), &cls) == 0) {
        FAIL(name, "ed_*"); nl_fabric_destroy(f); return;
    }
    if (nl_fabric_bind_large(f, &all, blob, sizeof(blob), 1, &region, &bound) != 0 ||
        bound != sizeof(blob) || !region || !nl_shm_copy_fallback(region)) {
        FAIL(name, "copy-bind"); nl_shm_destroy(region);
        nl_fabric_destroy(f); return;
    }
    nl_shm_destroy(region);
    region = NULL;
    if (nl_fabric_bind_large(f, &all, blob, sizeof(blob), 0, &region, &bound) != 0 ||
        bound != sizeof(blob)) {
        FAIL(name, "shm-bind"); nl_shm_destroy(region);
        nl_fabric_destroy(f); return;
    }
    nl_shm_destroy(region);
    nl_fabric_set_hung(f, "editor.walker", 1);
    if (nl_fabric_eval(f, &all, "hang", 10, out, sizeof(out)) != NL_FAB_ERR_DEADLINE) {
        FAIL(name, "deadline"); nl_fabric_destroy(f); return;
    }
    if (!nl_fabric_ready(f, "log") ||
        nl_fabric_call(f, "log", "write", "frame", NULL, 10, "fr", 1, "t", "a",
                       out, sizeof(out), &cls) != 0) {
        FAIL(name, "frame"); nl_fabric_destroy(f); return;
    }
    nl_fabric_cancel(f, "editor.walker");
    stale = all;
    gen = nl_fabric_generation(f, "editor.walker");
    nl_fabric_crash(f, "editor.walker");
    if (nl_fabric_eval(f, &stale, "1", 100, out, sizeof(out)) != NL_FAB_ERR_STALE) {
        FAIL(name, "stale-walker"); nl_fabric_destroy(f); return;
    }
    if (nl_fabric_generation(f, "editor.walker") <= gen) {
        FAIL(name, "gen"); nl_fabric_destroy(f); return;
    }
    if (!nl_fabric_walker_alive(f) ||
        nl_fabric_call(f, "log", "write", "up", NULL, 10, "up", 1, "t", "a",
                       out, sizeof(out), &cls) != 0) {
        FAIL(name, "survive-walker"); nl_fabric_destroy(f); return;
    }
    if (mint(f, "walk2", "nsi:nanolang/editor.walker", walker_rights(), NULL, &all) != 0 ||
        nl_fabric_eval(f, &all, "1", 100, out, sizeof(out)) != 0) {
        FAIL(name, "remint"); nl_fabric_destroy(f); return;
    }
    {
        NlAccounting wacc;
        if (nl_fabric_accounting(f, "editor.walker", &wacc) != 0) {
            FAIL(name, "wacc"); nl_fabric_destroy(f); return;
        }
        if (nl_fabric_set_budget(f, "editor.walker", "queue", wacc.queue) != 0 ||
            nl_fabric_eval(f, &all, "1", 100, out, sizeof(out)) != NL_FAB_ERR_QUOTA) {
            FAIL(name, "eval-quota"); nl_fabric_destroy(f); return;
        }
        if (nl_fabric_set_budget(f, "editor.walker", "queue", 128) != 0) {
            FAIL(name, "restore-q"); nl_fabric_destroy(f); return;
        }
    }
    nl_fabric_crash(f, "editor.freeze");
    if (nl_fabric_freeze(f, &freeze, "module", out, sizeof(out)) != NL_FAB_ERR_STALE) {
        FAIL(name, "stale-freeze"); nl_fabric_destroy(f); return;
    }
    if (!nl_fabric_freeze_alive(f) ||
        nl_fabric_call(f, "log", "write", "still", NULL, 10, "st", 1, "t", "a",
                       out, sizeof(out), &cls) != 0) {
        FAIL(name, "survive-freeze"); nl_fabric_destroy(f); return;
    }
    PASS(name);
    nl_fabric_destroy(f);
}

int main(void) {
    printf("NSI fabric tests\n");
    test_supervisor();
    test_restart_policies();
    test_fail_classes_retry();
    test_preserve_replace();
    test_host_and_conformance();
    test_migrated_services();
    test_budgets_cancel();
    test_editor_client();
    printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}

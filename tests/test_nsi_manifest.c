#include "nsi_manifest.h"
#include "nsi.h"
#include "cJSON.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int g_pass = 0;
static int g_fail = 0;

#define PASS(name) do { g_pass++; printf("  %-60s PASS\n", (name)); } while (0)
#define FAIL(name, msg) do { g_fail++; printf("  %-60s FAIL: %s\n", (name), (msg)); } while (0)

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    long sz;
    char *buf;
    size_t n;
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return NULL; }
    sz = ftell(fp);
    rewind(fp);
    buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(fp); return NULL; }
    n = fread(buf, 1, (size_t)sz, fp);
    fclose(fp);
    buf[n] = '\0';
    return buf;
}

static void test_log_manifest(void) {
    const char *test_name = "nsi_manifest: log portable nsi vs module.json build";
    NlNsiManifest *m = nl_nsi_manifest_load("modules/std/log/module.manifest.json");
    if (!m) { FAIL(test_name, "load"); return; }
    if (strcmp(nl_nsi_manifest_interface_id(m), "nsi:nanolang/log") != 0) {
        FAIL(test_name, "id"); nl_nsi_manifest_free(m); return;
    }
    if (!m->build_path || !strstr(m->build_path, "module.json")) {
        FAIL(test_name, "build"); nl_nsi_manifest_free(m); return;
    }
    if (strcmp(m->isolation, "in-process") != 0 || strcmp(m->adapter, "inproc") != 0) {
        FAIL(test_name, "policy"); nl_nsi_manifest_free(m); return;
    }
    PASS(test_name);
    nl_nsi_manifest_free(m);
}

static void test_unknown_isolation(void) {
    const char *test_name = "nsi_manifest: unknown isolation fails closed";
    const char *path = "/tmp/nl_nsi_bad_manifest.json";
    FILE *fp = fopen(path, "w");
    NlNsiManifest *m;
    if (!fp) { FAIL(test_name, "write"); return; }
    fprintf(fp, "{\"name\":\"x\",\"nsi\":{\"interface_id\":\"nsi:nanolang/x\","
                "\"interface_version\":\"0\",\"schema\":\"s\","
                "\"required_capabilities\":[],\"isolation\":\"windows-service\","
                "\"resource_budgets\":{\"queue\":1,\"memory_bytes\":1},"
                "\"restart\":\"never\",\"adapter\":\"inproc\"}}");
    fclose(fp);
    m = nl_nsi_manifest_load(path);
    unlink(path);
    if (m) { FAIL(test_name, "accepted"); nl_nsi_manifest_free(m); return; }
    PASS(test_name);
}

static void test_shared_interface(void) {
    const char *test_name = "nsi_manifest: graphics contract independent of impl";
    NlNsiManifest *sdl = nl_nsi_manifest_load("modules/sdl/module.manifest.json");
    NlNsiManifest *glfw = nl_nsi_manifest_load("modules/glfw/module.manifest.json");
    if (!sdl || !glfw) { FAIL(test_name, "load"); nl_nsi_manifest_free(sdl); nl_nsi_manifest_free(glfw); return; }
    if (strcmp(sdl->interface_id, glfw->interface_id) != 0 ||
        strcmp(sdl->interface_id, "nsi:nanolang/graphics") != 0) {
        FAIL(test_name, "shared"); nl_nsi_manifest_free(sdl); nl_nsi_manifest_free(glfw); return;
    }
    PASS(test_name);
    nl_nsi_manifest_free(sdl);
    nl_nsi_manifest_free(glfw);
}

static void test_inventory(void) {
    const char *test_name = "nsi_manifest: inventory covers manifests";
    char *raw = read_file("schema/nsi/inventory.json");
    cJSON *root;
    cJSON *mods;
    int n;
    int i;
    if (!raw) { FAIL(test_name, "read"); return; }
    root = cJSON_Parse(raw);
    free(raw);
    if (!root) { FAIL(test_name, "json"); return; }
    mods = cJSON_GetObjectItemCaseSensitive(root, "modules");
    if (!mods || !cJSON_IsArray(mods)) { FAIL(test_name, "array"); cJSON_Delete(root); return; }
    n = cJSON_GetArraySize(mods);
    if (n < 47) { FAIL(test_name, "count"); cJSON_Delete(root); return; }
    for (i = 0; i < n; i++) {
        cJSON *it = cJSON_GetArrayItem(mods, i);
        cJSON *path = cJSON_GetObjectItemCaseSensitive(it, "path");
        cJSON *iface = cJSON_GetObjectItemCaseSensitive(it, "interface_id");
        cJSON *priv = cJSON_GetObjectItemCaseSensitive(it, "privilege");
        FILE *fp;
        if (!path || !cJSON_IsString(path) || !iface || !priv) {
            FAIL(test_name, "fields"); cJSON_Delete(root); return;
        }
        fp = fopen(path->valuestring, "rb");
        if (!fp) { FAIL(test_name, path->valuestring); cJSON_Delete(root); return; }
        fclose(fp);
    }
    cJSON_Delete(root);
    PASS(test_name);
}

static void test_every_manifest_nsi(void) {
    const char *test_name = "nsi_manifest: every module.manifest.json has nsi";
    const char *paths[] = {
        "modules/std/log/module.manifest.json",
        "modules/vector2d/module.manifest.json",
        "modules/filesystem/module.manifest.json",
        "modules/pty/module.manifest.json",
        "modules/curl/module.manifest.json",
        "modules/sdl_mixer/module.manifest.json",
        "modules/sdl/module.manifest.json",
        "modules/glew/module.manifest.json",
        "modules/pybridge/module.manifest.json",
        "modules/math_ext/module.manifest.json",
        0
    };
    int i;
    for (i = 0; paths[i]; i++) {
        NlNsiManifest *m = nl_nsi_manifest_load(paths[i]);
        NlNsi *nsi;
        if (!m) { FAIL(test_name, paths[i]); return; }
        nsi = nl_nsi_load_path(m->schema_path);
        if (!nsi) { FAIL(test_name, m->schema_path); nl_nsi_manifest_free(m); return; }
        if (strcmp(nl_nsi_interface_id(nsi), m->interface_id) != 0) {
            FAIL(test_name, "iface-mismatch"); nl_nsi_free(nsi); nl_nsi_manifest_free(m); return;
        }
        nl_nsi_free(nsi);
        nl_nsi_manifest_free(m);
    }
    PASS(test_name);
}

int main(void) {
    test_log_manifest();
    test_unknown_isolation();
    test_shared_interface();
    test_inventory();
    test_every_manifest_nsi();
    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}

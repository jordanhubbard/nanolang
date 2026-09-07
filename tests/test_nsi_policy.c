#include "nsi_policy.h"
#include "nsi_cap.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_pass = 0;
static int g_fail = 0;

#define PASS(name) do { g_pass++; printf("  %-60s PASS\n", (name)); } while (0)
#define FAIL(name, msg) do { g_fail++; printf("  %-60s FAIL: %s\n", (name), (msg)); } while (0)

static void test_layers(void) {
    const char *test_name = "policy: effect map names five layers";
    NlEffectMap *m = nl_effect_map_load("schema/nsi/effect_map.v0.json");
    if (!m) { FAIL(test_name, "load"); return; }
    if (!nl_effect_map_has_layer(m, "source_effect") ||
        !nl_effect_map_has_layer(m, "module_requirement") ||
        !nl_effect_map_has_layer(m, "nanoisa_trap") ||
        !nl_effect_map_has_layer(m, "nsi_method") ||
        !nl_effect_map_has_layer(m, "capability")) {
        FAIL(test_name, "layers");
        nl_effect_map_free(m);
        return;
    }
    PASS(test_name);
    nl_effect_map_free(m);
}

static void test_inventory(void) {
    const char *test_name = "policy: IO inventory lists trap, method, cap";
    const char *fx[] = { "IO" };
    NlEffectMap *m = nl_effect_map_load("schema/nsi/effect_map.v0.json");
    NlEffectInventory inv;
    char *json;
    if (!m) { FAIL(test_name, "load"); return; }
    if (nl_effect_inventory_from_rows(m, fx, 1, &inv) != NL_POLICY_OK) {
        FAIL(test_name, "inventory");
        nl_effect_map_free(m);
        return;
    }
    if (inv.effect_n != 1 || (inv.required_rights & (NL_CAP_READ | NL_CAP_WRITE)) !=
        (NL_CAP_READ | NL_CAP_WRITE)) {
        FAIL(test_name, "rights");
        nl_effect_map_free(m);
        return;
    }
    json = nl_effect_inventory_json(&inv);
    if (!json || !strstr(json, "TRAP_PRINT") || !strstr(json, "nsi:nanolang/log#write") ||
        !strstr(json, "cap:nanolang/log.write")) {
        FAIL(test_name, "json");
        free(json);
        nl_effect_map_free(m);
        return;
    }
    free(json);
    PASS(test_name);
    nl_effect_map_free(m);
}

static void test_state_no_cap(void) {
    const char *test_name = "policy: State does not require a host capability";
    const char *fx[] = { "State" };
    NlEffectMap *m = nl_effect_map_load("schema/nsi/effect_map.v0.json");
    NlEffectInventory inv;
    if (!m) { FAIL(test_name, "load"); return; }
    if (nl_effect_inventory_from_rows(m, fx, 1, &inv) != NL_POLICY_OK) {
        FAIL(test_name, "inventory");
        nl_effect_map_free(m);
        return;
    }
    if (inv.required_rights != 0 || inv.cap_n != 0) {
        FAIL(test_name, "state-cap");
        nl_effect_map_free(m);
        return;
    }
    PASS(test_name);
    nl_effect_map_free(m);
}

static void test_reject_uncovered(void) {
    const char *test_name = "policy: uncovered grants fail closed";
    const char *fx[] = { "IO" };
    const char *grants[] = { "cap:nanolang/log.write" };
    NlEffectMap *m = nl_effect_map_load("schema/nsi/effect_map.v0.json");
    NlEffectInventory inv;
    if (!m) { FAIL(test_name, "load"); return; }
    nl_effect_inventory_from_rows(m, fx, 1, &inv);
    if (nl_deploy_check(&inv, NL_CAP_WRITE, grants, 1, 0) != NL_POLICY_ERR_UNCOVERED) {
        FAIL(test_name, "should reject missing READ and filesystem cap");
        nl_effect_map_free(m);
        return;
    }
    PASS(test_name);
    nl_effect_map_free(m);
}

static void test_accept_cover(void) {
    const char *test_name = "policy: covering grants accepted";
    const char *fx[] = { "IO" };
    const char *grants[] = { "cap:nanolang/log.write", "cap:nanolang/filesystem.open" };
    NlEffectMap *m = nl_effect_map_load("schema/nsi/effect_map.v0.json");
    NlEffectInventory inv;
    char *man;
    if (!m) { FAIL(test_name, "load"); return; }
    nl_effect_inventory_from_rows(m, fx, 1, &inv);
    if (nl_deploy_check(&inv, NL_CAP_READ | NL_CAP_WRITE, grants, 2, 0) != NL_POLICY_OK) {
        FAIL(test_name, "cover");
        nl_effect_map_free(m);
        return;
    }
    man = nl_deploy_manifest_json(&inv, NL_CAP_READ | NL_CAP_WRITE, grants, 2, 0, "");
    if (!man || !strstr(man, "\"accepted\":true") || !strstr(man, "inventory")) {
        FAIL(test_name, "manifest");
        free(man);
        nl_effect_map_free(m);
        return;
    }
    free(man);
    PASS(test_name);
    nl_effect_map_free(m);
}

static void test_unused_and_override(void) {
    const char *test_name = "policy: unused grants reported; override does not widen source";
    const char *fx[] = { "IO" };
    const char *grants[] = {
        "cap:nanolang/log.write",
        "cap:nanolang/filesystem.open",
        "cap:nanolang/extra"
    };
    NlEffectMap *m = nl_effect_map_load("schema/nsi/effect_map.v0.json");
    NlEffectInventory inv;
    NlEffectInventory after;
    char *man;
    if (!m) { FAIL(test_name, "load"); return; }
    nl_effect_inventory_from_rows(m, fx, 1, &inv);
    if (nl_deploy_unused_count(&inv, grants, 3) != 1) {
        FAIL(test_name, "unused");
        nl_effect_map_free(m);
        return;
    }
    man = nl_deploy_manifest_json(&inv, NL_CAP_READ, grants, 3, 1, "admin");
    if (!man || !strstr(man, "\"override\":true") ||
        !strstr(man, "source_declarations_widened\":false") ||
        !strstr(man, "\"accepted\":true")) {
        FAIL(test_name, "override-manifest");
        free(man);
        nl_effect_map_free(m);
        return;
    }
    free(man);
    after = inv;
    if (nl_deploy_source_widened(&inv, &after)) {
        FAIL(test_name, "widened");
        nl_effect_map_free(m);
        return;
    }
    PASS(test_name);
    nl_effect_map_free(m);
}

int main(void) {
    printf("NSI effect/policy tests\n");
    test_layers();
    test_inventory();
    test_state_no_cap();
    test_reject_uncovered();
    test_accept_cover();
    test_unused_and_override();
    printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}

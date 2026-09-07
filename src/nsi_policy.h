#ifndef NL_NSI_POLICY_H
#define NL_NSI_POLICY_H

#include <stddef.h>
#include <stdint.h>

#define NL_POLICY_OK 0
#define NL_POLICY_ERR 1
#define NL_POLICY_ERR_UNCOVERED 2
#define NL_POLICY_MAX_ROWS 32
#define NL_POLICY_MAX_NAME 80

typedef struct {
    char effect[16];
    char op[16];
    char trap[32];
    char method[NL_POLICY_MAX_NAME];
    char capability[64];
    uint32_t rights;
} NlEffectMapRow;

typedef struct {
    char layers[8][32];
    int layer_n;
    NlEffectMapRow rows[NL_POLICY_MAX_ROWS];
    int row_n;
} NlEffectMap;

typedef struct {
    char effects[NL_POLICY_MAX_ROWS][16];
    int effect_n;
    char traps[NL_POLICY_MAX_ROWS][32];
    int trap_n;
    char methods[NL_POLICY_MAX_ROWS][NL_POLICY_MAX_NAME];
    int method_n;
    char cap_ids[NL_POLICY_MAX_ROWS][64];
    int cap_n;
    uint32_t required_rights;
} NlEffectInventory;

NlEffectMap *nl_effect_map_load(const char *path);
void nl_effect_map_free(NlEffectMap *m);
int nl_effect_map_has_layer(const NlEffectMap *m, const char *layer);

int nl_effect_inventory_from_rows(const NlEffectMap *map,
                                  const char **effects, int n,
                                  NlEffectInventory *out);
char *nl_effect_inventory_json(const NlEffectInventory *inv);

char *nl_deploy_manifest_json(const NlEffectInventory *inv,
                              uint32_t granted_rights,
                              const char **granted_caps, int granted_n,
                              int override, const char *override_reason);

int nl_deploy_check(const NlEffectInventory *inv, uint32_t granted_rights,
                    const char **granted_caps, int granted_n, int override);
int nl_deploy_unused_count(const NlEffectInventory *inv,
                           const char **granted_caps, int granted_n);
int nl_deploy_source_widened(const NlEffectInventory *before,
                             const NlEffectInventory *after);

#endif

#ifndef NL_NSI_MANIFEST_H
#define NL_NSI_MANIFEST_H

#include <stddef.h>

/* Portable NSI block from module.manifest.json. Build metadata stays in
 * module.json (c_sources, headers). Unknown isolation/restart/adapter
 * values fail closed. */

typedef struct {
    char *interface_id;
    char *interface_version;
    char *schema_path;
    char **required_capabilities;
    size_t cap_count;
    char *isolation;
    int queue_budget;
    int memory_bytes;
    char *restart;
    char *adapter;
    char *build_path;
} NlNsiManifest;

void nl_nsi_manifest_free(NlNsiManifest *m);
NlNsiManifest *nl_nsi_manifest_load(const char *manifest_path);
const char *nl_nsi_manifest_interface_id(const NlNsiManifest *m);

#endif

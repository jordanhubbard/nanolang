#ifndef NANOLANG_MODULE_LINK_RESPONSE_H
#define NANOLANG_MODULE_LINK_RESPONSE_H

#include "module_builder.h"

/* Internal graph-capture boundary. A caller must identify its selected linker
 * before choosing a grammar; capture alone does not authorize cache reuse. */
typedef enum { MODULE_LINK_RESPONSE_GNU = 1, MODULE_LINK_RESPONSE_APPLE = 2 } ModuleLinkResponseGrammar;

/* I return an owned path to a retained response graph, or NULL on failure.
 * The caller frees the path string; retained files live with the module cache. */
char *module_capture_link_response(const ModuleBuildMetadata *meta, const char *source,
                                   ModuleLinkResponseGrammar grammar);

#endif

#ifndef NANOLANG_REFLECTION_H
#define NANOLANG_REFLECTION_H

#include "nanolang.h"
#include <stdbool.h>

/* Module reflection - output module exports as JSON for documentation */
bool emit_module_reflection(const char *output_path, ASTNode *program, Environment *env, const char *module_name);

#endif /* NANOLANG_REFLECTION_H */

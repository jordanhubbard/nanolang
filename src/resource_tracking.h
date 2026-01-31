#ifndef NANOLANG_RESOURCE_TRACKING_H
#define NANOLANG_RESOURCE_TRACKING_H

#include "nanolang.h"

/* Check if a struct type is a resource type */
bool is_resource_type(Environment *env, const char *struct_name);

/* Mark a variable as a resource if its type is a resource struct */
void mark_variable_as_resource_if_needed(Environment *env, const char *var_name, const char *struct_type_name);

/* Check resource usage and update state */
void check_resource_use(Environment *env, const char *var_name, int line, int column, bool *has_error);

/* Check resource consumption (ownership transfer) */
void check_resource_consume(Environment *env, const char *var_name, int line, int column, bool *has_error);

/* Check for resource leaks at end of scope */
void check_resource_leaks(Environment *env, bool *has_error);

#endif /* NANOLANG_RESOURCE_TRACKING_H */


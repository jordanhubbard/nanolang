#ifndef STDLIB_RUNTIME_H
#define STDLIB_RUNTIME_H

/* StringBuilder structure (shared with transpiler.c) */
typedef struct {
    char *buffer;
    int length;
    int capacity;
} StringBuilder;

/* StringBuilder functions */
StringBuilder *sb_create(void);
void sb_append(StringBuilder *sb, const char *str);

/* Generate standard library runtime functions */
void generate_string_operations(StringBuilder *sb);
void generate_file_operations(StringBuilder *sb);
void generate_dir_operations(StringBuilder *sb);
void generate_path_operations(StringBuilder *sb);
void generate_math_utility_builtins(StringBuilder *sb);

/* Generate complete stdlib runtime (calls all above functions) */
void generate_stdlib_runtime(StringBuilder *sb);

#endif /* STDLIB_RUNTIME_H */


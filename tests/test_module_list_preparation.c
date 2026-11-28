/* I compile literals and schema lists without unused global generators. */
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE
#endif
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../src/nanolang.h"
#include "../src/runtime/gc.h"
#include <errno.h>
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>

int g_argc;
char **g_argv;
static char project_root[PATH_MAX];
const char *get_project_root(void) { return project_root; }

static void put(const char *path, const char *text) {
    FILE *file = fopen(path, "w");
    assert(file && fputs(text, file) >= 0 && fclose(file) == 0);
}

int main(int argc, char **argv) {
    assert(argc == 2 && realpath(argv[1], project_root));
    assert(setvbuf(stdout, NULL, _IONBF, 0) == 0);
    char cwd[PATH_MAX];
    assert(getcwd(cwd, sizeof cwd));
    const char *temporary = getenv("TMPDIR");
    if (!temporary || !*temporary) temporary = "/tmp";
    char work[PATH_MAX];
    int length = snprintf(work, sizeof work, "%s/nano-list-preparation-XXXXXX", temporary);
    assert(length > 0 && (size_t)length < sizeof work && mkdtemp(work));
    assert(chdir(work) == 0 && mkdir("scripts", 0700) == 0);
    char source_modules[PATH_MAX];
    length = snprintf(source_modules, sizeof source_modules, "%s/src_nano", project_root);
    assert(length > 0 && (size_t)length < sizeof source_modules);
    assert(symlink(source_modules, "src_nano") == 0);
    char marker[PATH_MAX];
    length = snprintf(marker, sizeof marker, "%s/generator-called", work);
    assert(length > 0 && (size_t)length < sizeof marker);
    assert(setenv("NANO_MODULE_LIST_MARKER", marker, 1) == 0);
    put("scripts/generate_list.sh",
        "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$NANO_MODULE_LIST_MARKER\"\nexit 97\n");
    assert(chmod("scripts/generate_list.sh", 0700) == 0);
    put("inline.nano", "pub fn label() -> string { return \"List_UnusedMarker \" }\nshadow label { assert (== (label) \"List_UnusedMarker \") }\n");
    put("schema.nano", "import \"src_nano/generated/compiler_ast.nano\"\npub fn schema_probe(items: List<LexerToken>) -> int { return (list_LexerToken_length items) }\nshadow schema_probe {\n let items: List<LexerToken> = (list_LexerToken_new)\n assert (== (schema_probe items) 0)\n (list_LexerToken_free items)\n}\n");
    gc_init();
    Environment *env = create_environment();
    assert(env);
    printf("I retain my module-list preparation fixture at %s.\n", work);
    assert(compile_module_to_object("inline.nano", "inline.o", env, false, NULL, 0));
    struct stat object;
    assert(stat("inline.o", &object) == 0 && object.st_size > 0);
    assert(compile_module_to_object("schema.nano", "schema.o", env, false, NULL, 0));
    assert(stat("schema.o", &object) == 0 && object.st_size > 0);
    puts("I compile literal text and actual schema lists when the unused generator refuses.");
    put("rejected.nano", "pub fn broken() -> int { return missing_value }\n");
    put("rejected.o", "retained object bytes\n");
    assert(!compile_module_to_object("rejected.nano", "rejected.o", env, false, NULL, 0));
    FILE *prior = fopen("rejected.o", "r");
    char retained[64] = {0};
    assert(prior && fgets(retained, sizeof retained, prior) && fclose(prior) == 0);
    assert(strcmp(retained, "retained object bytes\n") == 0);
    errno = 0;
    assert(access(marker, F_OK) != 0 && errno == ENOENT);
    free_environment(env);
    clear_module_cache();
    gc_shutdown();
    puts("I preserve prior output and schema list declarations without launching a generator.");
    if (!getenv("NANO_KEEP_LIST_PREPARATION")) {
        assert(unlink("scripts/generate_list.sh") == 0 && rmdir("scripts") == 0);
        assert(unlink("inline.nano") == 0 && unlink("inline.o") == 0);
        assert(unlink("schema.nano") == 0 && unlink("schema.o") == 0);
        assert(unlink("src_nano") == 0);
        assert(unlink("rejected.nano") == 0 && unlink("rejected.o") == 0);
        assert(chdir(cwd) == 0 && rmdir(work) == 0);
    }
    return 0;
}

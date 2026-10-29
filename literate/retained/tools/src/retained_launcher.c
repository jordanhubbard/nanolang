/* I am the one native entry point behind build-native, host-make and run-cases.
 *
 * Literate AI accepts only hashed native binaries as toolchain commands. The real tools are installed on the
 * host, so before I run anything I re-hash every file named in pins.txt (next to me) and refuse on any
 * difference. I do not make the toolchain hermetic: I make a changed host toolchain fail closed.
 *
 * pins.txt lines: "PATH<TAB>value" sets the only PATH I pass on; "<sha256><TAB>/absolute/path" pins one file. */
#include <CommonCrypto/CommonDigest.h>
#include <libgen.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#define EXIT_PIN 70

static int hex_sha256(const char *path, char out[65]) {
    FILE *file = fopen(path, "rb");
    if (!file) return -1;
    CC_SHA256_CTX ctx;
    CC_SHA256_Init(&ctx);
    unsigned char buffer[65536];
    size_t count;
    while ((count = fread(buffer, 1, sizeof buffer, file)) > 0) CC_SHA256_Update(&ctx, buffer, (CC_LONG)count);
    int failed = ferror(file);
    fclose(file);
    if (failed) return -1;
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(digest, &ctx);
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) snprintf(out + 2 * i, 3, "%02x", digest[i]);
    return 0;
}

static int run(char *const argv[]) {
    pid_t pid = fork();
    if (pid < 0) return 127;
    if (pid == 0) {
        execv(argv[0], argv);
        _exit(127);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) return 127;
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}

int main(int argc, char **argv) {
    char self[PATH_MAX], resolved[PATH_MAX];
    uint32_t size = sizeof self;
    if (_NSGetExecutablePath(self, &size) != 0 || !realpath(self, resolved)) {
        fprintf(stderr, "launcher: cannot locate myself\n");
        return EXIT_PIN;
    }
    char copy[PATH_MAX], directory[PATH_MAX], name[PATH_MAX];
    snprintf(copy, sizeof copy, "%s", resolved);
    snprintf(directory, sizeof directory, "%s", dirname(copy));
    snprintf(copy, sizeof copy, "%s", resolved);
    snprintf(name, sizeof name, "%s", basename(copy));

    char pins_path[PATH_MAX];
    snprintf(pins_path, sizeof pins_path, "%s/pins.txt", directory);
    FILE *pins = fopen(pins_path, "r");
    if (!pins) {
        fprintf(stderr, "launcher: no pins.txt beside %s\n", name);
        return EXIT_PIN;
    }
    char line[PATH_MAX + 80], path_value[PATH_MAX * 4] = "";
    int checked = 0;
    while (fgets(line, sizeof line, pins)) {
        line[strcspn(line, "\n")] = 0;
        char *tab = strchr(line, '\t');
        if (!tab) continue;
        *tab = 0;
        if (strcmp(line, "PATH") == 0) {
            snprintf(path_value, sizeof path_value, "%s", tab + 1);
            continue;
        }
        char actual[65];
        if (hex_sha256(tab + 1, actual) != 0 || strcmp(actual, line) != 0) {
            fprintf(stderr, "launcher: toolchain pin mismatch for %s\n  pinned %s\n  actual %s\n", tab + 1, line,
                    hex_sha256(tab + 1, actual) == 0 ? actual : "(unreadable)");
            fclose(pins);
            return EXIT_PIN;
        }
        checked++;
    }
    fclose(pins);
    if (checked == 0 || path_value[0] == 0) {
        fprintf(stderr, "launcher: pins.txt pins nothing or sets no PATH\n");
        return EXIT_PIN;
    }
    setenv("PATH", path_value, 1);

    if (strcmp(name, "host-make") == 0) {
        char *args[argc + 2];
        args[0] = "/usr/bin/make";
        for (int i = 1; i < argc; i++) args[i] = argv[i];
        args[argc] = NULL;
        return run(args);
    }
    if (strcmp(name, "build-native") == 0) {
        /* The input tree is read-only and my working directory is the writable object root. I copy the tree
         * here once, preserving modes and times so make sees it as up to date, make it writable, build in
         * place, and export bin. Later actions (run-cases, host-make) run in this same directory. */
        if (argc != 3) {
            fprintf(stderr, "usage: build-native INPUT_ROOT OUTPUT_ROOT\n");
            return 2;
        }
        char source[PATH_MAX], destination[PATH_MAX];
        snprintf(source, sizeof source, "%s/.", argv[1]);
        char *copy_tree[] = {"/bin/cp", "-Rp", source, ".", NULL};
        int code = run(copy_tree);
        if (code != 0) return code;
        char *writable[] = {"/bin/chmod", "-R", "u+w", ".", NULL};
        code = run(writable);
        if (code != 0) return code;
        char *make[] = {"/usr/bin/make", "build", NULL};
        code = run(make);
        if (code != 0) return code;
        snprintf(destination, sizeof destination, "%s/bin", argv[2]);
        char *copy_out[] = {"/bin/cp", "-R", "bin", destination, NULL};
        return run(copy_out);
    }
    if (strcmp(name, "run-cases") == 0) {
        char script[PATH_MAX];
        snprintf(script, sizeof script, "%s/run-cases.py", directory);
        char *args[argc + 3];
        args[0] = "python3";
        args[1] = script;
        for (int i = 1; i < argc; i++) args[i + 1] = argv[i];
        args[argc + 1] = NULL;
        execvp("python3", args);
        fprintf(stderr, "launcher: cannot execute python3\n");
        return 127;
    }
    fprintf(stderr, "launcher: unknown tool name %s\n", name);
    return 2;
}

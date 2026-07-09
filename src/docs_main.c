/*
 * docs_main.c — nano-docs: nanolang documentation search CLI
 *
 * Searches across markdown docs, userguide, and module source files
 * for functions, types, examples, and explanatory text.
 *
 * Usage:
 *   nano-docs <query>                     — search everything
 *   nano-docs --fn <name>                 — search function signatures
 *   nano-docs --type <name>               — search type/struct/enum definitions
 *   nano-docs --module <mod>              — search within a specific module
 *   nano-docs --docs                      — search docs/ and userguide/ only
 *   nano-docs --context <n>               — show N lines of context (default: 3)
 *   nano-docs --list-fns                  — list all public functions
 *   nano-docs --list-modules              — list all modules
 *
 * Search sources:
 *   docs/          — reference docs (.md)
 *   userguide/     — tutorial and language guide (.md)
 *   modules/       — stdlib modules (.nano) — doc comments + pub fn signatures
 *   examples/      — example programs (.nano)
 *
 * Output format:
 *   FILE:LINE: matching line
 *   [context lines...]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdarg.h>
#include <dirent.h>
#include <sys/stat.h>
#include <fnmatch.h>
#include <ctype.h>

#define VERSION "0.1.0"
#define MAX_RESULTS 500
#define MAX_LINE    4096
#define MAX_PATH    1024

/* ── ANSI colour helpers ─────────────────────────────────────────────── */
static bool g_color = true;
#define COL_FILE    "\033[1;36m"
#define COL_MATCH   "\033[1;33m"
#define COL_CTX     "\033[2m"
#define COL_RESET   "\033[0m"
#define COL_SECTION "\033[1;34m"

static void colprintf(const char *col, const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    if (g_color) fputs(col, stdout);
    vprintf(fmt, ap);
    if (g_color) fputs(COL_RESET, stdout);
    va_end(ap);
}

/* ── Search options ──────────────────────────────────────────────────── */
typedef struct {
    const char *query;          /* Raw query string */
    bool case_insensitive;      /* -i: ignore case */
    bool fn_only;               /* --fn: only match pub fn / fn declarations */
    bool type_only;             /* --type: only match struct/enum/type definitions */
    bool docs_only;             /* --docs: only search .md files */
    bool modules_only;          /* --module: only search modules/ */
    const char *module_filter;  /* --module <name>: restrict to module */
    int  context_lines;         /* --context N: lines of context */
    bool list_fns;              /* --list-fns */
    bool list_modules;          /* --list-modules */
    bool verbose;
} DocOpts;

/* ── Result ──────────────────────────────────────────────────────────── */
typedef struct {
    char path[MAX_PATH];
    int  line;
    char text[MAX_LINE];
} DocResult;

static DocResult g_results[MAX_RESULTS];
static int       g_result_count = 0;

static void add_result(const char *path, int line, const char *text) {
    if (g_result_count >= MAX_RESULTS) return;
    DocResult *r = &g_results[g_result_count++];
    strncpy(r->path, path, MAX_PATH - 1);
    r->line = line;
    strncpy(r->text, text, MAX_LINE - 1);
}

/* ── String helpers ──────────────────────────────────────────────────── */
static bool str_contains_ci(const char *haystack, const char *needle) {
    /* Case-insensitive substring search */
    if (!haystack || !needle) return false;
    size_t nl = strlen(needle);
    for (size_t i = 0; haystack[i]; i++) {
        bool match = true;
        for (size_t j = 0; j < nl; j++) {
            if (!haystack[i + j]) { match = false; break; }
            if (tolower((unsigned char)haystack[i+j]) != tolower((unsigned char)needle[j])) {
                match = false; break;
            }
        }
        if (match) return true;
    }
    return false;
}

static bool line_matches(const char *line, const DocOpts *opts) {
    if (!opts->query) return true;
    return opts->case_insensitive
           ? str_contains_ci(line, opts->query)
           : (strstr(line, opts->query) != NULL);
}

static bool is_fn_decl(const char *line) {
    const char *p = line;
    while (*p == ' ') p++;
    return (strncmp(p, "pub fn ", 7) == 0 || strncmp(p, "fn ", 3) == 0 ||
            strncmp(p, "extern fn ", 10) == 0 || strncmp(p, "## fn ", 6) == 0 ||
            strncmp(p, "### `fn ", 8) == 0 || strstr(p, "fn ") != NULL);
}

static bool is_type_decl(const char *line) {
    const char *p = line;
    while (*p == ' ') p++;
    return (strncmp(p, "struct ", 7) == 0 || strncmp(p, "enum ", 5) == 0 ||
            strncmp(p, "union ", 6) == 0 || strncmp(p, "type ", 5) == 0 ||
            strncmp(p, "## `", 4) == 0 || strncmp(p, "### ", 4) == 0);
}

/* ── File scanning ───────────────────────────────────────────────────── */
static void scan_file(const char *path, const DocOpts *opts) {
    FILE *f = fopen(path, "r");
    if (!f) return;

    /* Store lines for context output */
    char *lines[256];
    int   nlines = 0;
    int   cap    = 256;
    char  linebuf[MAX_LINE];

    while (fgets(linebuf, sizeof(linebuf), f)) {
        /* Trim trailing newline */
        size_t len = strlen(linebuf);
        if (len > 0 && linebuf[len - 1] == '\n') linebuf[--len] = '\0';

        if (nlines < cap) lines[nlines++] = strdup(linebuf);

        int lineno = nlines;  /* 1-based */

        /* Apply filters */
        if (opts->fn_only   && !is_fn_decl(linebuf))   continue;
        if (opts->type_only && !is_type_decl(linebuf)) continue;

        if (line_matches(linebuf, opts)) {
            add_result(path, lineno, linebuf);
        }
    }
    fclose(f);

    for (int i = 0; i < nlines; i++) free(lines[i]);
}

static bool has_suffix(const char *path, const char *suffix) {
    size_t pl = strlen(path), sl = strlen(suffix);
    return pl >= sl && strcmp(path + pl - sl, suffix) == 0;
}

static void scan_dir_recursive(const char *dir, const char *ext_filter,
                                const DocOpts *opts) {
    DIR *d = opendir(dir);
    if (!d) return;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        char path[MAX_PATH];
        snprintf(path, sizeof(path), "%s/%s", dir, ent->d_name);
        struct stat st;
        if (stat(path, &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) {
            scan_dir_recursive(path, ext_filter, opts);
        } else if (S_ISREG(st.st_mode)) {
            if (!ext_filter || has_suffix(path, ext_filter) ||
                (ext_filter[0] == '*')) {
                scan_file(path, opts);
            }
        }
    }
    closedir(d);
}

/* ── List modes ──────────────────────────────────────────────────────── */
static void list_modules(void) {
    colprintf(COL_SECTION, "Available modules:\n\n");
    DIR *d = opendir("modules");
    if (!d) { printf("(modules/ not found)\n"); return; }
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        char path[MAX_PATH];
        snprintf(path, sizeof(path), "modules/%s", ent->d_name);
        struct stat st;
        if (stat(path, &st) == 0 && S_ISDIR(st.st_mode)) {
            printf("  modules/%-30s\n", ent->d_name);
        }
    }
    closedir(d);
}

static void list_fns(void) {
    colprintf(COL_SECTION, "Public functions (modules/std/):\n\n");
    DocOpts lopts = { .fn_only = true, .case_insensitive = false };
    scan_dir_recursive("modules/std", ".nano", &lopts);
    for (int i = 0; i < g_result_count; i++) {
        const char *line = g_results[i].text;
        const char *p = line;
        while (*p == ' ') p++;
        if (strncmp(p, "pub fn ", 7) == 0 || strncmp(p, "fn ", 3) == 0) {
            colprintf(COL_FILE, "  %s:%d: ", g_results[i].path, g_results[i].line);
            printf("%s\n", p);
        }
    }
}

/* ── Output results ──────────────────────────────────────────────────── */
static void print_results(const DocOpts *opts) {
    if (g_result_count == 0) {
        printf("No results found for '%s'\n", opts->query ? opts->query : "");
        return;
    }
    printf("\n");
    int shown = 0;
    const char *last_path = NULL;
    for (int i = 0; i < g_result_count; i++) {
        DocResult *r = &g_results[i];
        if (!last_path || strcmp(last_path, r->path) != 0) {
            if (last_path) printf("\n");
            colprintf(COL_SECTION, "── %s ──\n", r->path);
            last_path = r->path;
        }
        colprintf(COL_FILE, "%d", r->line);
        printf(": ");
        /* Highlight the query in the line */
        if (opts->query && g_color) {
            const char *pos = opts->case_insensitive
                ? NULL  /* skip highlight for case-insensitive */
                : strstr(r->text, opts->query);
            if (pos) {
                /* Print before, highlight, print after */
                fwrite(r->text, 1, (size_t)(pos - r->text), stdout);
                colprintf(COL_MATCH, "%s", opts->query);
                printf("%s", pos + strlen(opts->query));
            } else {
                printf("%s", r->text);
            }
        } else {
            printf("%s", r->text);
        }
        printf("\n");
        shown++;
    }
    printf("\n");
    printf("%d result%s\n", shown, shown != 1 ? "s" : "");
}

/* ── main ──────────────────────────────────────────────────────────────── */
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "nano-docs %s — nanolang documentation search\n\n", VERSION);
        fprintf(stderr, "Usage: nano-docs <query> [options]\n");
        fprintf(stderr, "       nano-docs --list-fns\n");
        fprintf(stderr, "       nano-docs --list-modules\n");
        fprintf(stderr, "       nano-docs --fn <name>     search function signatures\n");
        fprintf(stderr, "       nano-docs --type <name>   search type definitions\n");
        fprintf(stderr, "       nano-docs --module <mod>  restrict to module\n");
        fprintf(stderr, "       nano-docs --docs          search docs/ userguide/ only\n");
        fprintf(stderr, "       nano-docs -i <query>      case-insensitive\n");
        fprintf(stderr, "       nano-docs --context <n>   context lines (default: 3)\n");
        return 1;
    }

    /* Check if stdout is a terminal */
    g_color = isatty(1);

    DocOpts opts = {
        .query           = NULL,
        .case_insensitive = false,
        .fn_only         = false,
        .type_only       = false,
        .docs_only       = false,
        .modules_only    = false,
        .module_filter   = NULL,
        .context_lines   = 3,
        .list_fns        = false,
        .list_modules    = false,
        .verbose         = false,
    };

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--list-fns") == 0)   { opts.list_fns = true; }
        else if (strcmp(argv[i], "--list-modules") == 0) { opts.list_modules = true; }
        else if (strcmp(argv[i], "--fn") == 0 && i+1 < argc) {
            opts.fn_only = true; opts.query = argv[++i];
        }
        else if (strcmp(argv[i], "--type") == 0 && i+1 < argc) {
            opts.type_only = true; opts.query = argv[++i];
        }
        else if (strcmp(argv[i], "--module") == 0 && i+1 < argc) {
            opts.module_filter = argv[++i]; opts.modules_only = true;
        }
        else if (strcmp(argv[i], "--docs") == 0)  { opts.docs_only = true; }
        else if (strcmp(argv[i], "-i") == 0)       { opts.case_insensitive = true; }
        else if (strcmp(argv[i], "--no-color") == 0) { g_color = false; }
        else if (strcmp(argv[i], "--context") == 0 && i+1 < argc) {
            opts.context_lines = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--verbose") == 0) { opts.verbose = true; }
        else if (argv[i][0] != '-') {
            opts.query = argv[i];
        }
    }

    /* ── Dispatch special modes ────────────────────────────────────── */
    if (opts.list_modules) { list_modules(); return 0; }
    if (opts.list_fns)     { list_fns();     return 0; }

    if (!opts.query) {
        fprintf(stderr, "nano-docs: no query provided\n");
        return 1;
    }

    if (opts.verbose)
        fprintf(stderr, "[nano-docs] searching for '%s'...\n", opts.query);

    /* ── Search ────────────────────────────────────────────────────── */
    if (!opts.docs_only && !opts.modules_only) {
        /* Default: search everywhere */
        scan_dir_recursive("docs",      ".md",   &opts);
        scan_dir_recursive("userguide", ".md",   &opts);
        scan_dir_recursive("modules",   ".nano", &opts);
        scan_dir_recursive("examples",  ".nano", &opts);
    } else if (opts.docs_only) {
        scan_dir_recursive("docs",      ".md", &opts);
        scan_dir_recursive("userguide", ".md", &opts);
    } else if (opts.modules_only) {
        if (opts.module_filter) {
            char modpath[MAX_PATH];
            snprintf(modpath, sizeof(modpath), "modules/%s", opts.module_filter);
            scan_dir_recursive(modpath, ".nano", &opts);
        } else {
            scan_dir_recursive("modules", ".nano", &opts);
        }
    }

    print_results(&opts);

    return g_result_count > 0 ? 0 : 1;
}

/* I preserve original-input positions while replacement output changes size. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void *nl_regex_compile(const char *);
const char *nl_regex_replace_all(void *, const char *, const char *);
void nl_regex_free(void *);

int main(void) {
    const char *cases[][4] = {
        {"a", "a a", "long", "long long"},
        {"long", "long long", "a", "a a"},
        {"a", "a a", "", " "},
        {"a", "", "long", ""},
        {"z", "a a", "long", "a a"},
        {"^", "abc", "X", "Xabc"},
        {"$", "abc", "X", "abcX"},
        {"a*", "bbb", "X", "Xbbb"},
        {"a", "aa", "aa", "aaaa"},
        {"a", "zaaz", "long", "zlonglongz"},
        {"([a-z]+)", "a bb", "x", "x x"},
        {"a", "a a", "\\1", "\\1 \\1"},
        {"a*", "", "X", "X"},
        {"a", "a a", "λ", "λ λ"},
        {"a", "start a middle a end", "long", "stlongrt long middle long end"},
        {"^a", "aaa", "long", "longlonglong"},
        {"a$", "aaa", "long", "aalong"},
        {"a*", "aaab", "X", "XXb"},
    };
    size_t passed = 0;
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; ++i) {
        void *regex = nl_regex_compile(cases[i][0]);
        if (!regex) {
            fprintf(stderr, "I could not compile replacement case %zu.\n", i);
            return 2;
        }
        const char *result = nl_regex_replace_all(regex, cases[i][1], cases[i][2]);
        if (!result || strcmp(result, cases[i][3])) {
            fprintf(stderr, "I failed replacement case %zu: %s\n", i, result ? result : "null");
            return 3;
        }
        if (result != cases[i][1]) free((void *)result);
        nl_regex_free(regex);
        ++passed;
    }
    char input[1025];
    memset(input, 'a', sizeof input - 1); input[sizeof input - 1] = '\0';
    const char *patterns[] = {"a", "aa"};
    const char *replacements[] = {"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "x"};
    const size_t lengths[] = {32768, 512};
    for (size_t i = 0; i < 2; ++i) {
        void *regex = nl_regex_compile(patterns[i]);
        if (!regex) return 4;
        const char *result = nl_regex_replace_all(regex, input, replacements[i]);
        if (!result || strlen(result) != lengths[i]) return 5;
        for (size_t j = 0; j < lengths[i]; ++j)
            if (result[j] != replacements[i][0]) return 6;
        if (result != input) free((void *)result);
        nl_regex_free(regex);
        ++passed;
    }
    const char *borrowed = "a a";
    if (nl_regex_replace_all(NULL, borrowed, "long") != borrowed) return 7;
    ++passed;
    void *regex = nl_regex_compile("a");
    if (!regex) return 8;
    if (nl_regex_replace_all(regex, borrowed, NULL) != borrowed) return 9;
    ++passed;
    if (nl_regex_replace_all(regex, NULL, "long") != NULL) return 10;
    ++passed;
    nl_regex_free(regex);
    if (nl_regex_compile("[") != NULL) return 11;
    ++passed;
    printf("I passed %zu regex replacement cases.\n", passed);
    return 0;
}

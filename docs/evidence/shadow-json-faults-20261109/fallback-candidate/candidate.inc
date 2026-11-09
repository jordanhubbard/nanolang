static void shadow_json_escape(FILE *out, const char *s) {
    if (!s) return;
    const unsigned char *p = (const unsigned char *)s;
    while (*p) {
        const unsigned char *begin = p;
        while (*p >= 0x20 && *p != '\\' && *p != '"') p++;
        if (p != begin) {
            size_t wanted = (size_t)(p - begin);
            size_t written = fwrite(begin, 1, wanted, out);
            for (size_t i = written; i < wanted; i++) fputc((int)begin[i], out);
        }
        if (!*p) break;
        unsigned char c = *p++;
        switch (c) {
            case '\\': fputs("\\\\", out); break;
            case '"': fputs("\\\"", out); break;
            case '\n': fputs("\\n", out); break;
            case '\r': fputs("\\r", out); break;
            case '\t': fputs("\\t", out); break;
            default: fprintf(out, "\\u%04x", (unsigned int)c); break;
        }
    }
}

/* I flush each boundary so a killed shadow still identifies its source.
 * Timing is diagnostic only; the outer supervisor owns the deadline. */
static void shadow_trace_event(const char *event, const char *name, const char *file,
                               int line, double elapsed, int failures) {
    fprintf(stderr, "{\"event\":\"%s\",\"name\":\"", event);
    shadow_json_escape(stderr, name);
    fputs("\",\"source\":\"", stderr);
    shadow_json_escape(stderr, file);
    fprintf(stderr, "\",\"line\":%d,\"elapsed_seconds\":%.6f,\"failures\":%d}\n",
            line, elapsed, failures);
    fflush(stderr);
}


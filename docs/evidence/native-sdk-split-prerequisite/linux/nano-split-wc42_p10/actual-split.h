static DynArray* nl_str_split(const char* str, const char* delim) {
    DynArray* result = dyn_array_new(ELEM_STRING);
    if (!result) { fprintf(stderr, "I cannot allocate a complete split-string result.\n"); abort(); }
    if (!str) return result;
    size_t delim_len = strlen(delim);
    if (delim_len == 0) {
        size_t str_len = strnlen(str, 64*1024*1024);
        for (size_t i = 0; i < str_len; i++) {
            char* ch = gc_alloc_string(1);
            if (!ch) { fprintf(stderr, "I cannot allocate a complete split-string result.\n"); abort(); }
            ch[0] = str[i]; ch[1] = '\0';
            dyn_array_push_string(result, ch);
        }
        return result;
    }
    const char* start = str;
    const char* found;
    while ((found = strstr(start, delim)) != NULL) {
        size_t seg_len = (size_t)(found - start);
        char* seg = gc_alloc_string(seg_len);
        if (!seg) { fprintf(stderr, "I cannot allocate a complete split-string result.\n"); abort(); }
        memcpy(seg, start, seg_len);
        seg[seg_len] = '\0';
        dyn_array_push_string(result, seg);
        start = found + delim_len;
    }
    size_t rest_len = strlen(start);
    char* seg = gc_alloc_string(rest_len);
    if (!seg) { fprintf(stderr, "I cannot allocate a complete split-string result.\n"); abort(); }
    {
        memcpy(seg, start, rest_len);
        seg[rest_len] = '\0';
        dyn_array_push_string(result, seg);
    }
    return result;
}


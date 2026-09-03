# NanoISA v2 Module Format Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serialize and load NanoISA modules in the v2 container so verified signatures, layout-driven aggregates, and linked callable handles have an on-disk home.

**Architecture:** v2 lives beside v1, distinguished by `magic[3]` (`0x02` vs `0x01`); nothing reads or writes v2 until the final tasks switch producers over. Each section gets its own encoder/decoder pair in its own file, tested in isolation against a hand-built byte fixture, then wired into a whole-module serializer and loader. Sections are independent of one another, so tasks 3-9 can run in parallel once task 2 lands.

**Tech Stack:** C99, `-Wall -Wextra -Werror`, no external dependencies. Tests are standalone C binaries wired into `make test-units`.

**Spec:** `docs/superpowers/specs/2026-09-01-nanoisa-v2-module-format.md`

## Global Constraints

- C99 only. No `_Static_assert` (use the negative-array idiom), no designated-initializer tricks beyond C99, no compiler extensions.
- Everything compiles clean under `-Wall -Wextra -Werror` with **both** clang and gcc. GCC catches `-Wformat-truncation` that clang does not; CI's only GCC builds are `Memory Sanitizers` and `Code Coverage`.
- All integers are little-endian on the wire regardless of host byte order. Use the `rd16/rd32/rd64` and `wr16/wr32/wr64` helpers in `src/nanoisa/nvm_format_v2.c`; do not cast structs onto buffers.
- **Every bounds check uses subtraction, not addition.** `offset + size > limit` can wrap. Write `size > limit - offset`, having already established `offset <= limit`.
- Unknown enum values and reserved fields are rejected, never ignored. Reserved fields must be zero.
- Existing behaviour must not change until Task 12. v1 stays the default producer throughout.
- Test files follow the existing convention in `tests/nanoisa/test_nvm_format_v2.c`: `CHECK` / `CHECK_RESULT` macros, a `build_fixture` that starts valid, and mutation per rejection test.
- Every task ends green on `make test-units` and `make test-quick`.

---

## File Structure

Already built (PR #193):

- `src/nanoisa/nvm_format_v2.h` — header, section directory, `NvmV2Result`, LE helpers' declarations
- `src/nanoisa/nvm_format_v2.c` — header/directory read, write, validate
- `tests/nanoisa/test_nvm_format_v2.c` — 29 container tests

To create, one pair per section so each stays small and independently reviewable:

- `src/nanoisa/nvm_v2_sections.h` — every section's in-memory struct and its encode/decode declarations
- `src/nanoisa/nvm_v2_constants.c` — CONSTANTS
- `src/nanoisa/nvm_v2_signatures.c` — SIGNATURES
- `src/nanoisa/nvm_v2_layouts.c` — LAYOUTS
- `src/nanoisa/nvm_v2_functions.c` — FUNCTIONS
- `src/nanoisa/nvm_v2_globals.c` — GLOBALS
- `src/nanoisa/nvm_v2_imports.c` — IMPORTS and LINKS (same shape, share helpers)
- `src/nanoisa/nvm_v2_metadata.c` — METADATA and DEBUG (both trivial, both key/value-ish)
- `src/nanoisa/nvm_v2_module.c` — whole-module serialize and deserialize
- `tests/nanoisa/test_nvm_v2_<section>.c` — one per source file above

`CODE` needs no encoder: it is an opaque byte range the directory already locates.

---

### Task 1: Shared section-cursor helpers

**Files:**
- Create: `src/nanoisa/nvm_v2_sections.h`
- Create: `src/nanoisa/nvm_v2_cursor.c`
- Test: `tests/nanoisa/test_nvm_v2_cursor.c`
- Modify: `Makefile.gnu` (`NANOISA_SOURCES`, new `test-nvm-v2-cursor` target, add to `test-units`)

**Interfaces:**
- Consumes: `NvmV2Result` from `nvm_format_v2.h`
- Produces: `NvmV2Cursor`, `nvm_v2_cursor_init`, `nvm_v2_take`, `nvm_v2_u8/u16/u32/u64`, `nvm_v2_align4`, `nvm_v2_cursor_exhausted`

Every section decoder walks a byte range and must never read past it. Writing that check once removes the most likely bug from eight decoders.

- [ ] **Step 1: Write the failing test**

```c
/* tests/nanoisa/test_nvm_v2_cursor.c */
#include <stdio.h>
#include <string.h>
#include "nvm_v2_sections.h"

static int g_pass = 0, g_fail = 0;
#define CHECK(c, what) do { if (c) g_pass++; else { g_fail++; \
    printf("  FAIL: %s (%s:%d)\n", what, __FILE__, __LINE__); } } while (0)

static void test_reads_in_order(void) {
    uint8_t buf[8] = { 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08 };
    NvmV2Cursor c;
    nvm_v2_cursor_init(&c, buf, sizeof buf);
    uint8_t a; uint16_t b; uint32_t d;
    CHECK(nvm_v2_u8(&c, &a) == NVM_V2_OK && a == 0x01, "u8 reads first byte");
    CHECK(nvm_v2_u16(&c, &b) == NVM_V2_OK && b == 0x0302, "u16 is little-endian");
    CHECK(nvm_v2_u32(&c, &d) == NVM_V2_OK && d == 0x07060504, "u32 is little-endian");
    CHECK(nvm_v2_cursor_exhausted(&c) == false, "one byte remains");
}

static void test_read_past_end_is_rejected(void) {
    uint8_t buf[2] = { 0xAA, 0xBB };
    NvmV2Cursor c;
    nvm_v2_cursor_init(&c, buf, sizeof buf);
    uint32_t d;
    CHECK(nvm_v2_u32(&c, &d) == NVM_V2_ERR_TRUNCATED, "u32 past the end is rejected");
}

static void test_align4_rejects_nonzero_padding(void) {
    uint8_t good[4] = { 0xAA, 0x00, 0x00, 0x00 };
    uint8_t bad[4]  = { 0xAA, 0x00, 0x99, 0x00 };
    NvmV2Cursor c; uint8_t v;
    nvm_v2_cursor_init(&c, good, 4); nvm_v2_u8(&c, &v);
    CHECK(nvm_v2_align4(&c) == NVM_V2_OK, "zero padding accepted");
    nvm_v2_cursor_init(&c, bad, 4); nvm_v2_u8(&c, &v);
    CHECK(nvm_v2_align4(&c) == NVM_V2_ERR_SECTION_RANGE, "nonzero padding rejected");
}

int main(void) {
    printf("\n[nvm_v2_cursor] tests...\n\n");
    test_reads_in_order();
    test_read_past_end_is_rejected();
    test_align4_rejects_nonzero_padding();
    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `cc -std=c99 -Isrc -Isrc/nanoisa -o /tmp/t tests/nanoisa/test_nvm_v2_cursor.c src/nanoisa/nvm_v2_cursor.c src/nanoisa/nvm_format_v2.c src/nanoisa/nvm_format.c src/nanoisa/isa.c && /tmp/t`

Expected: compile error, `nvm_v2_sections.h` does not exist. Create the header with the declarations below and a `nvm_v2_cursor.c` whose functions all `return NVM_V2_OK;` without touching the cursor, then re-run. Expected: FAIL on every CHECK. Do not proceed until you have seen real failures rather than a compile error — a compile error does not prove the assertions are right.

```c
/* src/nanoisa/nvm_v2_sections.h -- declarations only for this step */
#ifndef NANOISA_NVM_V2_SECTIONS_H
#define NANOISA_NVM_V2_SECTIONS_H
#include "nvm_format_v2.h"

typedef struct {
    const uint8_t *base;
    size_t size;
    size_t pos;
} NvmV2Cursor;

void        nvm_v2_cursor_init(NvmV2Cursor *c, const uint8_t *base, size_t size);
bool        nvm_v2_cursor_exhausted(const NvmV2Cursor *c);
NvmV2Result nvm_v2_take(NvmV2Cursor *c, size_t n, const uint8_t **out);
NvmV2Result nvm_v2_u8 (NvmV2Cursor *c, uint8_t  *out);
NvmV2Result nvm_v2_u16(NvmV2Cursor *c, uint16_t *out);
NvmV2Result nvm_v2_u32(NvmV2Cursor *c, uint32_t *out);
NvmV2Result nvm_v2_u64(NvmV2Cursor *c, uint64_t *out);
NvmV2Result nvm_v2_align4(NvmV2Cursor *c);

#endif
```

- [ ] **Step 3: Implement**

```c
/* src/nanoisa/nvm_v2_cursor.c */
#include "nvm_v2_sections.h"

void nvm_v2_cursor_init(NvmV2Cursor *c, const uint8_t *base, size_t size) {
    c->base = base; c->size = size; c->pos = 0;
}

bool nvm_v2_cursor_exhausted(const NvmV2Cursor *c) { return c->pos >= c->size; }

NvmV2Result nvm_v2_take(NvmV2Cursor *c, size_t n, const uint8_t **out) {
    /* Subtraction form: c->pos <= c->size is the invariant, so this cannot
     * overflow the way `c->pos + n > c->size` could. */
    if (n > c->size - c->pos) return NVM_V2_ERR_TRUNCATED;
    if (out) *out = c->base + c->pos;
    c->pos += n;
    return NVM_V2_OK;
}

NvmV2Result nvm_v2_u8(NvmV2Cursor *c, uint8_t *out) {
    const uint8_t *p; NvmV2Result r = nvm_v2_take(c, 1, &p);
    if (r == NVM_V2_OK) *out = p[0];
    return r;
}

NvmV2Result nvm_v2_u16(NvmV2Cursor *c, uint16_t *out) {
    const uint8_t *p; NvmV2Result r = nvm_v2_take(c, 2, &p);
    if (r == NVM_V2_OK) *out = (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
    return r;
}

NvmV2Result nvm_v2_u32(NvmV2Cursor *c, uint32_t *out) {
    const uint8_t *p; NvmV2Result r = nvm_v2_take(c, 4, &p);
    if (r == NVM_V2_OK)
        *out = (uint32_t)p[0] | ((uint32_t)p[1] << 8)
             | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    return r;
}

NvmV2Result nvm_v2_u64(NvmV2Cursor *c, uint64_t *out) {
    const uint8_t *p; NvmV2Result r = nvm_v2_take(c, 8, &p);
    if (r != NVM_V2_OK) return r;
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)p[i] << (i * 8);
    *out = v;
    return NVM_V2_OK;
}

NvmV2Result nvm_v2_align4(NvmV2Cursor *c) {
    /* Padding must be zero. Accepting arbitrary filler would let two different
     * byte strings decode identically, which breaks lossless round-tripping. */
    while (c->pos % 4 != 0) {
        uint8_t pad;
        NvmV2Result r = nvm_v2_u8(c, &pad);
        if (r != NVM_V2_OK) return r;
        if (pad != 0) return NVM_V2_ERR_SECTION_RANGE;
    }
    return NVM_V2_OK;
}
```

- [ ] **Step 4: Run the test and confirm it passes**

Run the command from Step 2. Expected: `3 passed`-style output with `0 failed`.

- [ ] **Step 5: Wire into the build**

In `Makefile.gnu`: add `$(NANOISA_DIR)/nvm_v2_cursor.c` to `NANOISA_SOURCES`; add a `test-nvm-v2-cursor` target copied from the existing `test-nvm-format-v2` target with the names changed; append `test-nvm-v2-cursor` to the `test-units:` dependency list.

- [ ] **Step 6: Verify and commit**

```bash
make -f Makefile.gnu test-nvm-v2-cursor
make -f Makefile.gnu test-quick
git add src/nanoisa/nvm_v2_sections.h src/nanoisa/nvm_v2_cursor.c \
        tests/nanoisa/test_nvm_v2_cursor.c Makefile.gnu
git commit -m "feat(nanoisa): add bounds-checked cursor for v2 section decoding"
```

---

### Task 2: CONSTANTS section

**Files:**
- Create: `src/nanoisa/nvm_v2_constants.c`
- Modify: `src/nanoisa/nvm_v2_sections.h`
- Test: `tests/nanoisa/test_nvm_v2_constants.c`
- Modify: `Makefile.gnu`

**Interfaces:**
- Consumes: `NvmV2Cursor` and accessors from Task 1
- Produces: `NvmV2Constant`, `NvmV2Constants`, `nvm_v2_constants_decode`, `nvm_v2_constants_encoded_size`, `nvm_v2_constants_encode`, `nvm_v2_constants_free`

Wire format, repeated `count` times after a leading `u32 count`:

```
tag     u8      NanoValueTag
_pad    u8[3]   must be zero
length  u32     payload byte length
payload u8[length], then zero padding to a 4-byte boundary
```

Strings carry an explicit length so embedded zero bytes survive; this is the serialized half of the stored-string-length work.

- [ ] **Step 1: Add the declarations**

```c
/* append to src/nanoisa/nvm_v2_sections.h, before the #endif */

typedef struct {
    uint8_t        tag;      /* NanoValueTag */
    uint32_t       length;   /* payload bytes */
    const uint8_t *payload;  /* points into the caller's buffer; not owned */
} NvmV2Constant;

typedef struct {
    NvmV2Constant *items;
    uint32_t       count;
} NvmV2Constants;

/* Decodes in place: items[].payload aliases `data`, which must outlive `out`. */
NvmV2Result nvm_v2_constants_decode(const uint8_t *data, size_t size,
                                    NvmV2Constants *out);
void        nvm_v2_constants_free(NvmV2Constants *c);
size_t      nvm_v2_constants_encoded_size(const NvmV2Constants *c);
NvmV2Result nvm_v2_constants_encode(const NvmV2Constants *c,
                                    uint8_t *out, size_t size);
```

- [ ] **Step 2: Write the failing test**

```c
/* tests/nanoisa/test_nvm_v2_constants.c */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "nvm_v2_sections.h"
#include "isa.h"   /* TAG_STRING, TAG_INT, TAG_COUNT */

static int g_pass = 0, g_fail = 0;
#define CHECK(c, what) do { if (c) g_pass++; else { g_fail++; \
    printf("  FAIL: %s (%s:%d)\n", what, __FILE__, __LINE__); } } while (0)

/* "a\0b" -- three bytes with an embedded NUL, the case a strlen-based
 * encoder silently truncates. */
static const uint8_t EMBEDDED[3] = { 'a', 0x00, 'b' };

static size_t build(uint8_t *buf) {
    NvmV2Constant items[2];
    items[0].tag = TAG_STRING; items[0].length = 3; items[0].payload = EMBEDDED;
    static const uint8_t seven[1] = { 7 };
    items[1].tag = TAG_INT;    items[1].length = 1; items[1].payload = seven;
    NvmV2Constants c = { items, 2 };
    size_t n = nvm_v2_constants_encoded_size(&c);
    nvm_v2_constants_encode(&c, buf, n);
    return n;
}

static void test_round_trips_embedded_nul(void) {
    uint8_t buf[64];
    size_t n = build(buf);
    NvmV2Constants got;
    CHECK(nvm_v2_constants_decode(buf, n, &got) == NVM_V2_OK, "decodes");
    CHECK(got.count == 2, "two constants");
    CHECK(got.items[0].tag == TAG_STRING, "first is a string");
    CHECK(got.items[0].length == 3, "length is 3, not strlen's 1");
    CHECK(memcmp(got.items[0].payload, EMBEDDED, 3) == 0, "bytes survive the NUL");
    CHECK(got.items[1].tag == TAG_INT && got.items[1].length == 1, "second is int");
    nvm_v2_constants_free(&got);
}

static void test_length_past_end_is_rejected(void) {
    uint8_t buf[64];
    size_t n = build(buf);
    buf[8] = 0xFF;   /* first entry's length field */
    NvmV2Constants got;
    CHECK(nvm_v2_constants_decode(buf, n, &got) == NVM_V2_ERR_TRUNCATED,
          "a payload length past the end is rejected");
}

static void test_invalid_tag_is_rejected(void) {
    uint8_t buf[64];
    size_t n = build(buf);
    buf[4] = TAG_COUNT;   /* first entry's tag */
    NvmV2Constants got;
    CHECK(nvm_v2_constants_decode(buf, n, &got) == NVM_V2_ERR_SECTION_TYPE,
          "an out-of-range value tag is rejected");
}

static void test_nonzero_entry_padding_is_rejected(void) {
    uint8_t buf[64];
    size_t n = build(buf);
    buf[5] = 0x01;   /* first entry's _pad[0] */
    NvmV2Constants got;
    CHECK(nvm_v2_constants_decode(buf, n, &got) == NVM_V2_ERR_RESERVED_FLAGS,
          "nonzero entry padding is rejected");
}

static void test_truncated_count_is_rejected(void) {
    uint8_t buf[2] = { 0, 0 };
    NvmV2Constants got;
    CHECK(nvm_v2_constants_decode(buf, 2, &got) == NVM_V2_ERR_TRUNCATED,
          "a section too short to hold the count is rejected");
}

int main(void) {
    printf("\n[nvm_v2_constants] tests...\n\n");
    test_round_trips_embedded_nul();
    test_length_past_end_is_rejected();
    test_invalid_tag_is_rejected();
    test_nonzero_entry_padding_is_rejected();
    test_truncated_count_is_rejected();
    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
```

- [ ] **Step 3: Run the test and confirm it fails**

Create `src/nanoisa/nvm_v2_constants.c` with stub bodies (`return NVM_V2_OK;`, `return 0;`, empty `free`) so it links, then run:

`cc -std=c99 -Isrc -Isrc/nanoisa -o /tmp/t tests/nanoisa/test_nvm_v2_constants.c src/nanoisa/nvm_v2_constants.c src/nanoisa/nvm_v2_cursor.c src/nanoisa/nvm_format_v2.c src/nanoisa/nvm_format.c src/nanoisa/isa.c && /tmp/t`

Expected: FAIL on every CHECK.

- [ ] **Step 4: Implement**

```c
/* src/nanoisa/nvm_v2_constants.c */
#include <stdlib.h>
#include <string.h>
#include "nvm_v2_sections.h"
#include "isa.h"

static size_t pad4(size_t n) { return (n + 3u) & ~(size_t)3u; }

NvmV2Result nvm_v2_constants_decode(const uint8_t *data, size_t size,
                                    NvmV2Constants *out) {
    out->items = NULL; out->count = 0;
    NvmV2Cursor c;
    nvm_v2_cursor_init(&c, data, size);

    uint32_t count;
    NvmV2Result r = nvm_v2_u32(&c, &count);
    if (r != NVM_V2_OK) return r;
    if (count == 0) return NVM_V2_OK;

    /* Each entry is at least 8 bytes, so a count larger than the remaining
     * bytes allows is malformed. Reject before allocating, so a tiny hostile
     * section cannot ask for a huge allocation. */
    if ((size_t)count > (size - c.pos) / 8) return NVM_V2_ERR_TRUNCATED;

    NvmV2Constant *items = calloc(count, sizeof *items);
    if (!items) return NVM_V2_ERR_TRUNCATED;

    for (uint32_t i = 0; i < count; i++) {
        uint8_t tag, p0, p1, p2;
        uint32_t length;
        if ((r = nvm_v2_u8(&c, &tag))  != NVM_V2_OK) goto fail;
        if ((r = nvm_v2_u8(&c, &p0))   != NVM_V2_OK) goto fail;
        if ((r = nvm_v2_u8(&c, &p1))   != NVM_V2_OK) goto fail;
        if ((r = nvm_v2_u8(&c, &p2))   != NVM_V2_OK) goto fail;
        if (p0 || p1 || p2) { r = NVM_V2_ERR_RESERVED_FLAGS; goto fail; }
        if (tag >= TAG_COUNT)  { r = NVM_V2_ERR_SECTION_TYPE; goto fail; }
        if ((r = nvm_v2_u32(&c, &length)) != NVM_V2_OK) goto fail;

        const uint8_t *payload = NULL;
        if ((r = nvm_v2_take(&c, length, &payload)) != NVM_V2_OK) goto fail;
        if ((r = nvm_v2_align4(&c)) != NVM_V2_OK) goto fail;

        items[i].tag = tag;
        items[i].length = length;
        items[i].payload = payload;
    }

    out->items = items;
    out->count = count;
    return NVM_V2_OK;

fail:
    free(items);
    return r;
}

void nvm_v2_constants_free(NvmV2Constants *c) {
    if (!c) return;
    free(c->items);
    c->items = NULL;
    c->count = 0;
}

size_t nvm_v2_constants_encoded_size(const NvmV2Constants *c) {
    size_t n = 4;
    for (uint32_t i = 0; i < c->count; i++)
        n += 8 + pad4(c->items[i].length);
    return n;
}

NvmV2Result nvm_v2_constants_encode(const NvmV2Constants *c,
                                    uint8_t *out, size_t size) {
    size_t need = nvm_v2_constants_encoded_size(c);
    if (size < need) return NVM_V2_ERR_TRUNCATED;
    memset(out, 0, need);

    size_t p = 0;
    out[p++] = (uint8_t)c->count;
    out[p++] = (uint8_t)(c->count >> 8);
    out[p++] = (uint8_t)(c->count >> 16);
    out[p++] = (uint8_t)(c->count >> 24);

    for (uint32_t i = 0; i < c->count; i++) {
        out[p] = c->items[i].tag;
        p += 4;                       /* tag + three zero pad bytes */
        uint32_t len = c->items[i].length;
        out[p++] = (uint8_t)len;
        out[p++] = (uint8_t)(len >> 8);
        out[p++] = (uint8_t)(len >> 16);
        out[p++] = (uint8_t)(len >> 24);
        if (len) memcpy(out + p, c->items[i].payload, len);
        p += pad4(len);               /* memset already zeroed the padding */
    }
    return NVM_V2_OK;
}
```

- [ ] **Step 5: Run the test and confirm it passes**

Run the Step 3 command. Expected: `0 failed`.

- [ ] **Step 6: Wire into the build and commit**

```bash
# Makefile.gnu: add nvm_v2_constants.c to NANOISA_SOURCES, add a
# test-nvm-v2-constants target modelled on test-nvm-format-v2, and append it
# to test-units.
make -f Makefile.gnu test-nvm-v2-constants
make -f Makefile.gnu test-quick
git add src/nanoisa/nvm_v2_constants.c src/nanoisa/nvm_v2_sections.h \
        tests/nanoisa/test_nvm_v2_constants.c Makefile.gnu
git commit -m "feat(nanoisa): encode and decode the v2 CONSTANTS section"
```

---

### Tasks 3-9: the remaining sections

Each of these follows Task 2 exactly — declarations in `nvm_v2_sections.h`, one `.c`, one test, a Makefile target, a commit. They are mutually independent and can be worked in parallel once Task 1 lands. The pattern per task:

1. Add the struct and four function declarations to `nvm_v2_sections.h`
2. Write the test: a `build()` helper that encodes a two-entry fixture, a round-trip test, and one rejection test per malformed shape listed below
3. Stub the `.c`, run, confirm every CHECK fails
4. Implement using `NvmV2Cursor`, `pad4`, and the count-vs-remaining-bytes guard from Task 2
5. Run, confirm green
6. Wire into `NANOISA_SOURCES`, add the test target, append to `test-units`, commit

**Task 3: SIGNATURES** — `src/nanoisa/nvm_v2_signatures.c`

```
count        u32
per entry:
  param_count  u16
  result_count u16
  param_tags   u8[param_count],  pad to 4
  result_tags  u8[result_count], pad to 4
```

Struct: `{ uint16_t param_count; uint16_t result_count; const uint8_t *param_tags; const uint8_t *result_tags; }`.
Rejections to test: any tag `>= TAG_COUNT`; `param_count` or `result_count` running past the section; nonzero padding; truncated count.
This is the table that makes verified signatures checkable at load — functions, imports and links all reference a `signature_idx` into it, so it must land before Tasks 5, 7 and 8 can be wired in Task 10.

**Task 4: LAYOUTS** — `src/nanoisa/nvm_v2_layouts.c`

```
count        u32
per entry:
  kind        u8    0=struct 1=tuple 2=union 3=enum
  _pad        u8
  field_count u16
  name_idx    u32   CONSTANTS index, or 0xFFFFFFFF for anonymous
  per field:
    type_tag          u8
    _pad              u8[3]
    nested_layout_idx u32   0xFFFFFFFF when scalar
    name_idx          u32
```

Struct: `NvmV2LayoutField { uint8_t type_tag; uint32_t nested_layout_idx; uint32_t name_idx; }` and `NvmV2Layout { uint8_t kind; uint16_t field_count; uint32_t name_idx; NvmV2LayoutField *fields; }`.
Rejections to test: `kind > 3`; a `nested_layout_idx` that is **not lower-numbered than the entry containing it** (the spec requires the table be acyclic by construction, and a forward or self reference is how a decoder gets tricked into unbounded recursion); `field_count` past the section; nonzero padding.

**Task 5: FUNCTIONS** — `src/nanoisa/nvm_v2_functions.c`

```
count u32
per entry:
  name_idx      u32
  signature_idx u32
  code_offset   u64
  code_length   u64
  local_count   u16
  upvalue_count u16
  max_stack     u16   verifier-proven maximum operand depth
  flags         u16   reserved, must be zero
```

Note what moved: `arity`, `result_tag` and `result_count` are **not** here — they live in SIGNATURES. `max_stack` is new and is what lets the verifier discharge the maximum-operand-depth obligation statically instead of leaning on a runtime limit.
Rejections to test: nonzero `flags`; truncated entry; count larger than the remaining bytes allow.

**Task 6: GLOBALS** — `src/nanoisa/nvm_v2_globals.c`

```
count u32
per entry:
  name_idx u32
  type_tag u8
  flags    u8    bit 0 = mutable; other bits reserved, must be zero
  _pad     u16
  init_idx u32   CONSTANTS index, or 0xFFFFFFFF for zero-initialized
```

Rejections to test: `type_tag >= TAG_COUNT`; reserved `flags` bits set; nonzero `_pad`; truncated entry.
This section is the prerequisite for sizing globals from declarations rather than `VM_MAX_GLOBALS`, and for giving the verifier a real bound to check `LOAD_GLOBAL`/`STORE_GLOBAL` against.

**Task 7: IMPORTS and LINKS** — `src/nanoisa/nvm_v2_imports.c`

Both in one file: same shape, and keeping them together avoids duplicating the shared decode helper.

```
IMPORTS:                          LINKS:
count u32                         count u32
per entry:                        per entry:
  module_name_idx u32               module_name_idx u32
  symbol_name_idx u32               symbol_name_idx u32
  signature_idx   u32               signature_idx   u32
  kind            u8                flags           u32  bit 0 = weak
  _pad            u8[3]
```

Rejections to test: `kind > 1` (0=ffi, 1=coprocess); reserved bits in LINKS `flags`; nonzero `_pad`; truncated entry.
Parameter counts and type tags deliberately do **not** appear — they are in SIGNATURES, which is what removes v1's variable-length import tail.

**Task 8: METADATA** — `src/nanoisa/nvm_v2_metadata.c`

```
count u32
per entry:
  key_idx   u32   CONSTANTS index
  value_idx u32   CONSTANTS index
```

Rejections to test: truncated entry; count past the section. Free-form by design so adding a key is not a format change.

**Task 9: DEBUG** — add to `src/nanoisa/nvm_v2_metadata.c`

```
count u32
per entry:
  bytecode_offset u64   widened from v1's u32 to match CODE
  source_line     u32
  source_col      u32   1-based; 0 means unknown
```

Rejections to test: truncated entry; count past the section.

---

### Task 10: Whole-module serialize and deserialize

**Files:**
- Create: `src/nanoisa/nvm_v2_module.c`
- Modify: `src/nanoisa/nvm_v2_sections.h`
- Test: `tests/nanoisa/test_nvm_v2_module.c`
- Modify: `Makefile.gnu`

**Interfaces:**
- Consumes: every section codec from Tasks 2-9, plus `nvm_v2_write_header`, `nvm_v2_write_section`, `nvm_v2_validate` from `nvm_format_v2.h`
- Produces: `NvmV2Module`, `nvm_v2_module_serialize`, `nvm_v2_module_deserialize`, `nvm_v2_module_free`

`NvmV2Module` holds one optional struct per section plus the raw `CODE` bytes. Serialization emits the header, then the directory, then payloads in ascending section-type order, then patches the checksum. Deserialization runs `nvm_v2_validate` first and decodes only what the directory locates.

Cross-section validation belongs here, not in the individual codecs — a codec sees one section and cannot check an index into another:

- every `signature_idx` in FUNCTIONS, IMPORTS and LINKS is `< signatures.count`
- every `name_idx`, `init_idx`, `key_idx`, `value_idx` is `< constants.count` (or the sentinel where one is allowed)
- every `nested_layout_idx` is `< layouts.count`
- `entry_point` is `< functions.count`, or `NVM_V2_NO_ENTRY_POINT`
- every function's `[code_offset, code_offset + code_length)` lies inside the CODE section — **by subtraction**
- feature bits agree with the sections present: `FEATURE_LINKED` iff LINKS is non-empty, `FEATURE_FFI` iff IMPORTS is non-empty, `FEATURE_DEBUG` iff DEBUG is present

Tests: a full round-trip of a module using every section; then one rejection test per bullet above. Add `NVM_V2_ERR_INDEX_RANGE` and `NVM_V2_ERR_FEATURE_MISMATCH` to `NvmV2Result` and to `nvm_v2_result_name` — every enum value must have a name, since `nvm_v2_result_name` is what the test failure output prints.

Commit: `feat(nanoisa): serialize and deserialize whole v2 modules`

---

### Task 11: Convert an NvmModule to and from v2

**Files:**
- Create: `src/nanoisa/nvm_v2_convert.c`
- Test: `tests/nanoisa/test_nvm_v2_convert.c`
- Modify: `src/nanoisa/nvm_v2_sections.h`, `Makefile.gnu`

**Interfaces:**
- Consumes: `NvmV2Module` from Task 10, `NvmModule` from `nvm_format.h`
- Produces: `nvm_v2_from_nvm_module`, `nvm_v2_to_nvm_module`

This is the bridge that lets v2 be adopted without rewriting every producer at once. The interesting direction is v1-shaped in-memory module to v2 on disk, because it is where the structural differences surface:

- v1's string pool becomes CONSTANTS entries tagged `TAG_STRING`, carrying `mod->string_lengths[i]` rather than `strlen`
- each distinct `(arity, result_tag, result_count, param_types)` shape across `mod->functions` and `mod->imports` becomes one SIGNATURES entry; functions and imports then reference it by index. Deduplicate: two functions with the same shape must share an entry, or signature-index comparison stops being a valid equality test
- `mod->struct_count` / enums / unions become LAYOUTS entries
- `mod->module_refs` become LINKS entries
- `max_stack` is not available from a v1 module; emit `0` and record in the commit message that Task 13 must populate it from the verifier

Tests: build a small `NvmModule` by hand, convert to v2, serialize, deserialize, convert back, and assert the round-trip preserves function count, import count, string bytes **including an embedded NUL**, and signature identity. Add one test asserting two identically-shaped functions share a signature index.

Commit: `feat(nanoisa): convert between NvmModule and the v2 container`

---

### Task 12: Emit v2 behind a flag

**Files:**
- Modify: `src/nanovirt/main.c` (add `--emit-nvm-v2`)
- Modify: `modules/nanoisa/nanoisa.c` (`nanoisa_load_bytes` dispatches on `magic[3]`)
- Modify: `modules/nanoisa/module.json` (add the new sources)
- Test: `tests/nanoisa/test_nvm_v2_endtoend.c`
- Modify: `Makefile.gnu`

**Interfaces:**
- Consumes: Task 11's converters
- Produces: a `--emit-nvm-v2` flag and a magic-dispatching loader

`nanoisa_load_bytes` is the single funnel every consumer already goes through — the VM, the co-process (`cop_main.c:48`), the daemon (`vmd_server.c:205`) and generated wrappers — so dispatching on `magic[3]` there is the whole loader change. A v1 module keeps taking the v1 path; a v2 module takes the new one; anything else is rejected as it is today.

**Do not change the default.** v1 remains what `--emit-nvm` produces until Task 14.

Add the new sources to `modules/nanoisa/module.json`. Four separate link contexts consume the assembler and each fails independently — `Makefile.gnu`, `modules/nanoisa/module.json`, `modules/forth_see/module.json`, `examples/Makefile` and the object list in `src/nanovirt/wrapper_gen.c`. `make test-quick` builds none of them; `make test-units` and `make examples-core` do. Run both.

Test: compile a `.nano` file with `--emit-nvm-v2`, load it back through `nanoisa_load_bytes`, and assert it executes to the same result as the v1 build of the same source.

Commit: `feat(nanovirt): emit and load v2 modules behind --emit-nvm-v2`

---

### Task 13: Populate max_stack from the verifier

**Files:**
- Modify: `src/nanoisa/verifier.c`
- Modify: `src/nanoisa/nvm_v2_convert.c`
- Test: `tests/nanoisa/test_verifier.c`

`verify_stack_heights` already computes a height for every instruction but discards the maximum. Return it, store it in the FUNCTIONS entry during conversion, and have the loader check it: reject a module whose declared `max_stack` is below the height the verifier computes.

Decide and record which direction is authoritative. The spec leaves it open. Recommendation: the producer computes it and the verifier confirms, because that is cheaper at load and a mismatch is then a real signal that the producer and verifier disagree — which is worth failing on.

Tests: a module whose declared `max_stack` is too low is rejected; one that matches is accepted; a function with no instructions has `max_stack == 0`.

Commit: `feat(verifier): compute and check max operand depth`

**Note:** this depends on `verify_stack_heights` in its current form. The branch that collides with it is `task_c6e5d089` ("verify return shape, maximum operand depth, frame depth, ownership effects, and explicit termination"), which rewrites that function against a version predating #158's variadic stack-effect resolution and so needs redoing rather than rebasing. If that redo is in flight, coordinate — both compute the same maximum, and doing it twice in one function is how the two sides end up disagreeing.

PR #159 does **not** collide with this task; it never touches `verify_stack_heights`. I said otherwise in an earlier revision of this plan and was wrong.

---

### Task 14: Make v2 the default

**Files:**
- Modify: `src/nanovirt/main.c`
- Modify: `docs/NANOISA.md`
- Modify: `docs/ROADMAP.md`
- Modify: `CHANGELOG.md`

Switch `--emit-nvm` to produce v2 and retire `--emit-nvm-v2` as an alias. Make `nanoisa_load_bytes` reject v1 with the message from the spec:

```
module 'foo.nvm' was built for NanoISA v1 (NVM\x01);
rebuild it with nanoc 4.0 or later
```

`.nvm` files are build artifacts rather than distributed packages, so the rebuild cost falls on the build system. Confirm before landing: `git ls-files '*.nvm'` should return nothing. If it returns anything, those files need regenerating in the same commit.

Only now tick the two roadmap items, and only with evidence:
- *"design a NanoISA v2 module header with format version, ISA version, feature bits, total size, and bounded section directory"*
- *"serialize required code, constants, signatures, globals, imports, layouts, links, metadata, and optional debug sections"*

Follow the ledger convention in `docs/ROADMAP.md`: a `- [x]` line stating what shipped and which tests cover it, not merely that it is done.

Commit: `feat(nanoisa): make the v2 module format the default`

---

## Parallelisation

Task 1 must land first — everything else uses the cursor.

Tasks 2-9 are then mutually independent: eight workers, one section each, no shared files except `nvm_v2_sections.h` and `Makefile.gnu`. Both are append-only in these tasks, so conflicts are trivial ROADMAP-style line collisions rather than semantic ones. If the fleet has a merge queue, order the merges rather than the work.

Task 10 needs all of 2-9. Tasks 11-14 are strictly sequential after it.

Rough shape: Task 1 alone, then a wide fan-out, then a narrow tail. The tail is where the design questions live — cross-section validation, signature deduplication, the `max_stack` authority decision — so it is worth keeping on one pair of hands rather than splitting.

## Self-Review

**Spec coverage.** Header and directory: PR #193. CONSTANTS, SIGNATURES, LAYOUTS, FUNCTIONS, GLOBALS, IMPORTS, LINKS, METADATA, DEBUG: Tasks 2-9. CODE: no codec needed, the directory locates it; the range check is in Task 10. Serializer and loader: Task 10. Migration: Tasks 12 and 14. `max_stack`: Task 13. Removed compile-time maxima: implicit in Tasks 5, 6 and 10, which size from serialized counts. Roadmap ticks: Task 14.

**Gap I am recording rather than hiding.** The spec lists five open questions. This plan settles two — section ordering (ascending type, Task 10) and `max_stack` authority (producer computes, verifier confirms, Task 13) — and does **not** settle constant deduplication or the LINKS weak flag. Constant dedup is left to the producer as a size optimisation and the format does not require it; the weak flag is encoded and validated but nothing consumes it until 4.4's capability work. Signature dedup *is* required and is specified in Task 11, because signature-index equality depends on it.

**Not in scope.** Retiring the ~64 orphaned opcodes is a separate roadmap item; it becomes possible once v1 is gone, but bundling it here would make Task 14 unreviewable. The assembler's symbolic operands and lossless disassembly are likewise separate items that happen to be unblocked by SIGNATURES and LAYOUTS existing.

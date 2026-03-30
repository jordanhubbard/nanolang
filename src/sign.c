/*
 * sign.c — Ed25519 WASM module signing for nanoc
 *
 * Part 1 — nanoc sign <file.wasm>
 *   Loads (or generates) an Ed25519 keypair from ~/.nanoc/signing.key,
 *   strips any existing agentos.signature section, SHA-256s the clean
 *   bytes, signs the hash, appends the 149-byte custom section, and
 *   overwrites the file.
 *
 * Part 2 — nanoc verify <file.wasm>
 *   Finds the agentos.signature section, re-computes the SHA-256 of
 *   the module without that section, and verifies the Ed25519 signature.
 *
 * Part 3 — nano.toml sign_key integration:
 *   When nano.toml contains `sign_key = "path/to/key"`, nanoc build
 *   should auto-sign the output .wasm using that key file.
 *   The sign_key path overrides the default ~/.nanoc/signing.key.
 *   Wiring: parse sign_key in module_builder.c (nano.toml reader),
 *   then call wasm_sign_file(output_path, sign_key_path) after WASM emit.
 *
 * Compile:
 *   cc -std=c99 -D_GNU_SOURCE -Wall -Wextra -I/usr/include/openssl sign.c -lcrypto
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <pwd.h>

#if defined(__linux__)
#  include <sys/random.h>
#endif

#include <openssl/evp.h>
#include <openssl/sha.h>
#include <openssl/err.h>

/* -------------------------------------------------------------------------
 * Constants
 * ---------------------------------------------------------------------- */

#define SIG_SECTION_NAME      "agentos.signature"
#define SIG_SECTION_NAME_LEN  17          /* strlen("agentos.signature") */
#define SIG_PUBKEY_OFF        0
#define SIG_PUBKEY_LEN        32
#define SIG_SIG_OFF           32
#define SIG_SIG_LEN           64
#define SIG_HASH_OFF          96
#define SIG_HASH_LEN          32
#define SIG_PAYLOAD_LEN       128         /* pubkey[32] + sig[64] + hash[32] */

/*
 * Full encoded custom section layout (149 bytes):
 *   0x00           — section id (custom)
 *   0x92 0x01      — LEB128(146): section body size
 *   0x11           — LEB128(17):  name length
 *   "agentos.signature" (17 bytes)
 *   payload        (128 bytes)
 *
 * Body = 1 + 17 + 128 = 146 bytes
 * Total = 1 (id) + 2 (LEB size) + 146 (body) = 149 bytes
 */
#define SIG_SECTION_TOTAL     149

/* WASM magic header */
static const uint8_t WASM_MAGIC[8] = {
    0x00, 0x61, 0x73, 0x6D,       /* \0asm */
    0x01, 0x00, 0x00, 0x00        /* version 1 */
};

/* -------------------------------------------------------------------------
 * Helper: print hex bytes to stdout
 * ---------------------------------------------------------------------- */
static void hex_print(const uint8_t *b, int n)
{
    for (int i = 0; i < n; i++)
        printf("%02x", b[i]);
}

/* -------------------------------------------------------------------------
 * Helper: decode a LEB128 unsigned integer
 * Returns number of bytes consumed, or -1 on error.
 * ---------------------------------------------------------------------- */
static int leb128_decode(const uint8_t *buf, size_t buf_len,
                         uint64_t *out_val)
{
    uint64_t result = 0;
    int shift = 0;
    int i = 0;
    while (i < (int)buf_len) {
        uint8_t b = buf[i++];
        result |= (uint64_t)(b & 0x7F) << shift;
        shift += 7;
        if (!(b & 0x80)) {
            *out_val = result;
            return i;
        }
        if (shift >= 64) break;
    }
    return -1; /* truncated or too long */
}

/* -------------------------------------------------------------------------
 * Helper: scan WASM for the agentos.signature custom section.
 *
 * Returns true if found; sets *out_off to the byte offset of the section-id
 * byte (0x00) and *out_total to the number of bytes in the full section
 * (including the section-id and the LEB128 size field).
 * ---------------------------------------------------------------------- */
static bool wasm_find_sig_section(const uint8_t *wasm, size_t len,
                                  size_t *out_off, size_t *out_total)
{
    if (len < 8) return false;
    /* Require WASM magic */
    if (memcmp(wasm, WASM_MAGIC, 8) != 0) return false;

    size_t pos = 8; /* skip 8-byte header */

    while (pos < len) {
        size_t sec_start = pos;

        /* Section id */
        if (pos >= len) break;
        uint8_t sec_id = wasm[pos++];

        /* LEB128 section size */
        uint64_t sec_size = 0;
        int nb = leb128_decode(wasm + pos, len - pos, &sec_size);
        if (nb <= 0) break;
        pos += (size_t)nb;

        size_t body_start = pos;
        size_t body_end   = pos + (size_t)sec_size;
        if (body_end > len) break;

        if (sec_id == 0x00) {
            /* Custom section: read name */
            uint64_t name_len = 0;
            int nl = leb128_decode(wasm + pos, body_end - pos, &name_len);
            if (nl > 0 && name_len == SIG_SECTION_NAME_LEN) {
                size_t name_off = pos + (size_t)nl;
                if (name_off + name_len <= body_end &&
                    memcmp(wasm + name_off, SIG_SECTION_NAME,
                           SIG_SECTION_NAME_LEN) == 0)
                {
                    *out_off   = sec_start;
                    *out_total = body_end - sec_start;
                    return true;
                }
            }
        }

        pos = body_start + (size_t)sec_size;
    }
    return false;
}

/* -------------------------------------------------------------------------
 * Helper: get path to default key file.
 * Writes into buf[buf_len]. Returns 0 on success, -1 on error.
 * ---------------------------------------------------------------------- */
static int default_key_path(char *buf, size_t buf_len)
{
    const char *home = getenv("HOME");
    if (!home) {
        struct passwd *pw = getpwuid(getuid());
        if (!pw) {
            fprintf(stderr, "sign: cannot determine HOME directory\n");
            return -1;
        }
        home = pw->pw_dir;
    }
    int n = snprintf(buf, buf_len, "%s/.nanoc/signing.key", home);
    if (n < 0 || (size_t)n >= buf_len) {
        fprintf(stderr, "sign: home path too long\n");
        return -1;
    }
    return 0;
}

/* -------------------------------------------------------------------------
 * Helper: ensure directory exists with given mode.
 * Returns 0 on success, -1 on error.
 * ---------------------------------------------------------------------- */
static int ensure_dir(const char *path, mode_t mode)
{
    struct stat st;
    if (stat(path, &st) == 0) {
        if (S_ISDIR(st.st_mode)) return 0;
        fprintf(stderr, "sign: %s exists but is not a directory\n", path);
        return -1;
    }
    if (mkdir(path, mode) != 0 && errno != EEXIST) {
        fprintf(stderr, "sign: mkdir %s: %s\n", path, strerror(errno));
        return -1;
    }
    return 0;
}

/* -------------------------------------------------------------------------
 * load_or_gen_keypair
 *
 * Loads the 64-byte keypair file (seed[32] + pubkey[32]) from key_path.
 * If the file does not exist, generates a new keypair using getrandom(2)
 * for the seed and OpenSSL for deriving the public key, then saves it.
 *
 * Returns 0 on success, -1 on error.
 * ---------------------------------------------------------------------- */
static int load_or_gen_keypair(const char *key_path,
                               uint8_t seed[32], uint8_t pubkey[32])
{
    FILE *f = fopen(key_path, "rb");
    if (f) {
        uint8_t buf[64];
        if (fread(buf, 1, 64, f) != 64) {
            fprintf(stderr, "sign: key file %s is too short\n", key_path);
            fclose(f);
            return -1;
        }
        fclose(f);
        memcpy(seed,   buf,      32);
        memcpy(pubkey, buf + 32, 32);
        return 0;
    }

    /* File does not exist — generate new keypair */
    if (errno != ENOENT) {
        fprintf(stderr, "sign: open %s: %s\n", key_path, strerror(errno));
        return -1;
    }

    /* Fill seed with cryptographically secure random bytes */
#if defined(__linux__)
    ssize_t got = getrandom(seed, 32, 0);
    if (got != 32) {
        fprintf(stderr, "sign: getrandom failed: %s\n", strerror(errno));
        return -1;
    }
#elif defined(__APPLE__) || defined(__FreeBSD__)
    arc4random_buf(seed, 32);
#else
    /* POSIX fallback: /dev/urandom */
    {
        FILE *urandom = fopen("/dev/urandom", "rb");
        if (!urandom || fread(seed, 1, 32, urandom) != 32) {
            fprintf(stderr, "sign: failed to read /dev/urandom\n");
            if (urandom) fclose(urandom);
            return -1;
        }
        fclose(urandom);
    }
#endif

    /* Derive public key via OpenSSL */
    EVP_PKEY *pkey = EVP_PKEY_new_raw_private_key(EVP_PKEY_ED25519, NULL,
                                                   seed, 32);
    if (!pkey) {
        fprintf(stderr, "sign: EVP_PKEY_new_raw_private_key failed\n");
        ERR_print_errors_fp(stderr);
        return -1;
    }

    size_t pub_len = 32;
    if (EVP_PKEY_get_raw_public_key(pkey, pubkey, &pub_len) != 1 ||
        pub_len != 32)
    {
        fprintf(stderr, "sign: EVP_PKEY_get_raw_public_key failed\n");
        ERR_print_errors_fp(stderr);
        EVP_PKEY_free(pkey);
        return -1;
    }
    EVP_PKEY_free(pkey);

    /* Ensure ~/.nanoc/ directory exists */
    char dir_buf[4096];
    snprintf(dir_buf, sizeof(dir_buf), "%s", key_path);
    /* Strip filename from path to get directory */
    char *slash = strrchr(dir_buf, '/');
    if (slash) {
        *slash = '\0';
        if (ensure_dir(dir_buf, 0700) != 0)
            return -1;
    }

    /* Write key file (mode 0600) */
    int fd = open(key_path, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) {
        fprintf(stderr, "sign: create %s: %s\n", key_path, strerror(errno));
        return -1;
    }
    uint8_t kbuf[64];
    memcpy(kbuf,      seed,   32);
    memcpy(kbuf + 32, pubkey, 32);
    ssize_t written = write(fd, kbuf, 64);
    close(fd);
    if (written != 64) {
        fprintf(stderr, "sign: write %s failed\n", key_path);
        return -1;
    }

    fprintf(stderr, "sign: generated new keypair -> %s\n", key_path);
    return 0;
}

/* -------------------------------------------------------------------------
 * Helper: read entire file into a malloc'd buffer.
 * Returns 0 on success, -1 on error.
 * ---------------------------------------------------------------------- */
static int read_file(const char *path, uint8_t **out_buf, size_t *out_len)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "sign: open %s: %s\n", path, strerror(errno));
        return -1;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fprintf(stderr, "sign: fseek %s: %s\n", path, strerror(errno));
        fclose(f);
        return -1;
    }
    long sz = ftell(f);
    if (sz < 0) {
        fprintf(stderr, "sign: ftell %s: %s\n", path, strerror(errno));
        fclose(f);
        return -1;
    }
    rewind(f);
    uint8_t *buf = malloc((size_t)sz + 1);
    if (!buf) {
        fprintf(stderr, "sign: out of memory\n");
        fclose(f);
        return -1;
    }
    if ((size_t)sz > 0 && fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        fprintf(stderr, "sign: read %s: %s\n", path, strerror(errno));
        free(buf);
        fclose(f);
        return -1;
    }
    fclose(f);
    *out_buf = buf;
    *out_len = (size_t)sz;
    return 0;
}

/* -------------------------------------------------------------------------
 * Helper: write buffer to file (overwrite).
 * Returns 0 on success, -1 on error.
 * ---------------------------------------------------------------------- */
static int write_file(const char *path, const uint8_t *buf, size_t len)
{
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "sign: open %s for write: %s\n", path, strerror(errno));
        return -1;
    }
    if (len > 0 && fwrite(buf, 1, len, f) != len) {
        fprintf(stderr, "sign: write %s: %s\n", path, strerror(errno));
        fclose(f);
        return -1;
    }
    fclose(f);
    return 0;
}

/* -------------------------------------------------------------------------
 * Helper: build the 149-byte agentos.signature custom section.
 *
 * section_buf must be at least SIG_SECTION_TOTAL (149) bytes.
 * ---------------------------------------------------------------------- */
static void build_sig_section(uint8_t section_buf[SIG_SECTION_TOTAL],
                               const uint8_t pubkey[32],
                               const uint8_t sig[64],
                               const uint8_t hash[32])
{
    uint8_t *p = section_buf;

    /* Section id: custom (0x00) */
    *p++ = 0x00;

    /* LEB128(146): section body size = 1 + 17 + 128 = 146 */
    *p++ = 0x92;
    *p++ = 0x01;

    /* LEB128(17): name length */
    *p++ = 0x11;

    /* Name */
    memcpy(p, SIG_SECTION_NAME, SIG_SECTION_NAME_LEN);
    p += SIG_SECTION_NAME_LEN;

    /* Payload: pubkey[32] | sig[64] | hash[32] */
    memcpy(p, pubkey, 32); p += 32;
    memcpy(p, sig,    64); p += 64;
    memcpy(p, hash,   32); p += 32;

    /* Sanity check */
    (void)(p - section_buf); /* should be 149 */
}

/* -------------------------------------------------------------------------
 * Helper: compute SHA-256 over one or two memory regions (iovec-style).
 * Convenience for "bytes before section" + "bytes after section".
 * ---------------------------------------------------------------------- */
static int sha256_two(const uint8_t *a, size_t alen,
                      const uint8_t *b, size_t blen,
                      uint8_t out[32])
{
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    if (!ctx) return -1;
    if (EVP_DigestInit_ex(ctx, EVP_sha256(), NULL) != 1 ||
        EVP_DigestUpdate(ctx, a, alen)             != 1 ||
        (blen > 0 && EVP_DigestUpdate(ctx, b, blen) != 1))
    {
        EVP_MD_CTX_free(ctx);
        return -1;
    }
    unsigned int outlen = 32;
    int rc = EVP_DigestFinal_ex(ctx, out, &outlen);
    EVP_MD_CTX_free(ctx);
    return (rc == 1) ? 0 : -1;
}

/* -------------------------------------------------------------------------
 * Helper: Ed25519 sign.
 * Signs `data_len` bytes of `data` with the private seed and returns the
 * 64-byte signature in `sig_out`.
 * Returns 0 on success, -1 on error.
 * ---------------------------------------------------------------------- */
static int ed25519_sign(const uint8_t seed[32],
                        const uint8_t *data, size_t data_len,
                        uint8_t sig_out[64])
{
    EVP_PKEY *pkey = EVP_PKEY_new_raw_private_key(EVP_PKEY_ED25519, NULL,
                                                   seed, 32);
    if (!pkey) {
        fprintf(stderr, "sign: EVP_PKEY_new_raw_private_key failed\n");
        ERR_print_errors_fp(stderr);
        return -1;
    }

    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    if (!ctx) {
        EVP_PKEY_free(pkey);
        return -1;
    }

    /*
     * Ed25519 does not use a separate message digest; pass NULL for md.
     * EVP_DigestSign then operates directly on the supplied data.
     */
    if (EVP_DigestSignInit(ctx, NULL, NULL, NULL, pkey) != 1) {
        fprintf(stderr, "sign: EVP_DigestSignInit failed\n");
        ERR_print_errors_fp(stderr);
        EVP_MD_CTX_free(ctx);
        EVP_PKEY_free(pkey);
        return -1;
    }

    size_t sig_len = 64;
    if (EVP_DigestSign(ctx, sig_out, &sig_len, data, data_len) != 1 ||
        sig_len != 64)
    {
        fprintf(stderr, "sign: EVP_DigestSign failed\n");
        ERR_print_errors_fp(stderr);
        EVP_MD_CTX_free(ctx);
        EVP_PKEY_free(pkey);
        return -1;
    }

    EVP_MD_CTX_free(ctx);
    EVP_PKEY_free(pkey);
    return 0;
}

/* -------------------------------------------------------------------------
 * Helper: Ed25519 verify.
 * Returns 1 if valid, 0 if invalid, -1 on error.
 * ---------------------------------------------------------------------- */
static int ed25519_verify(const uint8_t pubkey[32],
                          const uint8_t *data, size_t data_len,
                          const uint8_t sig[64])
{
    EVP_PKEY *pkey = EVP_PKEY_new_raw_public_key(EVP_PKEY_ED25519, NULL,
                                                  pubkey, 32);
    if (!pkey) {
        fprintf(stderr, "verify: EVP_PKEY_new_raw_public_key failed\n");
        ERR_print_errors_fp(stderr);
        return -1;
    }

    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    if (!ctx) {
        EVP_PKEY_free(pkey);
        return -1;
    }

    if (EVP_DigestVerifyInit(ctx, NULL, NULL, NULL, pkey) != 1) {
        fprintf(stderr, "verify: EVP_DigestVerifyInit failed\n");
        ERR_print_errors_fp(stderr);
        EVP_MD_CTX_free(ctx);
        EVP_PKEY_free(pkey);
        return -1;
    }

    int rc = EVP_DigestVerify(ctx, sig, 64, data, data_len);
    EVP_MD_CTX_free(ctx);
    EVP_PKEY_free(pkey);

    /* rc == 1: valid, rc == 0: invalid, rc < 0: error */
    if (rc < 0) {
        ERR_clear_error();
        return -1;
    }
    return rc; /* 1 or 0 */
}

/* =========================================================================
 * wasm_sign_file — public API for nano.toml integration
 *
 * Signs wasm_path using the keypair at key_path (or default if NULL).
 * Returns 0 on success, -1 on error.
 * ====================================================================== */
int wasm_sign_file(const char *wasm_path, const char *key_path)
{
    char default_kp[4096];
    if (!key_path) {
        if (default_key_path(default_kp, sizeof(default_kp)) != 0)
            return -1;
        key_path = default_kp;
    }

    uint8_t seed[32], pubkey[32];
    if (load_or_gen_keypair(key_path, seed, pubkey) != 0)
        return -1;

    uint8_t *wasm = NULL;
    size_t   wasm_len = 0;
    if (read_file(wasm_path, &wasm, &wasm_len) != 0)
        return -1;

    /* Strip existing signature section if present */
    size_t sig_off = 0, sig_total = 0;
    const uint8_t *clean     = wasm;
    size_t         clean_len = wasm_len;
    uint8_t       *stripped  = NULL;

    if (wasm_find_sig_section(wasm, wasm_len, &sig_off, &sig_total)) {
        size_t new_len = wasm_len - sig_total;
        stripped = malloc(new_len);
        if (!stripped) {
            fprintf(stderr, "sign: out of memory\n");
            free(wasm);
            return -1;
        }
        memcpy(stripped,              wasm,              sig_off);
        memcpy(stripped + sig_off,    wasm + sig_off + sig_total,
               wasm_len - sig_off - sig_total);
        clean     = stripped;
        clean_len = new_len;
    }

    /* Compute SHA-256 of clean bytes */
    uint8_t hash[32];
    if (sha256_two(clean, clean_len, NULL, 0, hash) != 0) {
        fprintf(stderr, "sign: SHA-256 failed\n");
        free(stripped);
        free(wasm);
        return -1;
    }

    /* Sign the 32-byte hash */
    uint8_t sig[64];
    if (ed25519_sign(seed, hash, 32, sig) != 0) {
        free(stripped);
        free(wasm);
        return -1;
    }

    /* Build the 149-byte custom section */
    uint8_t section[SIG_SECTION_TOTAL];
    build_sig_section(section, pubkey, sig, hash);

    /* Write clean bytes + new section */
    size_t out_len = clean_len + SIG_SECTION_TOTAL;
    uint8_t *out = malloc(out_len);
    if (!out) {
        fprintf(stderr, "sign: out of memory\n");
        free(stripped);
        free(wasm);
        return -1;
    }
    memcpy(out,            clean,   clean_len);
    memcpy(out + clean_len, section, SIG_SECTION_TOTAL);

    int rc = write_file(wasm_path, out, out_len);

    free(out);
    free(stripped);
    free(wasm);

    if (rc == 0) {
        printf("pubkey: ");
        hex_print(pubkey, 32);
        printf("\n");
    }
    return rc;
}

/* =========================================================================
 * nanoc_sign_cmd — `nanoc sign <file.wasm>`
 *
 * argv[0] is the WASM filename.
 * Returns 0 on success, non-zero on failure.
 * ====================================================================== */
int nanoc_sign_cmd(int argc, char **argv)
{
    if (argc < 1) {
        fprintf(stderr, "usage: nanoc sign <file.wasm>\n");
        return 1;
    }
    const char *wasm_path = argv[0];

    char key_path[4096];
    if (default_key_path(key_path, sizeof(key_path)) != 0)
        return 1;

    return wasm_sign_file(wasm_path, key_path) == 0 ? 0 : 1;
}

/* =========================================================================
 * nanoc_verify_cmd — `nanoc verify <file.wasm>`
 *
 * argv[0] is the WASM filename.
 * Returns 0 if signature is valid, non-zero otherwise.
 * ====================================================================== */
int nanoc_verify_cmd(int argc, char **argv)
{
    if (argc < 1) {
        fprintf(stderr, "usage: nanoc verify <file.wasm>\n");
        return 1;
    }
    const char *wasm_path = argv[0];

    uint8_t *wasm = NULL;
    size_t   wasm_len = 0;
    if (read_file(wasm_path, &wasm, &wasm_len) != 0)
        return 1;

    /* Find the agentos.signature section */
    size_t sig_off = 0, sig_total = 0;
    if (!wasm_find_sig_section(wasm, wasm_len, &sig_off, &sig_total)) {
        printf("INVALID: no agentos.signature section\n");
        free(wasm);
        return 1;
    }

    /*
     * The section body starts after: section-id(1) + LEB128-size(2) = 3 bytes.
     * Then: LEB128-name-len(1) + name(17) = 18 bytes before payload.
     * Payload offset within section: 3 + 18 = 21 bytes from sig_off.
     */
    size_t payload_off = sig_off + 1 /* id */ + 2 /* LEB size */ +
                         1 /* LEB namelen */ + SIG_SECTION_NAME_LEN;
    if (payload_off + SIG_PAYLOAD_LEN > wasm_len) {
        fprintf(stderr, "verify: section payload truncated\n");
        free(wasm);
        return 1;
    }

    const uint8_t *payload       = wasm + payload_off;
    const uint8_t *stored_pubkey = payload + SIG_PUBKEY_OFF;
    const uint8_t *stored_sig    = payload + SIG_SIG_OFF;
    const uint8_t *stored_hash   = payload + SIG_HASH_OFF;

    /* Compute SHA-256 of the file excluding the signature section */
    uint8_t computed_hash[32];
    if (sha256_two(wasm,              sig_off,
                   wasm + sig_off + sig_total,
                   wasm_len - sig_off - sig_total,
                   computed_hash) != 0)
    {
        fprintf(stderr, "verify: SHA-256 computation failed\n");
        free(wasm);
        return 1;
    }

    /* Check stored hash matches computed hash */
    if (memcmp(stored_hash, computed_hash, 32) != 0) {
        printf("INVALID: hash mismatch\n");
        free(wasm);
        return 1;
    }

    /* Copy pubkey to local buffer before freeing wasm */
    uint8_t pubkey_copy[32];
    memcpy(pubkey_copy, stored_pubkey, 32);

    /* Verify Ed25519 signature over the 32-byte SHA-256 hash */
    int vrc = ed25519_verify(pubkey_copy, computed_hash, 32, stored_sig);
    free(wasm);

    if (vrc == 1) {
        printf("OK: pubkey=");
        hex_print(pubkey_copy, 32);
        printf("\n");
        return 0;
    } else {
        printf("INVALID: bad signature\n");
        return 1;
    }
}

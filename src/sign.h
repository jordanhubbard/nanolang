/*
 * sign.h — Ed25519 WASM module signing public API
 *
 * nanoc_sign_cmd:   `nanoc sign <file.wasm>`   — sign with ~/.nanoc/signing.key
 * nanoc_verify_cmd: `nanoc verify <file.wasm>` — verify agentos.signature section
 * wasm_sign_file:   programmatic signing (used by the build pipeline)
 */
#ifndef SIGN_H
#define SIGN_H

int nanoc_sign_cmd(int argc, char **argv);
int nanoc_verify_cmd(int argc, char **argv);
int wasm_sign_file(const char *wasm_path, const char *key_path);

#endif /* SIGN_H */

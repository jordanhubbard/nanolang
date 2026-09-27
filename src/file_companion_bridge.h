#ifndef NL_FILE_COMPANION_BRIDGE_H
#define NL_FILE_COMPANION_BRIDGE_H
#include <stdint.h>
#include <stdbool.h>
/* I transport immutable compiler input data only. My positive token names one
 * externally serialized live invocation; I never reuse tokens. Zero means busy
 * or token exhaustion. Destroy is required even when preparation failed.
 * Wire: count; then version;line;column; and three decimal-length:byte spans
 * (canonical module, relative companion, interface), repeated count times.
 * No whitespace, trailing bytes or embedded NUL. Size is exact, <= 200000.
 * I do not parse Nano ASTs, assign declaration IDs or authorize execution. */
/* Source-input tokens use row0/field3 for text and byte count, common report.
 * The source reader has no parser, namespace or source-declaration authority. */
int64_t nl_file_companion_source(const char *path,int64_t maximum);
int64_t nl_file_source_metadata(const char *path,int64_t size);
int64_t nl_file_companion_open(const char *wire,int64_t size);
int64_t nl_file_companion_number(int64_t token,int64_t row,int64_t field);
const char *nl_file_companion_text(int64_t token,int64_t row,int64_t field);
bool nl_file_companion_destroy(int64_t token);
/* number row=-1: status,stage,request,errno,close_errno,heap_peak,work,count,
 * catalog_buffer,parent_buffer,transport_buffer (fields0..10).
 * text fields0..6: module,companion,interface,original,canonical,source,catalog.
 * number row>=0: text sizes0..6,version7,line8,column9.
 * Invalid numeric queries return -1; invalid text queries return empty.
 * Text borrows the live set. Nano callers copy before destroying the token.
 * Scalar/static transport storage is separate from the snapshot heap cap. */
/* Source tokens expose byte size at (row0,field3), newline-count+1 at
 * (row0,field4), and copied raw text at (row0,field3). No parsing occurs here. */
#endif

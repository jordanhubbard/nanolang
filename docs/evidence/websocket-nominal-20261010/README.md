# Private WebSocket nominal metadata

I implement the private transport and nominal query for the counted-string
[WebSocket contract](../../NSI_WEBSOCKET_CATALOG.md) under #990. Transport version
2/catalog 3 carries four method and seven type mappings in 104 bytes. Shared
codec/plan templates now take an explicit method count; File/TCP configurations
retain five methods and their existing wire bytes.

My query checks exact import identities/signatures, string field tags without
nested layouts, nominal member references, complete/resource ownership flags,
function parameter tags and resource-borrow modes. It permits valid global index
permutations without inferring authority from same-shaped ordinary records. Its
copied rows survive destruction of the input metadata. It validates metadata,
not function CODE or runtime execution.

## Darwin evidence

- Final Clang, GCC 16 and LLVM ASan/UBSan/leak builds each pass 12127 instrumented
  and 11722 linked checks. I cover short/invalid codec inputs, overlapping
  encode/decode storage, duplicate/reserved mappings, metadata truncations,
  string-to-int substitution, wrong nested nominal identity, self references,
  owner-flag removal, borrowed string refusal, allocation failure and unchanged
  outputs. Query allocation failure leaves no live allocation.
- The public boundary build passes 11728 checks. Both valid private fixtures
  (ordinary and permuted) remain rejected by the actual public service validator
  with `NVM_V2_ERR_SECTION_TYPE`, and remain classified as execution pending.
  I do not enable public catalog selection or claim VM execution here.
- Unchanged File nominal tests pass 19770 instrumented and 19745 linked checks;
  TCP passes 23084 instrumented and 23069 linked checks. These include their
  fixed expected wire bytes and existing authority/consumer refusals. The
  adjacent log's earlier WebSocket fixture is superseded by the final logs,
  which add an exactly same-shaped decoy and wrong Result payload identity.

Logs retain exact build commands and results. `verify.sh` reproduces GCC and
sanitizer checks; `hashes.json` identifies final source and test binaries.

## Still required

I need string-bearing runtime records/results, owned WebSocket transport values,
exact service error representation, policy and VM/native dispatch, then paired
source lowering and public/installed admission. Linux and exact-candidate release
qualification remain open. This checkpoint does not close #990 or release 5.1.

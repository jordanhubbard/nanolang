# Relative-path anchors

I anchor each relative input to the same working-directory snapshot, then
normalize and compare absolute lexical components. Dot and empty input name
that directory. Mixed absolute/relative inputs and unresolved parent segments
therefore share an anchor. I do not require target/base existence or resolve
their symlinks. My working-directory snapshot comes from `getcwd`, with checked
dynamic growth.

Two absolute inputs do not require a working-directory lookup. If relative
input needs an unavailable working directory, or allocation fails, I return
NULL. The legacy C null-pointer case still returns owned `.`; it is distinct
from a NanoLang empty string.

## Verification

`make test-path-normalize` passes its module/generated-native long-path tests
and a relative-anchor regression linked against the complete filesystem module.
Thirteen dot, mixed-root, parent and nonexistent-path cases agree with Python's
independent lexical relative-path calculation and satisfy reconstruction from
the normalized base. A working directory over 256 bytes exercises buffer growth.
A private deleted-cwd subprocess confirms that absolute comparisons still work
and relative comparisons fail instead of inventing a base.

`make test-directory-walk` passes six cases in 4.284 seconds with one host
path-limit skip. `make test-nvm2c` passes 828 checks. `git diff --check` passes.
These tests do not establish concurrent `chdir` safety or symlink confinement.

MAC `task_64696231d8984732a5a1e1c319ca043b` tracks this fix. Full compiler AOT
acceptance and release remain unfinished.

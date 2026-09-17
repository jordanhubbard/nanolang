# My explicit host imports

I lower a bounded table of scalar host ABI signatures. Declared externs keep
ordered import metadata; implicit OS builtins use the C seed's `vm_*` names.
I reuse imports for repeated calls, evaluate arguments in order, and use
`CALL_EXTERN` followed by `RET` for returned host values. A defined local
function takes precedence over an implicit host builtin.

On Linux ARM64, `make -j8 test-nanoisa-src-nano` passes 86 baseline checks and
ten focused Python cases. My new host fixture compares 15 module/function
checks, including ordered import names, kinds, argument tags and return tags.
Both C-seed and self-hosted modules verify and execute `getenv`, declared
`nl_os_getenv`, argument count/access, and file existence in NanoVM and strict
C11 native output. Both produce `host-value`, `2`, and `sentinel` on separate
lines with a supplied environment and guest argument. Eight malformed extern
signatures/calls fail without publishing assembly.

My table covers the scalar host functions declared by the current compiler
and a bounded set of scalar OS builtins. This is not arbitrary FFI: unknown
symbols, nonmatching signatures, aggregate arguments/results and callbacks
remain unsupported. My raw-source path does not resolve module imports or
encode module artifact bindings. Runtime host resolution remains required;
native translation independently checks its own host adapter table and can
refuse an otherwise valid VM import.

I reran `src_nano/nanoc_v06.nano` emission. It now passes the host-call boundary
and first refuses `unsupported result type HashMap<string,int>`. I recorded
that continuation as `task_2c74662f99d44f7faac1fe7427e44325`; full compiler
emission and bootstrap equality remain open. This host slice is tracked by
`task_cc32b93698db405696235e96ce8fc194`.

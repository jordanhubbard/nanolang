# C File driver publication

I connect the C seed and `nano_virt` to actual File source compilation under
#989, from parent `523ff2c4d` on Darwin. I keep the full 5.1 scope open.

## Behavior

I add a loader entry point for consumers that independently lower checked File
graphs. Ordinary interpreter/editor consumers retain their refusal. I retain the
complete parsed graph until lowering finishes, check every body and ownership
flow, and derive main plus selected dependency/root shadows from exact retained
source identities. Root-only selection changes execution selection; it does not
skip type checking imported shadow bodies.

I pass emitted bytes to a shared publication transaction. I validate the main
module through the real native consumer without executing it, supervise all
selected shadows with fresh grants, and stage either binary NanoISA or native C
and its standalone executable. Source and immutable companion aliases, including
hardlinks, refuse before staging. Native compilation uses configured compiler and
flags, defaulting to `cc -O1`, under a separate 120-second process-group bound.
I finish auxiliary staging cleanup before replacing the destination atomically
with a file on the same filesystem. The publication API itself accepts no AST.

`--allow-temporary-files` grants this compilation's selected shadows. It is not
stored in the output. Native launchers require their own runtime flag, create,
revoke and destroy their invocation grant, and return the scalar's low byte.
`nano_virt --run` explicitly grants and executes the byte module when the flag is
present. A source graph without selected shadows can emit bytecode without a
compile-time grant; running that bytecode still requires an explicit grant.
Without `-o`, `nano_virt` checks/runs the selected shadows without publication.
Both C drivers accept `--emit-nvm` for these graphs; C-seed ordinary source still
uses its existing backend. Unqualified File output modes and instrumentation
options refuse before shadow execution or output changes.

I locate runtime resources from compiler paths, including basename invocation
through PATH, or explicit NANO_ROOT. My tests invoke both entry points from an
unrelated directory. Source checkout and installed-header layouts are recognized;
recognition alone is not installed platform qualification.

## Verification

- `neighbors.log`: 15 publication, immutable-companion, namespace, nominal-body
  and affine-ownership methods pass in 116.320 seconds. Independent Nano body and
  ownership probes run in VM and sanitized native execution; this run uses the C
  producer for those probes. Negative typing/ownership and default generic-loader
  refusal checks remain enabled. Earlier driver tests now accept successful checked
  byte publication or the exact remaining grant/lowering refusal instead of
  requiring every valid graph to hit the retired blanket guard.
- `path-tests.log`: seven final CLI and ordinary packaged-wrapper methods pass in
  13.453 seconds. Both C source routes emit identical File bytes, execute the five
  unchanged generated shadows with matching selected/start/completion records,
  run the resulting VM module and native program, and refuse absent runtime grants.
  Native symbol checks exclude `vm_execute` and `nvm_file_execute_cyclic_bytes`.
  I test imported failing shadows, explicit root-only selection, invalid unselected
  shadows, absent compile grants, source/companion hardlink aliases, failed native
  compilation, invalid deadline configuration, directory destinations, unsupported
  options, PATH invocation and no-output checking. Prior output and staging cleanup
  are checked on failures. Ordinary wrapper behavior and its documentation pass.
- `final-sanitized.log`: all four File CLI methods pass in 15.477 seconds with LLVM
  ASan/UBSan on `nano_virt` main, the loader, C File driver/lowerer, byte publication
  and supervisor. Other common objects and the File archive remain ordinary builds;
  I do not claim whole-runtime instrumentation. The C seed is ordinary in this run.
- `tests.log`, `final-tests.log` and `sanitized.log` retain the earlier passing
  integration/control runs before the PATH qualification was added.
- Build logs retain strict compiler commands; I remove trailing whitespace from
  `sanitized.log` for repository whitespace checks. `inputs.json` pins final sources.

## Still required

The independent Nano lowerer remains implemented and separately qualified, but
its actual driver still needs the byte-only host bridge and publication call.
I have not qualified every source/profile combination, richer/indirect borrows,
multiple catalogs, full installed Linux/Darwin behavior, new release bootstrap or
all Socket/backend requirements. This batch is actual C source publication, not
completion of #989 or 5.1. I preserve the user's untracked guide source unchanged.

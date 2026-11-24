# My checked source-C import graph

I emit one self-contained C99/C11 translation unit for imports that fit my
bounded hosted C profile. I retain each checked declaration and its original
source owner. A qualified or selective alias selects that declaration; a
matching function signature alone does not select a body.

`value.nano`:

```nano
pub fn answer() -> int { return 42 }
shadow answer { assert (== (answer) 42) }
```

`main.nano`:

```nano
module "value.nano" as values
fn main() -> int { return (values.answer) }
shadow main { assert (== (main) 42) }
```

```sh
bin/nanoc_c --target c main.nano -o program.c
cc -std=c99 -pedantic-errors -Wall -Wextra -Werror program.c -lm -o program
./program
printf '%s\n' "$?"
```

The program exits with status 42. Source-only emission checks the selected
profile and emits source; it does not execute dependency or root shadows.
Ordinary native compilation has its own supervised shadow gate.

I keep imported private helpers, scalar globals and root declarations distinct.
Different aliases of one source share the same declaration and global storage.
I initialize dependencies before their importers. I resolve a module body's
calls in its declaring namespace, including its own transitive aliases. My
private generated names avoid identifiers already present in the emitted plan.
Extern aliases retain their declared C symbol and trusted signature.

I evaluate declared direct and qualified arguments once, left to right, through
exact signature-typed local slots and C comma sequencing. These slots retain
supported scalar, record and union values. Existing scoped scalar/local-union
argument values retain enclosing returns: an earlier return prevents later
arguments and the call from running. Arrays, first-class callable values and
the other unsupported profile representations still receive checked refusal.
I retain nominal declaration identity rather than treating compatible layouts
as interchangeable types.

My compiler-owned support remains `static inline`. GCC and Clang receive a
private unused annotation for support that a particular program does not call;
other preprocessors retain plain C declarations. Normal diagnostics remain
active for user declarations. My consumer tests retain their strict standard
and warning flags.

My existing `c_backend_emit` and `c_backend_emit_fp` APIs accept a root AST and
keep their original options layout. My separate `c_backend_emit_graph` API
borrows a flattened declaration view, exact original owner programs and a
synchronous checked-function resolver. I require each runtime declaration of
each supplied owner exactly once. A foreign member, wrong owner, omission or
unresolved callee fails before output publication. The caller keeps all ASTs,
checked annotations and source owners alive until emission returns. I do not
mutate or free those borrowed declarations.

The CLI adapter lends its already loaded and checked graph. It restores the
original environment owner and file immediately after every resolver lookup.
I keep source-C output staged until emission succeeds, preserving prior output
on profile, graph, allocation and I/O refusals. Root-only API capability checks,
private visibility and exact nominal/callable/array guards remain required.

`make test-source-c-imports` runs the graph API controls and source/native
consumers. Release evidence separately records tested pins, providers, host
platforms and broader bootstrap acceptance.

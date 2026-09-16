# My union resource classification

I extend the record fixed point from `16e47ba8` to registered named unions.
Any variant field that reaches a resource makes the union resource-bearing;
records containing that union inherit the obligation. Shared iteration handles
record/union cycles without recursive depth limits. Each payload type is
resolved in its declaring module, preserving the querying module context.

`make test-resource-classification` exits zero. It retains the 300-record
cycle checks and adds empty variants, a resource in a later payload field,
record/union cycles, union self-cycles with and without a resource path, and
same-named union definitions in two modules. Removing the resource path
removes the classification on the next query. Host-local log:
`/tmp/nanolang-union-resource-classification.log`.

This is named-type classification, not generic substitution, tuple analysis,
collection enforcement or path-sensitive ownership. The failing C-seed
parity baseline remains open. Allocation failure continues to report failure
and conservatively retain the obligation; failure injection is not claimed.

MAC `task_91ae827be4154eaa8f22698aeecc8cf1` is running under Natasha when
inspected. I attach evidence and coordinate without changing task ownership.

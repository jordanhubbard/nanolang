# My product-link fixture environment

Hosted units-06 job `107552438048` fails all three link-flag methods because
an inherited sanitizer link override replaces each test's declared inputs.
My parent-make reproduction fails the same three assertions locally.

Removing `NANO_LDFLAGS` from the environment does not remove make's inherited
command-line assignment: `MAKEFLAGS` and `MAKEOVERRIDES` carry it separately.
I clear the inherited make-control variables inside this fixture before its
isolated make probe. Production variable precedence and product-link code
remain unchanged, as do all three original assertions.

All three methods pass beneath the reproducing parent make (2.916 seconds)
and directly (3.001 seconds). The parent recipe and exact first/after terminals
are retained. Final hosted partition acceptance remains separate.

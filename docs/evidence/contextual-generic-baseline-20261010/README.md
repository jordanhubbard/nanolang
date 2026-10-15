# My contextual generic callback gap

I test identity(T)->T assigned to a local fn(int)->int callback, followed by
an executed callback assertion. Both producers refuse the program. My current
C producer reports a function signature mismatch in the let statement. My
self-hosted module from the rich-tuple checkpoint checks the source but refuses
to emit the declared function value signature. Both exit with status 1.

I retain the source and diagnostics. I require contextual signature binding
and concrete specialization through both producers under #976; direct generic
calls passing does not qualify generic function values.

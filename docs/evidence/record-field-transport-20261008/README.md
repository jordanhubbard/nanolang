# Exact and tagged record-array fields

I retain the two-direction parser reduction in the preceding checkpoint.
My expanded initial matrix has 12 native representation refusals and four
Boolean assembler failures (`true`/`false` where raw bytecode requires 1/0).
After normalizing record-field carriers, the 12 assembled cases pass. I correct
the Boolean fixture spelling, preserve both failed logs, and add collection
stress to all 16 scalar transport cases: integer, Boolean, float and owned
string, exact/tagged in both directions, replacement and append.

I constrain the payload separately from its optional wrapper. Native writes
copy the record and normalize carrier tags, checking presence and exact payload
tags when unboxing. I retain array ownership and runtime field compatibility.
A nested-array record-field control clears the global root and requires the
record to keep its owned string child alive through actual collections.
Wrong payloads retain prior generated output. An absent integer lookup fails
in both VM and sanitized native execution instead of becoming an exact field.

The previous parser run's referenced temporary program.nvm is no longer
present. I preserve that failed inspection; no replay qualification is claimed.
Fresh compiler/bootstrap and the unchanged full parser corpus remain required.

My final focused gate passes all 21 methods in 18.660 seconds, including
the full byte/nested/source-slice corpus and four new transport methods.
Existing record scalar-tag and float-record gates pass. The full native gate
passes 2,434 execution checks, 3,098 shape checks and 379 callable checks.
The source/tool hashes in `tested-inputs.json` remain unchanged afterward.
Fresh installed-stage/parser acceptance is still pending.

# Parser record-field reduction

I reduce the integrated parser's `ARR_SET` kinds 8/0 failure to a one-field
record array. Both directions execute and assert the correct replacement value
in NanoVM. Native translation refuses exact integer into tagged integer storage
(kinds 8/0), and tagged integer into exact integer storage (kinds 0/8).

I retain both failures and their raw modules' assembly and commands. The
production correction remains open under #978; these controls do not replace
the complete parser corpus or its installed-stage/platform gates.

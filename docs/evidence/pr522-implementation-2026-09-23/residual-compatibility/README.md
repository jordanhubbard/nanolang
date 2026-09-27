# My retained compatibility baseline

At source commit `180422cd0`, I run the unchanged union-literal context,
native generic-record-field and aggregate-match suites with fresh Stage 1.
Seven of 23 methods pass; 16 fail. `baseline.json` names every failure and
records the command, compiler selection and uncompressed terminal hash.

My failures still separate into record arrays stored in unions, concrete generic
union fields stored in ordinary records, and aggregate/integer-selector match
results. Diagnostics refuse unsupported local types, exact union identity or
expression-arm result shapes. I retain these acceptance cases unchanged; source
refusal is still a compatibility failure for PR #522. This run does not cover
all retained hosted failures or qualify either other compiler.

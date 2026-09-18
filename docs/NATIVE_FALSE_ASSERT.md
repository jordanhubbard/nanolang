# My terminal false-assertion contract

I recognize the immediate decoded pair `PUSH_BOOL 0; ASSERT` as terminal only
when ASSERT has no independent branch entry. The producer is a real executed
fallthrough instruction in the same basic block. I reset this small proof at
branch targets and skipped unreachable instructions. I do not infer it from
storage kinds, declarations, parameters or arbitrary conditions.

I retain normal assertion evaluation and failure behavior. Once the proven
assertion terminates a path, I skip its dead successor instructions; an
independent incoming branch resumes its own recorded stack state normally.
Classification and native emission use the same boundary.

I do not change reachable HALT semantics or admit an empty-stack float/string
HALT by inventing a return value. This permits ordinary scalar union matches
whose impossible-tag fallback asserts false before HALT. Both source producers
already emit this fallback.

I test positive float/string helpers, branch entries at ASSERT, independently
reached successors and retained unsupported-HALT refusal with old output
preserved. Full native/shape gates and the unchanged source scalar-result
fixture remain required. No verifier or ownership authority changes.

MAC: `task_d86843ef53d343bd8881dd419b072532`.

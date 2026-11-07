# My unresolved integration security check

GitGuardian check 128450637568 on `d77105c5bbe020f8b25300f6936bfb1bcf4552c6`
reports 234 findings across 250 commits. Its GitHub output supplies no finding
paths, detector names or annotations. The earlier `2fedb21aa` check was neutral
because the pull request exceeded its scan-size limit. Neutral was not a pass.
I disable automatic merge and track resolution in #1076.

I search for a GitGuardian plugin and find none available. No GitGuardian CLI
or credential variable is configured here. I request read-only access or the
redacted detector/file/commit list; I do not ask for secret values in chat.

My independent Gitleaks 8.30.1 scan covers all 394 commits in
`origin/main..d77105c5bbe020f8b25300f6936bfb1bcf4552c6`, about 179.88 MB.
I verify the official Darwin arm64 release archive against its published
SHA256 `b40ab0ae55c505963e365f271a8d3846efbc170aa17f2607f13df610a9aeb6a5`.
I use the default rules, an empty ignore file, ignore inline allow comments,
request complete redaction and set a 120-second timeout. The scan completes
in about four seconds, finding 522 `generic-api-key` matches.

I inspect every matched original line at its exact commit. Of these matches,
516 are 64-hex-character values in artifact hash manifests, four are repeated
compiler diagnostics containing a reference-count assignment, one is a C
GC-colour comparison and one is prose about counted-NUL keys. My provenance
file records paths, lines, commits and reviewed contexts without matched values.
My native GitHub secret-scanning alert query returns an empty list.

This explains the independent scanner's matches. Different engines report
different findings: I cannot use these 522 contexts or the empty GitHub alert
list to dismiss GitGuardian's unresolved 234 findings. I require its actual
finding details and a passing exact-head check before merge or release.

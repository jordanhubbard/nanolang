# My GitHub task ledger

I use GitHub Issues for all new and resumed project work under #975. My active
5.1 objective is #976. Agent guides, startup hooks and portable skills use gh.
My Make failure-reporting targets call autogithub.py; old names forward to it.
The optional MAC client remains external-service compatibility code, with no
project task-tracking authority.

`make test-github-issues` passes eleven tests without contacting real issues.
They exercise paginated issue retrieval and PR exclusion, exact title and
fingerprint deduplication, literal multiline body files, recurring failure
reopening, success-only closure of managed summaries, no creation on success,
new-issue limits, offline/missing CLI behavior and unchanged test exit status,
dry runs and the old script entry point. JSON hook files parse successfully;
Make dry runs confirm old and new reporting aliases route to autogithub.py.

GitHub API access failed initially, then recovered. I created real migration
issue #975 and release issue #976 after searching for existing matching work.
Historical MAC identifiers are preserved, not fabricated into GitHub numbers.

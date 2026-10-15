# I recover interrupted package installation

Under #982 I retain the x64 strict-examples failure from
[CI38015642804](https://github.com/jordanhubbard/nanolang/actions/runs/38015642804/job/114105165733)
at `d8c4c495c`. The first apt attempt exits137 while unpacking packages.
Both subsequent attempts exit100 because dpkg requires `--configure -a`.
I retain the [terminal excerpt](failure-excerpt.log). This job never reaches
compiler/example qualification. The log does not establish why the first
process was killed.

I run pending dpkg configuration and apt dependency repair at the start of
each subsequent attempt, inside its existing timeout. Configuration may report
missing dependencies, so I preserve that diagnostic and require apt's
`--fix-broken install` to succeed before normal update/install. Attempts,
deadlines, backoff and workflow budgets stay unchanged.

My [stubbed helper tests](helpers.log) reproduce interrupted unpacking, failed
configuration followed by successful dependency repair, and persistent repair
failure. They also retain ordinary retries, budget exhaustion, exit diagnostics,
usage, workflow call-site and no-timeout-host checks. No local package manager
or sudo operation runs. All checks pass; all five [workflow checks](workflow.log)
also pass. Actual Linux dependency recovery still requires the next CI result.

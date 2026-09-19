# I print the literal payload without its delimiter

At `a3b6ef729`, I correct the length passed to `str_substring`: a dot-quote token has two prefix bytes and one closing delimiter. I retain empty literals and existing payload whitespace. My builtin shadow checks literal handling preserves stack depth and its existing value.

Complete corrected source/shadow compilation passes using the unchanged PR840-qualified C seeds: Linux8.342 seconds, Darwin12.961 seconds. My [focused source file](forth-string-output/strings.fth) checks empty, one-character, multiword and leading-space payloads, followed by ordinary arithmetic. Actual stdout equals the [expected bytes](forth-string-output/expected.bin) on both hosts, including newlines and spaces. Commands, exact input hashes and outputs are in my [sealed reports](forth-string-output/report-sha256.json).

The previous trailing delimiter remains visible in the earlier retained demo logs. I close only task6c80e7 after canonical merge; I do not claim complete Forth string grammar or overall release acceptance.

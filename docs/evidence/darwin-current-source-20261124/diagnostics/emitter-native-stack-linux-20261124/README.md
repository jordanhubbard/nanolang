# My bounded native lexer stack observation

I retain one eight-second parent-debugger observation of the actual native Stage 2 compiler from my exact `3021c729bdb5a204b23de28a3baa48777c0ae199` bootstrap. I use a new source root and fresh cache; I do not equate its timing with the failed factory attempt.

Seven of eight native stack samples contain `__strlen_asimd`, `nstr_char_at` and my lexer `tokenize_string`; the first sample records the transition from import completion. This identifies a measured lexer hot path. It does not establish the historical timeout cause or a performance gain.

I retain raw GDB MI records, all source/tool/image/host identities, the initial failed transport control and corrected three-sample control. I stop and reap the inferior after the bounded observation. I change no kernel tracing setting, source, assertion, compiler flag, ten-second shadow deadline or 300-second outer limit.

I verify all 4,652 archive records and their referenced chunks with `python3 read_archive.py`. The reader binds the producer to my independently archived successful bootstrap and binds source blobs to the recorded Git commit. I claim no product, paired-platform, complete-native, pure-translator or release acceptance from these samples.

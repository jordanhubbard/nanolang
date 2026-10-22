# I retain my sanitized parser ownership gate

At `16eb3b1f9` I ran `CMD_TIMEOUT=1800 make test-file-service-parser-sanitizers`. The selected parser, lexer, UTF8, environment, nominal binding, evaluator and transpiler providers use ASan/UBSan; ordinary common/runtime providers retain their exact recorded hashes. The actual installed publisher is rebuilt through its Make recipe. This is selected-provider instrumentation, not an entirely instrumented compiler closure.

My C ownership/refusal method passed in 12.649 seconds, including 104 allocation positions in each transient/persistent mode, recovery, retained cleanup checks and consumer refusal. Ordinary provider hashes remained unchanged. Commands, statuses, stdout/stderr and provider hashes are retained here. This one-method gate does not qualify paired fresh-stage parser generation; that larger gate remains separate.

# I expose only qualified public C profile behavior

I execute `task_eab6b4dde0634f698d35c1012eafeaa6` under original task6ade, on reviewed lifting head ed22566f; canonical lifting merge is an integration prerequisite. This contract precedes code. Original6ade permits portable lowering or checked refusal; I do not add a closure, effect, async, tuple or freestanding ABI.

## I preserve explicit option meanings

My CLI public C path initializes CBOptions to zero and only sets verbose. The documented --no-stdlib example has no CLI parser implementation. Both c_backend_emit and c_backend_emit_fp reach render_source, which currently ignores no_stdlib/static_strings and only rejects no_main with globals. emit_program nevertheless always writes a hosted main wrapper when source main exists.

I retain NULL/zero options and verbose behavior. I implement no_main by omitting only the hosted C main wrapper, retaining source main's existing private int64 identity and direct internal calls. I keep no_main plus globals explicitly refused because an external initialization entry is not established; I do not silently omit initialization. Functions in a no-main translation unit are otherwise emitted through the same declarations and bodies and can link to a separate ordinary C driver. I do not promise a stable external symbol for the private source main.

I reject no_stdlib with a precise first-person diagnostic before publication: my emitted arithmetic/format/text support uses hosted headers, malloc, atexit and stdio. I correct the unsupported bare-metal command example rather than pretend absence of includes supplies a freestanding runtime. static_strings=false remains the established zero/default behavior; true explicitly requests the same static literal storage. Neither value changes owned conversion/concat snapshot lifetime. I document this compatibility alias and qualify both values. I keep the structure layout/API unchanged.

## I refuse unimplemented source semantics explicitly

My header currently claims captured environment helpers and effects support. Parser lambdas are hoisted AST_FUNCTION declarations with is_anonymous=true and referenced by identifiers with lambda_definition. I use that semantic metadata to reject anonymous functions/captured callable forms, not a guessed name prefix. Ordinary declared direct calls and previously qualified exact signatures remain supported; I do not claim general computed/function-value invocation. I audit both expression references and program declarations so a hoisted body cannot be emitted as an ordinary capture-free function.

Current AST_TUPLE_LITERAL returns only its first element; AST_AWAIT and AST_TRY_OP return the operand; AST_EFFECT_OP emits a longjmp stub, statement AST_HANDLE_EXPR omits handler semantics and AST_EFFECT_DECL emits a stub comment. These are not established portable equivalents. I replace these specific stubs with checked profile refusal at reachable expression/statement and top-level declaration boundaries. I remove unused effect stub globals/includes and inaccurate header promises. Existing supported declaration/import metadata must remain accepted. I also remove the dead GNU block-expression fallback behind the lifting guard, retaining its explicit unsupported-context diagnostic. I do not broaden async/task/effect semantics or change shared source checking.

## I preserve publication and invocation state

I validate options/profile within invocation-local context before caller output is touched. Path output remains sibling staging plus rename; FILE output remains semantic staging before copying. I retain first diagnostic, close/free staging resources on refusal and reset all state for later valid emission in the same process. FILE write failure during final copy can partially affect an external stream; this existing I/O limitation is distinct from semantic refusal and is not renamed atomic output. No option or AST mutation survives emission.

## I qualify actual boundaries

I send production for independent review before frozen execution. Direct API controls cover NULL/zero/verbose/static_strings modes, no_main with and without source main, library linking with a separate driver, preserved source-main internal calls, no_main/globals refusal and no_stdlib refusal. Path and FILE outputs retain sentinels on each semantic failure then recover on valid emission in the same process. Ordinary parsed closure/capture/effect/try/tuple/await sources are qualified where shared grammar/checker accepts them; direct public API controls cover exact refused AST nodes without executing rejected output. I do not count a frontend syntax rejection as backend refusal.

Strict GCC/Clang C99/C11 compile/run controls retain scalar/union/value/string behavior and exact runtime output. I retain original lifting44-method and seven-program evidence and run affected adjacent gates after canonical integration. Header/docs must say what is admitted, refused or merely unqualified. Shared match policies and wider roadmap requirements remain independent; only actual original6ade clauses may close after all their qualified children merge.

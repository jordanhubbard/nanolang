# My strict optional native helper boundary

I retain both first 303 native-source failures after fresh full bootstrap and all
five focused configurations passed. C-seed O0/O2 passed the first declared-call
vector; Stage1 O0 refused strict compilation of optional static support. The
native eight-method suite stopped at its first method, and the original eighteen
methods remain unreached. I do not infer complete list parity from bootstrap.
MAC `task_02a1b4af967eafa357fb0210e94679d8` tracks this prerequisite under
`task_2752a051dce443d0ada447c46b667561`.

I preserve helper definitions, signatures, storage class, link visibility,
preprocessor decisions, arithmetic source identities and every compiler flag.
My proposed emitter helper returns only statements of the exact form
`    (void)function_name;`. A discarded function designator undergoes C's function-to-pointer conversion;
I emit no explicit address storage or observable invocation, evaluate no capture
or argument expressions, and read no runtime data. I add these
statements to the real C `main` from `gen_c_program_with_modules` and the real
shadow C `main` from `transpile_parser_mode`. The shadow product still includes
the renamed ordinary entry, and its actual entry retains the original
`nl_run_shadow_entry(__nano_shadow_entry, 10)` call. I add no C helper definition,
constructor, exported symbol or runtime callback. The lower-level string
assembler `gen_c_program` generates no entry and is not a complete standalone
native product; I do not put block statements at its file scope.

My exact catalog contains these 52 existing function declarations:

```text
nano_rt_f64_arithmetic_result
nano_rt_f64_add
nano_rt_f64_sub
nano_rt_f64_mul
nano_rt_f64_div
nano_rt_f64_nonfinite
nano_rt_f64_format
nano_rt_f64_print
nl_array_slice
nl_bytes_from_string
nl_string_from_bytes
nl_path_normalize
nl_path_join
nl_path_basename
nl_path_dirname
nl_path_relpath
nl_file_read
nl_fs_mkdir_p
nl_file_copy
nl_dir_copy
nl_get_argv
nl_getenv
nl_env_get
nl_exec_shell
nl_exec_capture
nl_os_getcwd
nl_os_exit
nl_os_chdir
nl_os_file_exists
nl_os_tmp_dir
nl_os_mktemp
nl_os_mktemp_dir
cast_string
to_string
nl_map
nano_rt_reduce_int
nl_filter
nl_map_str
nano_rt_reduce_str
nl_filter_str
nl_map_float
nano_rt_reduce_float
nl_filter_float
nl_filter_bool
nl_map_new
nl_map_put
nl_map_get
nl_map_has
nl_map_size
nl_timing_get_microseconds
nl_timing_get_nanoseconds
nl_get_time_ms
```

I derived this catalog from the retained actual first-failure generated C at
`/home/jkh/nanolang-qualification/record-lists-303-linux-matrix/native-source/nano-native-record-lists-7cc0tx_8/temporary/nano_native_1789982242_0/program.c`, SHA256
`7268f8556ab9aa54ffb9ce7489a255a2b2c7f6517ae16213b19ea083565899d0`, then matched each declaration to
`gen_c_runtime`, `gen_binary64_arithmetic_runtime` or
`gen_binary64_format_runtime`. None of these 52 declarations is conditionally
removed by the current emitter. The binary64 include guards prevent duplicate
definitions while retaining the same declarations. `CLOCK_REALTIME` and
`__MACH__` select bodies of timing helpers, not whether the names exist.
`NL_HAS_USER_FLOAT_TO_STRING` controls a different, noncatalog function; I do not
refer to that function or change its guard. I do not include generated shadow
functions, user declarations, or actual list-specialization operations in this
fixed runtime catalog. If a catalog declaration later becomes conditional, its
reference must acquire the identical guard rather than broadening emission.

I will add a shadow that checks the exact statement spelling/count and absence
of call syntax, plus assertions that ordinary and real shadow entries contain
the reference block before their existing work. All existing runtime-body,
shadow-generation and eighteen-plus-eight source assertions remain unchanged.

Discarded function designators have no runtime helper effects, but I do not
assume a compiler will leave object/link dependencies unchanged. Before claiming
this repair I will retain O0 objects and compare their undefined-symbol sets and
actual link closure against the retained pre-reference generated C on GCC,
Apple Clang and Homebrew Clang. The old counterfactual C necessarily emits
unused-function diagnostics; a separate object-only audit may leave that one
warning visible without promoting it to an error, solely to measure the old
undefined-symbol set. That diagnostic control requires review before execution;
it is never qualification, never executed, and never replaces the real unchanged
strict O0/O2 compiler commands. I will stop if references add a previously absent
external dependency and revise emission before qualification. The real gates
retain `-Wall -Wextra -Werror`, exact original provider/link selections, output
sentinels, allocation domains and fixed deadlines.

After source/fixture review I will rebuild the changed Nano producer and its
full bootstrap closure, then finish actual eight native methods and eighteen original methods and required
adjacency under immutable input maps. Already passed 303 focused C ownership and
storage gates keep their exact source attribution; no change to their header or
owning providers is proposed. Full source/list/enum/whole-Make/integration and
fixed-point acceptance remain open.

## My measured object boundary and revised reference form

My nonexecuted e10 object audit retained 85 identical undefined symbols on GCC,
but Apple Clang changed from 40 to 88 and introduced 48 dependencies. I stopped
there; Homebrew was not reached. The discarded function expressions caused
otherwise unused static bodies to be emitted at O0. I do not claim the original
no-call expression was link-neutral. Both object files, exact commands, visible
warnings, symbol listings and unchanged input maps remain in the home paths
`native-helper-e10-linux-objects` and `native-helper-e10-puck-objects`, with a
complete local copy of the Darwin evidence.

My revised production contract replaces each proposed `(void)helper;` with
`(void)sizeof(&helper);`. The operand has complete function-pointer object type,
not function type or variably modified type. The address expression is therefore
genuinely unevaluated; I neither invoke the helper nor materialize/store its
address. All 52 exact declarations, both real entry placements and their source
guards stay unchanged. The earlier contract/catalog above remains evidence of
the first proposal; this paragraph supersedes only its expression form. I will
retain and compare all three compiler O0 undefined-symbol sets before new strict
qualification. A compiler that still introduces dependencies or rejects the
strict new object blocks qualification; I do not relax that acceptance.

## My explicit optional declaration contract

The sizeof proposal at 608 passed GCC with 85 identical undefined symbols, but
Apple Clang rejected the strict new object with
`-Wunneeded-internal-declaration`. I retain that terminal too; Homebrew remained
unreached, and 608 is not qualified. I now state intended optional use on these
same 52 owned declarations instead of adding expressions to entry functions.

Only `src_nano/transpiler.nano` changes. At the end of `gen_c_runtime`, after all
52 existing definitions are visible, I emit a file-scope portability macro:

```c
#if defined(__GNUC__) || defined(__clang__)
#define NANO_RT_OPTIONAL_SUPPORT(name) \
    static __typeof__(name) name __attribute__((unused));
#else
#define NANO_RT_OPTIONAL_SUPPORT(name)
#endif
```

I then emit exactly one invocation for each catalog name, with no extra trailing
semicolon, and undefine the macro. Each supported-compiler redeclaration retains
static linkage and obtains the exact already declared function type. The type
operand is unevaluated, and the attribute declares intentional optional use; it
does not request emission or a call. The empty fallback retains existing source
on other compilers; I claim strict acceptance only for my selected GCC/Clang
providers. I do not disable a warning class or relax checks inside any helper.

All original helper bodies, C signatures, binary64 header/generated-source bytes
and preprocessor guards remain unchanged. The earlier entry expressions are
removed entirely, returning both actual entry functions to their original
statements. Only the newly added proposal-specific shadow checks migrate from
entry-reference strings to the declaration block; all original source/fixture
assertions remain. The lower-level runtime string assembler also receives the
same annotations through `gen_c_runtime`, with no requirement for a new entry.

The exact catalog/source audit and three-compiler object comparison remain
mandatory. I will construct a separately attributed nonexecuted counterfactual
from retained 303 generated C by inserting only this exact block after runtime
helpers, then compare old/new O0 symbols. The actual newly built producer must
still generate, strictly compile, link and run the full original eighteen and
new eight methods. Neither counterfactual is semantic qualification.

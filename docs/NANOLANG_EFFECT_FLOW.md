# My private effect-arm flow collector

I collect the shared handler-entry and arm-reachability contract from
`src/nanoisa/nvm2c_effect_flow.h` in pure NanoLang at
`src_nano/nvm2c/effect_flow.nano`. This is one dependency of #1142's callable
collector and representation analysis. I have not connected it to installed
translation or granted executable effect admission.

```nano
module "src_nano/nvm2c/effect_flow.nano" as effects

let result: effects.Collection = (effects.collect code functions names)
```

I take raw code, function ranges/local counts and operation-name bytes. I decode
every declared body with my existing ordinary function decoder, including dead
instructions and bodies without registrations. I reject malformed ranges,
instructions and branch/direct-call targets before returning a valid collection.
I return `valid: false` with no handlers on refusal; partial C allocation state
is not part of this public contract.

For each registration, in function/instruction order, I retain the function,
canonical operation index, relative arm target, first local, local count and
`code_length + 1` boolean flow map. A handler target must name an instruction
strictly inside its body. Its local interval must fit the function. Duplicate
operation names select the earliest matching string index. I match the C
reference's NUL-terminated comparison, including embedded NUL bytes.

I visit ordinary fallthrough, both conditional successors and the sole
unconditional jump successor. I stop at RET, HALT, EFFECT_RESUME and TAIL_CALL.
Nested handler registration follows its ordinary next instruction; it does not
enter its nested arm. MATCH_TAG and FILE_RESULT_BRANCH retain their reference
collector behavior: decode validates their target, while arm flow follows the
next instruction. I can mark the terminal boundary but never treat it as an
instruction. Loops terminate after each byte boundary is visited once.

My [Linux receipt](evidence/darwin-current-source-20261124/diagnostics/effect-flow-five-linux-20261126/README.md)
compares complete ordered records/maps against the actual unchanged C header,
including its ordinary decoder. It checks ordinary/sanitized C and five routes,
818 matrix packets, imported shadows, malformed metadata/decode, every primary
opcode, duplicates, nested arms and 128 generated labelled control-flow graphs.
The generated graphs derive their expected maps before byte encoding. I retain
the first fixture-oracle failure and the separate storage-admission refusal.

Run the permanent gate with an explicitly qualified NanoLang compiler to include
all five routes:

```sh
NANOLANG_TRANSLATOR_SOURCE_COMPILER=/absolute/path/to/nanoc_stage2 \
  make test-nanolang-translator-effect-flow
```

Without that option the ordinary test helper selects its three C-produced
routes. My recorded five-route receipt binds the actual compiler and its six
host libraries; a filename alone does not establish that provenance.

The complete callable collector/linker, variant projection provenance, finite
indirect dispatch, representation/body/runtime emission, module/profile/ownership
composition, paired platform qualification and installed cutover remain open.
This helper does not repair #1262 or satisfy all69 release requirements.

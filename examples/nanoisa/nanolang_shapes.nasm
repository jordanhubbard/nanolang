; nanolang_shapes.nasm — what NanoLang's constructs look like in NanoISA
;
; Every operand here is symbolic: functions, string constants and branch
; targets are named rather than numbered. The assembler resolves the names and
; the disassembler reconstructs them, so this file and the bytecode it produces
; are two views of the same thing rather than two things kept in sync by hand.
;
; Assemble and run:
;   nanoisa asm examples/nanoisa/nanolang_shapes.nasm -o shapes.nvm
;   nano_vm shapes.nvm

; ── Constants ───────────────────────────────────────────────────────────────
; A named .string binds a symbol to its pool index, so PUSH_STR takes the name.
.string s_banner "NanoISA symbolic assembly"
.string s_fizz   "fizz"
.string s_plain  "plain"

; ── Symbols ─────────────────────────────────────────────────────────────────
; .symbol binds a name to an index in a table the module already has. Here it
; names the functions defined below so CALL can refer to them by name.
.symbol function fn_classify 1
.symbol function fn_sum_to   2

.entry 0

; ── main ────────────────────────────────────────────────────────────────────
; Returns 0. Prints a banner, classifies a number, and sums 1..5.
.function main 0 1 0 int 1
  PUSH_STR s_banner
  PRINTLN

  ; classify(15) — the result is a string, printed and discarded
  PUSH_I64 15
  CALL fn_classify
  PRINTLN

  ; sum_to(5) — discard the result; this function returns 0 unconditionally
  PUSH_I64 5
  CALL fn_sum_to
  POP

  PUSH_I64 0
  RET
.end

; ── classify(n) -> string ───────────────────────────────────────────────────
; `if n % 3 == 0 then "fizz" else "plain"`.
;
; Both arms leave exactly one value, which is what makes the join verifiable:
; the verifier requires every path reaching a label to agree on stack height.
.function classify 1 1 0 string 1
  LOAD_LOCAL 0
  PUSH_I64 3
  I64_REM_S
  PUSH_I64 0
  I64_EQ
  JMP_FALSE not_divisible
  PUSH_STR s_fizz
  JMP done
not_divisible:
  PUSH_STR s_plain
done:
  RET
.end

; ── sum_to(n) -> int ────────────────────────────────────────────────────────
; A counted loop written the way codegen writes one: the accumulator and the
; index live in locals, and the operand stack is empty at the top of every
; iteration. Keeping the loop's join points at a constant height is not a
; style preference -- a height that differs between the fall-in edge and the
; back edge is rejected.
.function sum_to 1 3 0 int 1
  PUSH_I64 0
  STORE_LOCAL 1          ; acc = 0
  PUSH_I64 1
  STORE_LOCAL 2          ; i = 1
loop:
  LOAD_LOCAL 2
  LOAD_LOCAL 0
  I64_GT_S               ; i > n ?
  JMP_TRUE loop_end
  LOAD_LOCAL 1
  LOAD_LOCAL 2
  I64_ADD
  STORE_LOCAL 1          ; acc += i
  LOAD_LOCAL 2
  PUSH_I64 1
  I64_ADD
  STORE_LOCAL 2          ; i += 1
  JMP loop
loop_end:
  LOAD_LOCAL 1
  RET
.end

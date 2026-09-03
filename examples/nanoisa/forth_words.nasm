; forth_words.nasm — Forth words as NanoISA, using the stack directly
;
; NanoLang keeps values in locals and uses the operand stack for expression
; temporaries. Forth does the opposite: the stack IS the calling convention,
; and words are defined by what they do to it. NanoISA supports both, which is
; the point of this file next to nanolang_shapes.nasm.
;
; The stack-shuffling primitives Forth needs -- DUP, SWAP, ROT3, PICK, ROLL --
; are ISA instructions rather than library calls, because a runtime library
; cannot reach into the operand stack to implement them.
;
; Assemble and run:
;   nanoisa asm examples/nanoisa/forth_words.nasm -o forth.nvm
;   nano_vm forth.nvm

.string s_label "SQUARE 7, CUBE 3, and 10 gcd 4:"

.symbol function fn_square 1
.symbol function fn_cube   2
.symbol function fn_gcd    3

.entry 0

.function main 0 1 0 int 1
  PUSH_STR s_label
  PRINTLN

  PUSH_I64 7
  CALL fn_square         ; ( 7 -- 49 )
  PRINTLN

  PUSH_I64 3
  CALL fn_cube           ; ( 3 -- 27 )
  PRINTLN

  PUSH_I64 10
  PUSH_I64 4
  CALL fn_gcd            ; ( 10 4 -- 2 )
  PRINTLN

  PUSH_I64 0
  RET
.end

; : SQUARE ( n -- n^2 )  DUP * ;
;
; A direct transcription. DUP is (1 -> 2): it requires a value to duplicate,
; which is why the verifier charges it a pop as well as two pushes -- the
; requirement and the net effect are different numbers, and only tracking both
; catches a DUP on an empty stack.
.function square 1 1 0 int 1
  LOAD_LOCAL 0           ; the argument arrives as a local; put it on the stack
  DUP                    ; ( n -- n n )
  I64_MUL                ; ( n n -- n^2 )
  RET
.end

; : CUBE ( n -- n^3 )  DUP DUP * * ;
.function cube 1 1 0 int 1
  LOAD_LOCAL 0
  DUP
  DUP                    ; ( n -- n n n )
  I64_MUL                ; ( n n n -- n n^2 )
  I64_MUL                ; ( n n^2 -- n^3 )
  RET
.end

; : GCD ( a b -- gcd )  BEGIN DUP WHILE SWAP OVER MOD REPEAT DROP ;
;
; Euclid's algorithm with the operands kept on the stack rather than in
; locals, which is what a Forth front end emits. The loop's join is at a
; constant depth of two, and the verifier requires that: an iteration that
; left the stack one deeper than it found it would be rejected here rather
; than overflowing at run time.
.function gcd 2 2 0 int 1
  LOAD_LOCAL 0
  LOAD_LOCAL 1           ; ( -- a b )
loop:
  DUP                    ; ( a b -- a b b )
  PUSH_I64 0
  I64_EQ                 ; ( a b b -- a b flag )
  JMP_TRUE done
  ; ( a b ) -> ( b  a mod b )
  SWAP                   ; ( a b -- b a )
  PICK 1                 ; copy b over the top: ( b a -- b a b )
  I64_REM_S              ; ( b a b -- b  a mod b )
  JMP loop
done:
  POP                    ; drop the zero: ( a 0 -- a )
  RET
.end

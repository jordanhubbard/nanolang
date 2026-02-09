(** * NanoCore: OCaml Extraction

    Extracts the computable evaluator [eval_program] from EvalFn.v to OCaml.
    The generated code can be used as a reference interpreter for testing
    the C implementation.

    Usage:
      cd formal && make extract
      # Generates extracted/EvalFn.ml and extracted/EvalFn.mli
*)

From NanoCore Require Import Syntax.
From NanoCore Require Import Semantics.
From NanoCore Require Import EvalFn.

(** Re-state extraction directives (needed in this compilation unit) *)
From Stdlib Require Extraction.
From Stdlib Require ExtrOcamlBasic.
From Stdlib Require ExtrOcamlString.
From Stdlib Require ExtrOcamlZBigInt.

Extraction Language OCaml.

Extract Inductive bool => "bool" [ "true" "false" ].
Extract Inductive nat => "int" [ "0" "succ" ]
  "(fun fO fS n -> if n = 0 then fO () else fS (n - 1))".
Extract Inductive list => "list" [ "[]" "(::)" ].
Extract Inductive prod => "( * )" [ "(,)" ].
Extract Inductive option => "option" [ "Some" "None" ].

(** Extract [eval_program] and all its dependencies *)
Separate Extraction eval_program.

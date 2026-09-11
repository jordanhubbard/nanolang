(** I check reference-evaluator regressions by reduction in Rocq. These examples
    complement, rather than replace, the general soundness obligation. *)
From Stdlib Require Import String ZArith Bool.
From NanoCore Require Import Syntax EvalFn.
Open Scope string_scope.
Open Scope Z_scope.

Definition logical_test_env := ECons "x" (VInt 0) ENil.
Definition logical_effect (b : bool) :=
  ESeq (ESet "x" (EInt 1)) (EBool b).

Example logical_truth_tables : forall a b,
  eval_fn 3 ENil (EBinOp OpAnd (EBool a) (EBool b)) =
    Some (ENil, VBool (andb a b)) /\
  eval_fn 3 ENil (EBinOp OpOr (EBool a) (EBool b)) =
    Some (ENil, VBool (orb a b)).
Proof. intros a b; destruct a, b; split; reflexivity. Qed.

Example logical_skips_stuck_rhs :
  eval_fn 3 ENil (EBinOp OpAnd (EBool false) (EVar "missing")) =
    Some (ENil, VBool false) /\
  eval_fn 3 ENil (EBinOp OpOr (EBool true) (EVar "missing")) =
    Some (ENil, VBool true).
Proof. split; reflexivity. Qed.

Example logical_requires_needed_rhs :
  eval_fn 3 ENil (EBinOp OpAnd (EBool true) (EVar "missing")) = None /\
  eval_fn 3 ENil (EBinOp OpOr (EBool false) (EVar "missing")) = None.
Proof. split; reflexivity. Qed.

Example logical_skips_effects :
  eval_fn 5 logical_test_env (EBinOp OpAnd (EBool false) (logical_effect true)) =
    Some (logical_test_env, VBool false) /\
  eval_fn 5 logical_test_env (EBinOp OpOr (EBool true) (logical_effect false)) =
    Some (logical_test_env, VBool true).
Proof. split; reflexivity. Qed.

Example logical_runs_needed_effects :
  eval_fn 5 logical_test_env (EBinOp OpAnd (EBool true) (logical_effect false)) =
    Some (ECons "x" (VInt 1) ENil, VBool false) /\
  eval_fn 5 logical_test_env (EBinOp OpOr (EBool false) (logical_effect true)) =
    Some (ECons "x" (VInt 1) ENil, VBool true).
Proof. split; reflexivity. Qed.

Example logical_preserves_left_effects :
  eval_fn 5 logical_test_env (EBinOp OpAnd (logical_effect false) (EVar "missing")) =
    Some (ECons "x" (VInt 1) ENil, VBool false) /\
  eval_fn 5 logical_test_env (EBinOp OpOr (logical_effect true) (EVar "missing")) =
    Some (ECons "x" (VInt 1) ENil, VBool true).
Proof. split; reflexivity. Qed.

Example logical_rejects_needed_nonbooleans :
  eval_fn 3 ENil (EBinOp OpAnd (EBool true) (EInt 0)) = None /\
  eval_fn 3 ENil (EBinOp OpOr (EBool false) (EInt 0)) = None /\
  eval_fn 3 ENil (EBinOp OpAnd (EInt 0) (EBool false)) = None /\
  eval_fn 3 ENil (EBinOp OpOr (EInt 0) (EBool true)) = None.
Proof. repeat split; reflexivity. Qed.

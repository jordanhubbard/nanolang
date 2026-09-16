(** I check reference-evaluator regressions by reduction in Rocq. These examples
    complement, rather than replace, the general soundness theorem. *)
From Stdlib Require Import String ZArith Bool List.
From NanoCore Require Import Syntax EvalFn.
Import ListNotations.
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

Example arithmetic_zero_divisors_fail :
  eval_fn 3 ENil (EBinOp OpDiv (EInt 7) (EInt 0)) = None /\
  eval_fn 3 ENil (EBinOp OpMod (EInt 7) (EInt 0)) = None.
Proof. split; reflexivity. Qed.

Example let_restores_shadowed_binding :
  eval_fn 6 logical_test_env
    (ELet "x" (EInt 2) (ESeq (ESet "x" (EInt 3)) (EVar "x"))) =
    Some (logical_test_env, VInt 3).
Proof. reflexivity. Qed.

Example let_preserves_outer_mutation :
  eval_fn 6 logical_test_env (ELet "y" (EInt 2) (ESet "x" (EVar "y"))) =
    Some (ECons "x" (VInt 2) ENil, VUnit).
Proof. reflexivity. Qed.

Example closure_body_does_not_mutate_caller :
  eval_fn 6 logical_test_env
    (EApp (ELam "y" TInt (ESeq (ESet "x" (EInt 9)) (EVar "y"))) (EInt 5)) =
    Some (logical_test_env, VInt 5).
Proof. reflexivity. Qed.

Example recursive_closure_countdown :
  eval_fn 12 ENil
    (EApp (EFix "f" "n" TInt TInt
      (EIf (EBinOp OpEq (EVar "n") (EInt 0)) (EInt 0)
        (EApp (EVar "f") (EBinOp OpSub (EVar "n") (EInt 1))))) (EInt 2)) =
    Some (ENil, VInt 0).
Proof. reflexivity. Qed.

Example while_threads_mutation :
  eval_fn 10 logical_test_env
    (EWhile (EBinOp OpLt (EVar "x") (EInt 2))
      (ESet "x" (EBinOp OpAdd (EVar "x") (EInt 1)))) =
    Some (ECons "x" (VInt 2) ENil, VUnit).
Proof. reflexivity. Qed.

Example match_restores_shadowed_binding :
  eval_fn 5 logical_test_env
    (EMatch (EConstruct "Some" (EInt 3) (TVariant [("Some", TInt)]))
      [("Some", "x", EVar "x")]) = Some (logical_test_env, VInt 3).
Proof. reflexivity. Qed.

Example empty_aggregates :
  eval_fn 1 ENil (EArray []) = Some (ENil, VArray []) /\
  eval_fn 1 ENil (ERecord []) = Some (ENil, VRecord []) /\
  eval_fn 1 ENil (ETuple []) = Some (ENil, VTuple []).
Proof. repeat split; reflexivity. Qed.

Example array_literal_evaluation_order :
  eval_fn 7 logical_test_env (EArray [logical_effect true; EVar "x"]) =
    Some (ECons "x" (VInt 1) ENil, VArray [VBool true; VInt 1]).
Proof. reflexivity. Qed.

Example tuple_literal_evaluation_order :
  eval_fn 7 logical_test_env (ETuple [logical_effect true; EVar "x"]) =
    Some (ECons "x" (VInt 1) ENil, VTuple [VBool true; VInt 1]).
Proof. reflexivity. Qed.

Example record_literal_evaluation_order :
  eval_fn 7 logical_test_env
    (ERecord [("first", logical_effect true); ("second", EVar "x")]) =
    Some (ECons "x" (VInt 1) ENil,
      VRecord [("first", VBool true); ("second", VInt 1)]).
Proof. reflexivity. Qed.

Example aggregate_indexing :
  eval_fn 5 ENil (EIndex (EArray [EInt 3]) (EInt 0)) = Some (ENil, VInt 3) /\
  eval_fn 5 ENil (ETupleIndex (ETuple [EBool true; EInt 3]) 1) = Some (ENil, VInt 3).
Proof. split; reflexivity. Qed.

Example aggregate_index_bounds :
  eval_fn 5 ENil (EIndex (EArray [EInt 3]) (EInt 1)) = None /\
  eval_fn 5 ENil (ETupleIndex (ETuple [EInt 3]) 1) = None.
Proof. split; reflexivity. Qed.

Example array_functional_updates :
  eval_fn 5 ENil (EArraySet (EArray [EInt 3]) (EInt 0) (EInt 8)) =
    Some (ENil, VArray [VInt 8]) /\
  eval_fn 5 ENil (EArrayPush (EArray [EInt 3]) (EInt 8)) =
    Some (ENil, VArray [VInt 3; VInt 8]).
Proof. split; reflexivity. Qed.

Example record_update_and_read :
  eval_fn 6 (ECons "r" (VRecord [("field", VInt 0)]) ENil)
    (ESeq (ESetField "r" "field" (EInt 8)) (EField (EVar "r") "field")) =
    Some (ECons "r" (VRecord [("field", VInt 8)]) ENil, VInt 8).
Proof. reflexivity. Qed.

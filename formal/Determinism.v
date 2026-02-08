(** * NanoCore: Determinism of Evaluation

    If a well-typed expression evaluates to two different results,
    they must be the same. This proves the evaluation relation is
    a partial function. *)

From Stdlib Require Import ZArith.
From Stdlib Require Import Bool.
From Stdlib Require Import String.
From Stdlib Require Import List.
Import ListNotations.
From NanoCore Require Import Syntax.
From NanoCore Require Import Semantics.
Open Scope Z_scope.
Open Scope string_scope.

(** ** Helper tactics *)

(** Apply one determinism IH to a matching eval hypothesis.
    Either solves the goal (via discriminate on value mismatch)
    or establishes equalities and substitutes. *)
Ltac det_step :=
  match goal with
  | [ IH : forall _ _, eval ?env ?e _ _ -> _ = _ /\ _ = _,
      H : eval ?env ?e _ _ |- _ ] =>
    let Hv := fresh "Hv" in let He := fresh "He" in
    destruct (IH _ _ H) as [Hv He]; clear IH; clear H;
    try discriminate Hv;
    try discriminate He;
    try (injection Hv; intros; subst; clear Hv);
    try (injection He; intros; subst; clear He);
    try subst
  end.

(** Apply all available IHs *)
Ltac det_IHs := repeat det_step.

(** Resolve identical function applications producing Some *)
Ltac det_func :=
  match goal with
  | [ H1 : ?f = Some ?a, H2 : ?f = Some ?b |- _ ] =>
    rewrite H1 in H2; injection H2; intros; subst; clear H2
  end.

(** Contradiction: arith and cmp binops are disjoint *)
Ltac det_arith_cmp_contra :=
  match goal with
  | [ H1 : eval_arith_binop ?op _ _ = Some _,
      H2 : eval_cmp_binop ?op _ _ = Some _ |- _ ] =>
    destruct op; simpl in *; discriminate
  end.

(** Combined solver *)
Ltac det_solve :=
  det_IHs;
  try det_func;
  try det_arith_cmp_contra;
  try (split; reflexivity);
  try (split; congruence).

(** ** Determinism Theorem *)

Theorem eval_deterministic : forall renv e renv' v,
  eval renv e renv' v ->
  forall renv'' v',
  eval renv e renv'' v' ->
  v = v' /\ renv' = renv''.
Proof.
  intros renv e renv' v Heval.
  induction Heval; intros renv'' v' H2;
    inversion H2; subst; clear H2; det_solve.
Qed.

(** * NanoCore: Type Soundness

    This file proves type soundness for NanoCore via preservation.

    - **Preservation**: If a well-typed expression evaluates to a value,
      that value has the expected type.

    This guarantees: "well-typed programs produce well-typed results."
*)

From Stdlib Require Import ZArith.
From Stdlib Require Import Bool.
From Stdlib Require Import String.
From NanoCore Require Import Syntax.
From NanoCore Require Import Semantics.
From NanoCore Require Import Typing.
Open Scope string_scope.

(** ** Value typing

    Defines when a value has a given type. *)

Inductive val_has_type : val -> ty -> Prop :=
  | VT_Int : forall n, val_has_type (VInt n) TInt
  | VT_Bool : forall b, val_has_type (VBool b) TBool
  | VT_Clos : forall x body clos_env t1 t2 c,
      env_ctx_agree clos_env c ->
      has_type (CtxCons x t1 c) body t2 ->
      val_has_type (VClos x body clos_env) (TArrow t1 t2)

(** Agreement between runtime environments and typing contexts *)

with env_ctx_agree : env -> ctx -> Prop :=
  | ECA_Nil : env_ctx_agree ENil CtxNil
  | ECA_Cons : forall x v e t c,
      val_has_type v t ->
      env_ctx_agree e c ->
      env_ctx_agree (ECons x v e) (CtxCons x t c).

(** ** Scheme for mutual induction *)

Scheme val_has_type_ind2 := Induction for val_has_type Sort Prop
  with env_ctx_agree_ind2 := Induction for env_ctx_agree Sort Prop.

(** ** Canonical Forms Lemmas *)

Lemma canonical_int : forall v,
  val_has_type v TInt -> exists n, v = VInt n.
Proof.
  intros v H. inversion H. exists n. reflexivity.
Qed.

Lemma canonical_bool : forall v,
  val_has_type v TBool -> exists b, v = VBool b.
Proof.
  intros v H. inversion H. exists b. reflexivity.
Qed.

Lemma canonical_arrow : forall v t1 t2,
  val_has_type v (TArrow t1 t2) ->
  exists x body clos_env, v = VClos x body clos_env.
Proof.
  intros v t1 t2 H. inversion H. subst.
  exists x, body, clos_env. reflexivity.
Qed.

(** ** Environment lookup agrees with context lookup *)

Lemma agree_lookup : forall e c x t,
  env_ctx_agree e c ->
  ctx_lookup x c = Some t ->
  exists v, env_lookup x e = Some v /\ val_has_type v t.
Proof.
  intros e c x t Hagree.
  induction Hagree; intros Hlookup.
  - simpl in Hlookup. discriminate.
  - simpl in Hlookup. simpl.
    destruct (String.eqb x x0) eqn:Heq.
    + inversion Hlookup; subst. exists v. split; [reflexivity | assumption].
    + apply IHHagree. assumption.
Qed.

(** ** Agreement preserved by extension *)

Lemma agree_cons : forall e c x v t,
  env_ctx_agree e c ->
  val_has_type v t ->
  env_ctx_agree (ECons x v e) (CtxCons x t c).
Proof.
  intros. constructor; assumption.
Qed.

(** ** Helper lemmas for operator typing *)

Lemma eval_arith_binop_type : forall op n1 n2 v,
  eval_arith_binop op n1 n2 = Some v ->
  val_has_type v (binop_res_type op).
Proof.
  intros op n1 n2 v Heval.
  destruct op; simpl in *; try discriminate.
  - inversion Heval; subst. constructor.
  - inversion Heval; subst. constructor.
  - inversion Heval; subst. constructor.
  - destruct (Z.eqb n2 0); [discriminate | inversion Heval; subst; constructor].
  - destruct (Z.eqb n2 0); [discriminate | inversion Heval; subst; constructor].
Qed.

Lemma eval_cmp_binop_type : forall op n1 n2 v,
  eval_cmp_binop op n1 n2 = Some v ->
  val_has_type v (binop_res_type op).
Proof.
  intros op n1 n2 v Heval.
  destruct op; simpl in *; try discriminate;
    inversion Heval; subst; constructor.
Qed.

(** ** Impossible evaluation lemmas

    When we know a value has a certain type but an eval rule
    requires a different constructor, we derive a contradiction. *)

(** Inversion lemma for closure typing *)
Lemma clos_type_inv : forall x body cenv t1 t2,
  val_has_type (VClos x body cenv) (TArrow t1 t2) ->
  exists c,
    env_ctx_agree cenv c /\
    has_type (CtxCons x t1 c) body t2.
Proof.
  intros. inversion H; subst.
  exists c. split; assumption.
Qed.

(** ** Preservation Theorem

    We prove preservation by induction on the evaluation derivation.
    This is essential because the App case evaluates the closure body,
    which is not a syntactic subexpression of the application. Inducting
    on the evaluation gives us IH for all sub-evaluations, including
    the body evaluation inside the closure. *)

Theorem preservation : forall renv e result,
  eval renv e result ->
  forall gamma t,
  has_type gamma e t ->
  env_ctx_agree renv gamma ->
  val_has_type result t.
Proof.
  intros renv e result Heval.
  induction Heval; intros gamma ty0 Htype Hagree.

  - (* E_Int *)
    inversion Htype; subst. constructor.

  - (* E_Bool *)
    inversion Htype; subst. constructor.

  - (* E_Var *)
    inversion Htype; subst.
    pose proof (agree_lookup _ _ _ _ Hagree H2) as [v' [Hl Hvt]].
    rewrite H in Hl. inversion Hl; subst. assumption.

  - (* E_BinArith *)
    inversion Htype; subst.
    + eapply eval_arith_binop_type; eassumption.
    + (* logic op - impossible *)
      exfalso.
      assert (val_has_type (VInt n1) TBool) by (eapply IHHeval1; eassumption).
      match goal with [ H : val_has_type (VInt _) TBool |- _ ] => inversion H end.
    + (* eq bool - impossible *)
      exfalso.
      assert (val_has_type (VInt n1) TBool) by (eapply IHHeval1; eassumption).
      match goal with [ H : val_has_type (VInt _) TBool |- _ ] => inversion H end.

  - (* E_BinCmp *)
    inversion Htype; subst.
    + eapply eval_cmp_binop_type; eassumption.
    + exfalso.
      assert (val_has_type (VInt n1) TBool) by (eapply IHHeval1; eassumption).
      match goal with [ H : val_has_type (VInt _) TBool |- _ ] => inversion H end.
    + exfalso.
      assert (val_has_type (VInt n1) TBool) by (eapply IHHeval1; eassumption).
      match goal with [ H : val_has_type (VInt _) TBool |- _ ] => inversion H end.

  - (* E_BinEqBool *)
    inversion Htype; subst.
    + exfalso.
      assert (val_has_type (VBool b1) TInt) by (eapply IHHeval1; eassumption).
      match goal with [ H : val_has_type (VBool _) TInt |- _ ] => inversion H end.
    + exfalso.
      match goal with [ H : binop_arg_type _ = TBool |- _ ] => simpl in H; discriminate end.
    + constructor.

  - (* E_BinNeBool *)
    inversion Htype; subst.
    + exfalso.
      assert (val_has_type (VBool b1) TInt) by (eapply IHHeval1; eassumption).
      match goal with [ H : val_has_type (VBool _) TInt |- _ ] => inversion H end.
    + exfalso.
      match goal with [ H : binop_arg_type _ = TBool |- _ ] => simpl in H; discriminate end.
    + constructor.

  - (* E_And_True *)
    inversion Htype; subst.
    + exfalso.
      assert (val_has_type (VBool true) TInt) by (eapply IHHeval1; eassumption).
      match goal with [ H : val_has_type (VBool _) TInt |- _ ] => inversion H end.
    + eapply IHHeval2; eassumption.
    + exfalso.
      match goal with [ H : binop_allows_bool_args _ = true |- _ ] => simpl in H; discriminate end.

  - (* E_And_False - dead rule, subsumed by E_And_Short *)
    inversion Htype; subst.
    + exfalso.
      assert (val_has_type (VBool false) TInt) by (eapply IHHeval; eassumption).
      match goal with [ H : val_has_type (VBool _) TInt |- _ ] => inversion H end.
    + constructor.
    + exfalso.
      match goal with [ H : binop_allows_bool_args _ = true |- _ ] => simpl in H; discriminate end.

  - (* E_And_Short *)
    inversion Htype; subst.
    + exfalso.
      assert (val_has_type (VBool false) TInt) by (eapply IHHeval; eassumption).
      match goal with [ H : val_has_type (VBool _) TInt |- _ ] => inversion H end.
    + constructor.
    + exfalso.
      match goal with [ H : binop_allows_bool_args _ = true |- _ ] => simpl in H; discriminate end.

  - (* E_Or_False *)
    inversion Htype; subst.
    + exfalso.
      assert (val_has_type (VBool false) TInt) by (eapply IHHeval1; eassumption).
      match goal with [ H : val_has_type (VBool _) TInt |- _ ] => inversion H end.
    + eapply IHHeval2; eassumption.
    + exfalso.
      match goal with [ H : binop_allows_bool_args _ = true |- _ ] => simpl in H; discriminate end.

  - (* E_Or_Short *)
    inversion Htype; subst.
    + exfalso.
      assert (val_has_type (VBool true) TInt) by (eapply IHHeval; eassumption).
      match goal with [ H : val_has_type (VBool _) TInt |- _ ] => inversion H end.
    + constructor.
    + exfalso.
      match goal with [ H : binop_allows_bool_args _ = true |- _ ] => simpl in H; discriminate end.

  - (* E_Neg *)
    inversion Htype; subst. constructor.

  - (* E_Not *)
    inversion Htype; subst. constructor.

  - (* E_IfTrue *)
    inversion Htype; subst.
    eapply IHHeval2; eassumption.

  - (* E_IfFalse *)
    inversion Htype; subst.
    eapply IHHeval2; eassumption.

  - (* E_Let *)
    inversion Htype; subst.
    eapply IHHeval2.
    + eassumption.
    + apply agree_cons; [assumption | eapply IHHeval1; eassumption].

  - (* E_Lam *)
    inversion Htype; subst.
    econstructor; eassumption.

  - (* E_App *)
    inversion Htype; subst.
    match goal with
    | [ H1 : has_type _ e1 (TArrow ?ta ?tr),
        H2 : has_type _ e2 ?ta |- val_has_type _ ?tr ] =>
      assert (Hf : val_has_type (VClos x body clos_env) (TArrow ta tr))
        by (eapply IHHeval1; eassumption);
      assert (Harg : val_has_type v2 ta)
        by (eapply IHHeval2; eassumption);
      apply clos_type_inv in Hf;
      destruct Hf as [c' [Hcenv Hbody_type]];
      eapply IHHeval3;
      [ exact Hbody_type | apply agree_cons; assumption ]
    end.
Qed.

(** ** Type soundness corollary *)

Corollary soundness : forall e t v,
  has_type CtxNil e t ->
  eval ENil e v ->
  val_has_type v t.
Proof.
  intros.
  eapply preservation; try eassumption.
  constructor.
Qed.

(** ** Progress (stated as conjecture)

    For NanoCore without loops or mutation, all well-typed closed expressions
    terminate and produce a value. This follows from strong normalization of
    the simply-typed lambda calculus with base types. *)

Conjecture progress : forall e t,
  has_type CtxNil e t ->
  exists v, eval ENil e v.

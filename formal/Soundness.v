(** * NanoCore: Type Soundness

    This file proves type soundness for NanoCore via preservation.

    - **Preservation**: If a well-typed expression evaluates to a value,
      that value has the expected type, and the environment remains
      well-typed (env_ctx_agree preserved through mutation).

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
  | VT_Unit : val_has_type VUnit TUnit
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

Lemma canonical_unit : forall v,
  val_has_type v TUnit -> v = VUnit.
Proof.
  intros v H. inversion H. reflexivity.
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

(** ** Agreement preserved by update

    If the environment and context agree, and we update a variable
    with a value of the correct type, agreement is preserved. *)

Lemma agree_update : forall e c x t v,
  env_ctx_agree e c ->
  ctx_lookup x c = Some t ->
  val_has_type v t ->
  env_ctx_agree (env_update x v e) c.
Proof.
  intros e c x t v Hagree.
  induction Hagree; intros Hlookup Hvt.
  - simpl in Hlookup. discriminate.
  - simpl. simpl in Hlookup.
    destruct (String.eqb x x0) eqn:Heq.
    + inversion Hlookup; subst. constructor; [assumption | assumption].
    + constructor; [assumption | apply IHHagree; assumption].
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

(** ** Inversion lemma for closure typing *)

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
    The theorem shows both:
    1. The result value has the expected type
    2. The output environment agrees with the typing context

    Property (2) is essential for threading agreement through
    sequential composition, while loops, and binary operators
    whose subexpressions may contain mutations.

    E_Let pops the let binding from the output environment,
    and E_App uses the caller's environment (lexical scoping),
    so both preserve the agreement with the original context. *)

Theorem preservation : forall renv e renv' result,
  eval renv e renv' result ->
  forall gamma t,
  has_type gamma e t ->
  env_ctx_agree renv gamma ->
  val_has_type result t /\ env_ctx_agree renv' gamma.
Proof.
  intros renv e renv' result Heval.
  induction Heval; intros gamma ty0 Htype Hagree.

  - (* E_Int *)
    inversion Htype; subst. split; [constructor | assumption].

  - (* E_Bool *)
    inversion Htype; subst. split; [constructor | assumption].

  - (* E_Unit *)
    inversion Htype; subst. split; [constructor | assumption].

  - (* E_Var *)
    inversion Htype; subst.
    pose proof (agree_lookup _ _ _ _ Hagree H2) as [v' [Hl Hvt]].
    rewrite H in Hl. inversion Hl; subst.
    split; [assumption | assumption].

  - (* E_BinArith *)
    inversion Htype; subst.
    + (* T_BinOp *)
      match goal with
      | [ He1 : has_type _ e1 TInt, He2 : has_type _ e2 TInt |- _ ] =>
        destruct (IHHeval1 _ _ He1 Hagree) as [Hvt1 Hagree1];
        destruct (IHHeval2 _ _ He2 Hagree1) as [Hvt2 Hagree2];
        split; [eapply eval_arith_binop_type; eassumption | assumption]
      end.
    + (* T_BinLogic - impossible: e1 typed as Bool but evals to Int *)
      exfalso.
      match goal with
      | [ Ht : has_type _ e1 TBool |- _ ] =>
        destruct (IHHeval1 _ _ Ht Hagree) as [Hvt1 _];
        inversion Hvt1
      end.
    + (* T_BinEqBool - impossible: e1 typed as Bool but evals to Int *)
      exfalso.
      match goal with
      | [ Ht : has_type _ e1 TBool |- _ ] =>
        destruct (IHHeval1 _ _ Ht Hagree) as [Hvt1 _];
        inversion Hvt1
      end.

  - (* E_BinCmp *)
    inversion Htype; subst.
    + match goal with
      | [ He1 : has_type _ e1 TInt, He2 : has_type _ e2 TInt |- _ ] =>
        destruct (IHHeval1 _ _ He1 Hagree) as [Hvt1 Hagree1];
        destruct (IHHeval2 _ _ He2 Hagree1) as [Hvt2 Hagree2];
        split; [eapply eval_cmp_binop_type; eassumption | assumption]
      end.
    + exfalso.
      match goal with
      | [ Ht : has_type _ e1 TBool |- _ ] =>
        destruct (IHHeval1 _ _ Ht Hagree) as [Hvt1 _]; inversion Hvt1
      end.
    + exfalso.
      match goal with
      | [ Ht : has_type _ e1 TBool |- _ ] =>
        destruct (IHHeval1 _ _ Ht Hagree) as [Hvt1 _]; inversion Hvt1
      end.

  - (* E_BinEqBool *)
    inversion Htype; subst.
    + exfalso.
      match goal with
      | [ Ht : has_type _ e1 TInt |- _ ] =>
        destruct (IHHeval1 _ _ Ht Hagree) as [Hvt1 _]; inversion Hvt1
      end.
    + exfalso.
      match goal with [ H : binop_arg_type _ = TBool |- _ ] => simpl in H; discriminate end.
    + match goal with
      | [ He1 : has_type _ e1 TBool, He2 : has_type _ e2 TBool |- _ ] =>
        destruct (IHHeval1 _ _ He1 Hagree) as [_ Hagree1];
        destruct (IHHeval2 _ _ He2 Hagree1) as [_ Hagree2];
        split; [constructor | assumption]
      end.

  - (* E_BinNeBool *)
    inversion Htype; subst.
    + exfalso.
      match goal with
      | [ Ht : has_type _ e1 TInt |- _ ] =>
        destruct (IHHeval1 _ _ Ht Hagree) as [Hvt1 _]; inversion Hvt1
      end.
    + exfalso.
      match goal with [ H : binop_arg_type _ = TBool |- _ ] => simpl in H; discriminate end.
    + match goal with
      | [ He1 : has_type _ e1 TBool, He2 : has_type _ e2 TBool |- _ ] =>
        destruct (IHHeval1 _ _ He1 Hagree) as [_ Hagree1];
        destruct (IHHeval2 _ _ He2 Hagree1) as [_ Hagree2];
        split; [constructor | assumption]
      end.

  - (* E_And_True *)
    inversion Htype; subst.
    + exfalso.
      match goal with
      | [ Ht : has_type _ e1 TInt |- _ ] =>
        destruct (IHHeval1 _ _ Ht Hagree) as [Hvt1 _]; inversion Hvt1
      end.
    + match goal with
      | [ He1 : has_type _ e1 TBool, He2 : has_type _ e2 TBool |- _ ] =>
        destruct (IHHeval1 _ _ He1 Hagree) as [_ Hagree1];
        destruct (IHHeval2 _ _ He2 Hagree1) as [Hvt2 Hagree2];
        split; [assumption | assumption]
      end.
    + exfalso.
      match goal with [ H : binop_allows_bool_args _ = true |- _ ] => simpl in H; discriminate end.

  - (* E_And_False *)
    inversion Htype; subst.
    + exfalso.
      match goal with
      | [ Ht : has_type _ e1 TInt |- _ ] =>
        destruct (IHHeval _ _ Ht Hagree) as [Hvt1 _]; inversion Hvt1
      end.
    + destruct (IHHeval _ _ H5 Hagree) as [_ Hagree1].
      split; [constructor | assumption].
    + exfalso.
      match goal with [ H : binop_allows_bool_args _ = true |- _ ] => simpl in H; discriminate end.

  - (* E_And_Short *)
    inversion Htype; subst.
    + exfalso.
      match goal with
      | [ Ht : has_type _ e1 TInt |- _ ] =>
        destruct (IHHeval _ _ Ht Hagree) as [Hvt1 _]; inversion Hvt1
      end.
    + destruct (IHHeval _ _ H5 Hagree) as [_ Hagree1].
      split; [constructor | assumption].
    + exfalso.
      match goal with [ H : binop_allows_bool_args _ = true |- _ ] => simpl in H; discriminate end.

  - (* E_Or_False *)
    inversion Htype; subst.
    + exfalso.
      match goal with
      | [ Ht : has_type _ e1 TInt |- _ ] =>
        destruct (IHHeval1 _ _ Ht Hagree) as [Hvt1 _]; inversion Hvt1
      end.
    + match goal with
      | [ He1 : has_type _ e1 TBool, He2 : has_type _ e2 TBool |- _ ] =>
        destruct (IHHeval1 _ _ He1 Hagree) as [_ Hagree1];
        destruct (IHHeval2 _ _ He2 Hagree1) as [Hvt2 Hagree2];
        split; [assumption | assumption]
      end.
    + exfalso.
      match goal with [ H : binop_allows_bool_args _ = true |- _ ] => simpl in H; discriminate end.

  - (* E_Or_Short *)
    inversion Htype; subst.
    + exfalso.
      match goal with
      | [ Ht : has_type _ e1 TInt |- _ ] =>
        destruct (IHHeval _ _ Ht Hagree) as [Hvt1 _]; inversion Hvt1
      end.
    + destruct (IHHeval _ _ H5 Hagree) as [_ Hagree1].
      split; [constructor | assumption].
    + exfalso.
      match goal with [ H : binop_allows_bool_args _ = true |- _ ] => simpl in H; discriminate end.

  - (* E_Neg *)
    inversion Htype; subst.
    destruct (IHHeval _ _ H1 Hagree) as [_ Hagree1].
    split; [constructor | assumption].

  - (* E_Not *)
    inversion Htype; subst.
    destruct (IHHeval _ _ H1 Hagree) as [_ Hagree1].
    split; [constructor | assumption].

  - (* E_IfTrue *)
    inversion Htype; subst.
    match goal with
    | [ Hc : has_type _ e1 TBool, Ht : has_type _ e2 _ |- _ ] =>
      destruct (IHHeval1 _ _ Hc Hagree) as [_ Hagree1];
      exact (IHHeval2 _ _ Ht Hagree1)
    end.

  - (* E_IfFalse *)
    inversion Htype; subst.
    match goal with
    | [ Hc : has_type _ e1 TBool, Hf : has_type _ e3 _ |- _ ] =>
      destruct (IHHeval1 _ _ Hc Hagree) as [_ Hagree1];
      exact (IHHeval2 _ _ Hf Hagree1)
    end.

  - (* E_Let *)
    inversion Htype; subst.
    match goal with
    | [ He1 : has_type _ e1 ?t1, He2 : has_type (CtxCons _ ?t1 _) e2 _ |- _ ] =>
      destruct (IHHeval1 _ _ He1 Hagree) as [Hvt1 Hagree1];
      assert (Hagree_ext : env_ctx_agree (ECons x v1 renv1) (CtxCons x t1 gamma))
        by (apply agree_cons; assumption);
      destruct (IHHeval2 _ _ He2 Hagree_ext) as [Hvt2 Hagree2];
      (* Hagree2 : env_ctx_agree (ECons x vx renv_out) (CtxCons x t1 gamma) *)
      inversion Hagree2; subst;
      split; assumption
    end.

  - (* E_Set *)
    inversion Htype; subst.
    match goal with
    | [ Hctx : ctx_lookup _ _ = Some ?t, He : has_type _ e ?t |- _ ] =>
      destruct (IHHeval _ _ He Hagree) as [Hvt Hagree1];
      split;
      [ constructor
      | eapply agree_update; eassumption ]
    end.

  - (* E_Seq *)
    inversion Htype; subst.
    match goal with
    | [ He1 : has_type _ e1 _, He2 : has_type _ e2 _ |- _ ] =>
      destruct (IHHeval1 _ _ He1 Hagree) as [_ Hagree1];
      exact (IHHeval2 _ _ He2 Hagree1)
    end.

  - (* E_WhileTrue *)
    inversion Htype; subst.
    match goal with
    | [ Hc : has_type _ cond TBool, Hb : has_type _ body TUnit |- _ ] =>
      destruct (IHHeval1 _ _ Hc Hagree) as [_ Hagree1];
      destruct (IHHeval2 _ _ Hb Hagree1) as [_ Hagree2];
      exact (IHHeval3 _ _ Htype Hagree2)
    end.

  - (* E_WhileFalse *)
    inversion Htype; subst.
    match goal with
    | [ Hc : has_type _ cond TBool |- _ ] =>
      destruct (IHHeval _ _ Hc Hagree) as [_ Hagree1];
      split; [constructor | assumption]
    end.

  - (* E_Lam *)
    inversion Htype; subst.
    split; [econstructor; eassumption | assumption].

  - (* E_App *)
    inversion Htype; subst.
    match goal with
    | [ H1 : has_type _ e1 (TArrow ?ta ?tr),
        H2 : has_type _ e2 ?ta |- _ ] =>
      destruct (IHHeval1 _ _ H1 Hagree) as [Hf Hagree1];
      destruct (IHHeval2 _ _ H2 Hagree1) as [Harg Hagree2];
      apply clos_type_inv in Hf;
      destruct Hf as [c' [Hcenv Hbody_type]];
      assert (Hagree_body : env_ctx_agree (ECons x v2 clos_env) (CtxCons x ta c'))
        by (apply agree_cons; assumption);
      destruct (IHHeval3 _ _ Hbody_type Hagree_body) as [Hvt _];
      split; [exact Hvt | exact Hagree2]
    end.
Qed.

(** ** Type soundness corollary *)

Corollary soundness : forall e t v renv',
  has_type CtxNil e t ->
  eval ENil e renv' v ->
  val_has_type v t.
Proof.
  intros.
  eapply preservation; try eassumption.
  constructor.
Qed.

(** ** Progress

    The progress theorem is proved in [Progress.v] using small-step
    semantics: a well-typed closed expression is either a value or
    can take a reduction step. See Progress.v for details. *)

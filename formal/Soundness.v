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
From Stdlib Require Import List.
Import ListNotations.
From NanoCore Require Import Syntax.
From NanoCore Require Import Semantics.
From NanoCore Require Import Typing.
Open Scope string_scope.

(** ** Value typing

    Defines when a value has a given type. *)

Inductive val_has_type : val -> ty -> Prop :=
  | VT_Int : forall n, val_has_type (VInt n) TInt
  | VT_Bool : forall b, val_has_type (VBool b) TBool
  | VT_String : forall s, val_has_type (VString s) TString
  | VT_Unit : val_has_type VUnit TUnit
  | VT_Clos : forall x body clos_env t1 t2 c,
      env_ctx_agree clos_env c ->
      has_type (CtxCons x t1 c) body t2 ->
      val_has_type (VClos x body clos_env) (TArrow t1 t2)
  | VT_FixClos : forall f x body clos_env t1 t2 c,
      env_ctx_agree clos_env c ->
      has_type (CtxCons x t1 (CtxCons f (TArrow t1 t2) c)) body t2 ->
      val_has_type (VFixClos f x body clos_env) (TArrow t1 t2)
  | VT_Array : forall vs t,
      Forall (fun v => val_has_type v t) vs ->
      val_has_type (VArray vs) (TArray t)
  | VT_Record : forall fvs fts,
      Forall2 (fun fv ft => fst fv = fst ft /\ val_has_type (snd fv) (snd ft)) fvs fts ->
      val_has_type (VRecord fvs) (TRecord fts)
  | VT_Construct : forall tag v fts t,
      assoc_lookup tag fts = Some t ->
      val_has_type v t ->
      val_has_type (VConstruct tag v) (TVariant fts)

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

Lemma canonical_string : forall v,
  val_has_type v TString -> exists s, v = VString s.
Proof.
  intros v H. inversion H. exists s. reflexivity.
Qed.

Lemma canonical_unit : forall v,
  val_has_type v TUnit -> v = VUnit.
Proof.
  intros v H. inversion H. reflexivity.
Qed.

Lemma canonical_record : forall v fts,
  val_has_type v (TRecord fts) -> exists fvs, v = VRecord fvs.
Proof.
  intros v fts H. inversion H. exists fvs. reflexivity.
Qed.

Lemma canonical_array : forall v t,
  val_has_type v (TArray t) -> exists vs, v = VArray vs.
Proof.
  intros v t H. inversion H. exists vs. reflexivity.
Qed.

Lemma canonical_arrow : forall v t1 t2,
  val_has_type v (TArrow t1 t2) ->
  (exists x body clos_env, v = VClos x body clos_env) \/
  (exists f x body clos_env, v = VFixClos f x body clos_env).
Proof.
  intros v t1 t2 H. inversion H; subst.
  - left. exists x, body, clos_env. reflexivity.
  - right. exists f, x, body, clos_env. reflexivity.
Qed.

Lemma canonical_variant : forall v fts,
  val_has_type v (TVariant fts) -> exists tag v', v = VConstruct tag v'.
Proof.
  intros v fts H. inversion H. exists tag, v0. reflexivity.
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

(** ** Inversion lemma for fix closure typing *)

Lemma fix_clos_type_inv : forall f x body cenv t1 t2,
  val_has_type (VFixClos f x body cenv) (TArrow t1 t2) ->
  exists c,
    env_ctx_agree cenv c /\
    has_type (CtxCons x t1 (CtxCons f (TArrow t1 t2) c)) body t2.
Proof.
  intros. inversion H; subst.
  exists c. split; assumption.
Qed.

(** ** Helper: Forall and nth_error *)

Lemma Forall_nth_error : forall (A : Type) (P : A -> Prop) (l : list A) (n : nat) (x : A),
  Forall P l -> nth_error l n = Some x -> P x.
Proof.
  intros A P l n x HF Hnth.
  rewrite Forall_forall in HF.
  apply HF. eapply nth_error_In. eassumption.
Qed.

(** ** Inversion lemma for array typing *)

Lemma array_type_inv : forall vs t,
  val_has_type (VArray vs) (TArray t) ->
  Forall (fun v => val_has_type v t) vs.
Proof.
  intros. inversion H; subst. assumption.
Qed.

(** ** Array indexing preserves types *)

Lemma array_nth_type : forall vs t n v,
  val_has_type (VArray vs) (TArray t) ->
  nth_error vs n = Some v ->
  val_has_type v t.
Proof.
  intros vs t n v Hvt Hnth.
  apply array_type_inv in Hvt.
  rewrite Forall_forall in Hvt.
  apply Hvt. eapply nth_error_In. exact Hnth.
Qed.

(** ** List update preserves Forall *)

Lemma list_update_Forall : forall (A : Type) (P : A -> Prop) (l : list A) n v,
  Forall P l -> P v -> Forall P (list_update n v l).
Proof.
  intros A P l. induction l as [|x rest IH]; intros n v HF Hv.
  - destruct n; exact HF.
  - destruct n.
    + constructor; [assumption |]. inversion HF; assumption.
    + inversion HF; subst. constructor; [assumption | apply IH; assumption].
Qed.

(** ** App preserves Forall *)

Lemma app_Forall : forall (A : Type) (P : A -> Prop) (l : list A) v,
  Forall P l -> P v -> Forall P (l ++ [v]).
Proof.
  intros. apply Forall_app. split; [assumption | constructor; [assumption | constructor]].
Qed.

(** ** Inversion lemma for record typing *)

Lemma record_type_inv : forall fvs fts,
  val_has_type (VRecord fvs) (TRecord fts) ->
  Forall2 (fun fv ft => fst fv = fst ft /\ val_has_type (snd fv) (snd ft)) fvs fts.
Proof.
  intros. inversion H; subst. assumption.
Qed.

(** ** Record field lookup preserves types *)

Lemma record_field_type : forall fvs fts f v,
  Forall2 (fun fv ft => fst fv = fst ft /\ val_has_type (snd fv) (snd ft)) fvs fts ->
  assoc_lookup f fvs = Some v ->
  exists t, assoc_lookup f fts = Some t /\ val_has_type v t.
Proof.
  intros fvs fts f v HF2 Hlookup.
  induction HF2 as [| [f1 v1] [f2 t2] fvs' fts' [Hname Hvt] HF2' IH].
  - simpl in Hlookup. discriminate.
  - simpl in *. subst.
    destruct (String.eqb f f2) eqn:Heq.
    + inversion Hlookup; subst. exists t2. split; [reflexivity | assumption].
    + apply IH. assumption.
Qed.

(** Record field lookup: type-directed version *)

Lemma record_field_lookup : forall fvs fts f t,
  Forall2 (fun fv ft => fst fv = fst ft /\ val_has_type (snd fv) (snd ft)) fvs fts ->
  assoc_lookup f fts = Some t ->
  exists v, assoc_lookup f fvs = Some v /\ val_has_type v t.
Proof.
  intros fvs fts f t HF2 Hlookup.
  induction HF2 as [| [f1 v1] [f2 t2] fvs' fts' [Hname Hvt] HF2' IH].
  - simpl in Hlookup. discriminate.
  - simpl in *. subst.
    destruct (String.eqb f f2) eqn:Heq.
    + inversion Hlookup; subst. exists v1. split; [reflexivity | assumption].
    + apply IH. assumption.
Qed.

(** ** Record field update preserves typing *)

Lemma assoc_update_preserves_type : forall fvs fts f v t,
  Forall2 (fun fv ft => fst fv = fst ft /\ val_has_type (snd fv) (snd ft)) fvs fts ->
  assoc_lookup f fts = Some t ->
  val_has_type v t ->
  Forall2 (fun fv ft => fst fv = fst ft /\ val_has_type (snd fv) (snd ft))
          (assoc_update f v fvs) fts.
Proof.
  intros fvs fts f v t HF2.
  induction HF2 as [| [f1 v1] [f2 t2] fvs' fts' [Hname Hvt] HF2' IH];
    intros Hlookup Hvnew.
  - simpl in Hlookup. discriminate.
  - simpl in *. subst.
    destruct (String.eqb f f2) eqn:Heq.
    + inversion Hlookup; subst.
      constructor; [simpl; split; [reflexivity | assumption] | assumption].
    + constructor; [simpl; split; [reflexivity | assumption] | apply IH; assumption].
Qed.

(** ** Variant inversion *)

Lemma construct_type_inv : forall tag v fts,
  val_has_type (VConstruct tag v) (TVariant fts) ->
  exists t, assoc_lookup tag fts = Some t /\ val_has_type v t.
Proof.
  intros. inversion H; subst. exists t. split; assumption.
Qed.

(** ** Branch typing helpers *)

Lemma branches_type_find : forall ctx branches fts t tag t_payload,
  branches_type ctx branches fts t ->
  assoc_lookup tag fts = Some t_payload ->
  exists x body, find_branch tag branches = Some (x, body).
Proof.
  intros ctx branches fts t tag t_payload Hbt Hlookup.
  induction Hbt.
  - simpl in Hlookup. discriminate.
  - simpl in *. destruct (String.eqb tag tag0) eqn:Heq.
    + exists x, body. reflexivity.
    + apply IHHbt. assumption.
Qed.

Lemma branches_type_payload : forall ctx branches fts t tag x body t_payload,
  branches_type ctx branches fts t ->
  assoc_lookup tag fts = Some t_payload ->
  find_branch tag branches = Some (x, body) ->
  has_type (CtxCons x t_payload ctx) body t.
Proof.
  intros ctx branches fts t tag x body t_payload Hbt Hlookup Hfind.
  induction Hbt.
  - simpl in Hlookup. discriminate.
  - simpl in *. destruct (String.eqb tag tag0) eqn:Heq.
    + inversion Hlookup; subst. inversion Hfind; subst. assumption.
    + apply IHHbt; assumption.
Qed.

(** ** Tactic for impossible type cases

    When an expression evaluates to a value of one type (e.g., VInt)
    but the typing rule requires a different type (e.g., TString),
    we derive a contradiction via the IH. *)

Ltac type_contradiction :=
  exfalso;
  match goal with
  | [ IH : forall _ _, has_type _ ?e _ -> _ -> _,
      Ht : has_type _ ?e ?T,
      Hagr : env_ctx_agree _ _ |- _ ] =>
    destruct (IH _ _ Ht Hagr) as [?Hvt _]; inversion Hvt
  end.

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

  - (* E_String *)
    inversion Htype; subst. split; [constructor | assumption].

  - (* E_Unit *)
    inversion Htype; subst. split; [constructor | assumption].

  - (* E_Var *)
    inversion Htype; subst.
    pose proof (agree_lookup _ _ _ _ Hagree H2) as [v' [Hl Hvt]].
    rewrite H in Hl. inversion Hl; subst.
    split; [assumption | assumption].

  - (* E_BinArith: int op int -> int *)
    inversion Htype; subst.
    + (* T_BinOp *)
      match goal with
      | [ He1 : has_type _ e1 TInt, He2 : has_type _ e2 TInt |- _ ] =>
        destruct (IHHeval1 _ _ He1 Hagree) as [Hvt1 Hagree1];
        destruct (IHHeval2 _ _ He2 Hagree1) as [Hvt2 Hagree2];
        split; [eapply eval_arith_binop_type; eassumption | assumption]
      end.
    + (* T_BinLogic - impossible: e1 typed Bool but evals to Int *)
      type_contradiction.
    + (* T_BinEqBool - impossible *)
      type_contradiction.
    + (* T_StrCat - impossible: e1 typed String but evals to Int *)
      type_contradiction.
    + (* T_BinEqStr - impossible *)
      type_contradiction.

  - (* E_BinCmp: int op int -> bool *)
    inversion Htype; subst.
    + match goal with
      | [ He1 : has_type _ e1 TInt, He2 : has_type _ e2 TInt |- _ ] =>
        destruct (IHHeval1 _ _ He1 Hagree) as [Hvt1 Hagree1];
        destruct (IHHeval2 _ _ He2 Hagree1) as [Hvt2 Hagree2];
        split; [eapply eval_cmp_binop_type; eassumption | assumption]
      end.
    + type_contradiction.
    + type_contradiction.
    + type_contradiction.
    + type_contradiction.

  - (* E_BinEqBool: bool == bool -> bool *)
    inversion Htype; subst.
    + (* T_BinOp: impossible - e1 typed Int but evals to Bool *)
      type_contradiction.
    + (* T_BinLogic: impossible - OpEq not a logic op *)
      exfalso.
      match goal with [ H : binop_arg_type _ = TBool |- _ ] => subst; simpl in H; discriminate end.
    + (* T_BinEqBool: the real case *)
      match goal with
      | [ He1 : has_type _ e1 TBool, He2 : has_type _ e2 TBool |- _ ] =>
        destruct (IHHeval1 _ _ He1 Hagree) as [_ Hagree1];
        destruct (IHHeval2 _ _ He2 Hagree1) as [_ Hagree2];
        split; [constructor | assumption]
      end.
    + (* T_StrCat: impossible - OpEq is not OpStrCat *)
      exfalso. subst. discriminate.
    + (* T_BinEqStr: impossible - e1 typed String but evals to Bool *)
      type_contradiction.

  - (* E_BinNeBool: bool != bool -> bool *)
    inversion Htype; subst.
    + type_contradiction.
    + exfalso.
      match goal with [ H : binop_arg_type _ = TBool |- _ ] => subst; simpl in H; discriminate end.
    + match goal with
      | [ He1 : has_type _ e1 TBool, He2 : has_type _ e2 TBool |- _ ] =>
        destruct (IHHeval1 _ _ He1 Hagree) as [_ Hagree1];
        destruct (IHHeval2 _ _ He2 Hagree1) as [_ Hagree2];
        split; [constructor | assumption]
      end.
    + exfalso. subst. discriminate.
    + type_contradiction.

  - (* E_StrCat: string ++ string -> string *)
    inversion Htype; subst.
    + (* T_BinOp: impossible - e1 typed Int but evals to String *)
      type_contradiction.
    + (* T_BinLogic: impossible *)
      type_contradiction.
    + (* T_BinEqBool: impossible *)
      type_contradiction.
    + (* T_StrCat: the real case *)
      match goal with
      | [ He1 : has_type _ e1 TString, He2 : has_type _ e2 TString |- _ ] =>
        destruct (IHHeval1 _ _ He1 Hagree) as [_ Hagree1];
        destruct (IHHeval2 _ _ He2 Hagree1) as [_ Hagree2];
        split; [constructor | assumption]
      end.
    + (* T_BinEqStr: impossible - OpStrCat not eq/ne *)
      exfalso.
      match goal with [ H : binop_allows_string_args OpStrCat = true |- _ ] => simpl in H; discriminate end.

  - (* E_BinEqStr: string == string -> bool *)
    inversion Htype; subst.
    + type_contradiction.
    + type_contradiction.
    + type_contradiction.
    + (* T_StrCat: impossible - OpEq not OpStrCat *)
      exfalso. subst. discriminate.
    + (* T_BinEqStr: the real case *)
      match goal with
      | [ He1 : has_type _ e1 TString, He2 : has_type _ e2 TString |- _ ] =>
        destruct (IHHeval1 _ _ He1 Hagree) as [_ Hagree1];
        destruct (IHHeval2 _ _ He2 Hagree1) as [_ Hagree2];
        split; [constructor | assumption]
      end.

  - (* E_BinNeStr: string != string -> bool *)
    inversion Htype; subst.
    + type_contradiction.
    + type_contradiction.
    + type_contradiction.
    + exfalso. subst. discriminate.
    + match goal with
      | [ He1 : has_type _ e1 TString, He2 : has_type _ e2 TString |- _ ] =>
        destruct (IHHeval1 _ _ He1 Hagree) as [_ Hagree1];
        destruct (IHHeval2 _ _ He2 Hagree1) as [_ Hagree2];
        split; [constructor | assumption]
      end.

  - (* E_And_True *)
    inversion Htype; subst.
    + type_contradiction.
    + match goal with
      | [ He1 : has_type _ e1 TBool, He2 : has_type _ e2 TBool |- _ ] =>
        destruct (IHHeval1 _ _ He1 Hagree) as [_ Hagree1];
        destruct (IHHeval2 _ _ He2 Hagree1) as [Hvt2 Hagree2];
        split; [assumption | assumption]
      end.
    + exfalso.
      match goal with [ H : binop_allows_bool_args _ = true |- _ ] => simpl in H; discriminate end.
    + type_contradiction.

  - (* E_And_False *)
    inversion Htype; subst.
    + type_contradiction.
    + match goal with
      | [ He1 : has_type _ e1 TBool |- _ ] =>
        destruct (IHHeval _ _ He1 Hagree) as [_ Hagree1];
        split; [constructor | assumption]
      end.
    + exfalso.
      match goal with [ H : binop_allows_bool_args _ = true |- _ ] => simpl in H; discriminate end.
    + type_contradiction.

  - (* E_And_Short *)
    inversion Htype; subst.
    + type_contradiction.
    + match goal with
      | [ He1 : has_type _ e1 TBool |- _ ] =>
        destruct (IHHeval _ _ He1 Hagree) as [_ Hagree1];
        split; [constructor | assumption]
      end.
    + exfalso.
      match goal with [ H : binop_allows_bool_args _ = true |- _ ] => simpl in H; discriminate end.
    + type_contradiction.

  - (* E_Or_False *)
    inversion Htype; subst.
    + type_contradiction.
    + match goal with
      | [ He1 : has_type _ e1 TBool, He2 : has_type _ e2 TBool |- _ ] =>
        destruct (IHHeval1 _ _ He1 Hagree) as [_ Hagree1];
        destruct (IHHeval2 _ _ He2 Hagree1) as [Hvt2 Hagree2];
        split; [assumption | assumption]
      end.
    + exfalso.
      match goal with [ H : binop_allows_bool_args _ = true |- _ ] => simpl in H; discriminate end.
    + type_contradiction.

  - (* E_Or_Short *)
    inversion Htype; subst.
    + type_contradiction.
    + match goal with
      | [ He1 : has_type _ e1 TBool |- _ ] =>
        destruct (IHHeval _ _ He1 Hagree) as [_ Hagree1];
        split; [constructor | assumption]
      end.
    + exfalso.
      match goal with [ H : binop_allows_bool_args _ = true |- _ ] => simpl in H; discriminate end.
    + type_contradiction.

  - (* E_Neg *)
    inversion Htype; subst.
    destruct (IHHeval _ _ H1 Hagree) as [_ Hagree1].
    split; [constructor | assumption].

  - (* E_Not *)
    inversion Htype; subst.
    destruct (IHHeval _ _ H1 Hagree) as [_ Hagree1].
    split; [constructor | assumption].

  - (* E_StrLen *)
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

  - (* E_App: non-recursive closure *)
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

  - (* E_ArrayNil *)
    inversion Htype; subst.
    split; [constructor; constructor | assumption].

  - (* E_ArrayCons *)
    inversion Htype; subst.
    match goal with
    | [ He : has_type _ e ?t, Hes : has_type _ (EArray es) (TArray ?t) |- _ ] =>
      destruct (IHHeval1 _ _ He Hagree) as [Hvt1 Hagree1];
      destruct (IHHeval2 _ _ Hes Hagree1) as [Hvt2 Hagree2];
      apply array_type_inv in Hvt2;
      split; [constructor; constructor; assumption | assumption]
    end.

  - (* E_Index *)
    inversion Htype; subst.
    destruct (IHHeval1 _ _ ltac:(eassumption) Hagree) as [Hvt1 Hagree1].
    destruct (IHHeval2 _ _ ltac:(eassumption) Hagree1) as [_ Hagree2].
    split; [| assumption].
    eapply array_nth_type; eassumption.

  - (* E_ArrayLen *)
    inversion Htype; subst.
    match goal with
    | [ He : has_type _ e (TArray _) |- _ ] =>
      destruct (IHHeval _ _ He Hagree) as [_ Hagree1];
      split; [constructor | assumption]
    end.

  - (* E_RecordNil *)
    inversion Htype; subst.
    split; [constructor; constructor | assumption].

  - (* E_RecordCons *)
    inversion Htype; subst.
    match goal with
    | [ He : has_type _ e ?t, Hes : has_type _ (ERecord es) (TRecord ?fts) |- _ ] =>
      destruct (IHHeval1 _ _ He Hagree) as [Hvt1 Hagree1];
      destruct (IHHeval2 _ _ Hes Hagree1) as [Hvt2 Hagree2];
      apply record_type_inv in Hvt2;
      split; [constructor; constructor; [simpl; split; [reflexivity | assumption] | assumption] | assumption]
    end.

  - (* E_Field *)
    inversion Htype; subst.
    destruct (IHHeval _ _ ltac:(eassumption) Hagree) as [Hvt1 Hagree1].
    apply record_type_inv in Hvt1.
    split; [| assumption].
    match goal with
    | [ Hval : assoc_lookup f fvs = Some v,
        Htyp : assoc_lookup f ?fts = Some ty0 |- _ ] =>
      destruct (record_field_type _ _ _ _ Hvt1 Hval) as [t' [Ht' Hvt']];
      rewrite Htyp in Ht'; inversion Ht'; subst;
      assumption
    end.

  - (* E_SetField *)
    inversion Htype; subst.
    destruct (IHHeval _ _ ltac:(eassumption) Hagree) as [Hvt_new Hagree1].
    split; [constructor |].
    eapply agree_update; [exact Hagree1 | eassumption |].
    (* Need: val_has_type (VRecord (assoc_update f v fvs)) (TRecord fts) *)
    pose proof (agree_lookup _ _ _ _ Hagree1 ltac:(eassumption)) as [vrec [Hlookup_rec Hvt_rec]].
    rewrite H in Hlookup_rec. inversion Hlookup_rec; subst.
    apply record_type_inv in Hvt_rec.
    constructor.
    eapply assoc_update_preserves_type; eassumption.

  - (* E_Fix *)
    inversion Htype; subst.
    split; [econstructor; eassumption | assumption].

  - (* E_AppFix: recursive closure application *)
    inversion Htype; subst.
    destruct (IHHeval1 _ _ ltac:(eassumption) Hagree) as [Hf Hagree1].
    destruct (IHHeval2 _ _ ltac:(eassumption) Hagree1) as [Harg Hagree2].
    split; [| exact Hagree2].
    apply fix_clos_type_inv in Hf.
    destruct Hf as [c' [Hcenv Hbody_type]].
    eapply IHHeval3; [exact Hbody_type |].
    constructor; [assumption |].
    constructor; [| assumption].
    econstructor; eassumption.

  - (* E_Construct *)
    inversion Htype; subst.
    destruct (IHHeval _ _ ltac:(eassumption) Hagree) as [Hvt Hagree1].
    split; [| assumption].
    econstructor; eassumption.

  - (* E_Match *)
    inversion Htype; subst.
    destruct (IHHeval1 _ _ ltac:(eassumption) Hagree) as [Hvt_scr Hagree1].
    apply construct_type_inv in Hvt_scr.
    destruct Hvt_scr as [t_payload [Htag Hvt_payload]].
    assert (Hbody_type : has_type (CtxCons x t_payload gamma) body ty0).
    { eapply branches_type_payload; eassumption. }
    assert (Hagree_ext : env_ctx_agree (ECons x v renv1) (CtxCons x t_payload gamma))
      by (apply agree_cons; assumption).
    destruct (IHHeval2 _ _ Hbody_type Hagree_ext) as [Hvt_result Hagree_out].
    inversion Hagree_out; subst.
    split; assumption.

  - (* E_ArraySet *)
    inversion Htype; subst.
    destruct (IHHeval1 _ _ ltac:(eassumption) Hagree) as [Hvt1 Hagree1].
    destruct (IHHeval2 _ _ ltac:(eassumption) Hagree1) as [_ Hagree2].
    destruct (IHHeval3 _ _ ltac:(eassumption) Hagree2) as [Hvt3 Hagree3].
    apply array_type_inv in Hvt1.
    split; [| assumption].
    constructor. apply list_update_Forall; assumption.

  - (* E_ArrayPush *)
    inversion Htype; subst.
    destruct (IHHeval1 _ _ ltac:(eassumption) Hagree) as [Hvt1 Hagree1].
    destruct (IHHeval2 _ _ ltac:(eassumption) Hagree1) as [Hvt2 Hagree2].
    apply array_type_inv in Hvt1.
    split; [| assumption].
    constructor. apply app_Forall; assumption.

  - (* E_StrIndex *)
    inversion Htype; subst.
    destruct (IHHeval1 _ _ ltac:(eassumption) Hagree) as [_ Hagree1].
    destruct (IHHeval2 _ _ ltac:(eassumption) Hagree1) as [_ Hagree2].
    split; [constructor | assumption].
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

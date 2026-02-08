(** * NanoCore: Semantic Equivalence (Big-Step ↔ Small-Step)

    For the pure (mutation-free) fragment of NanoCore, big-step
    evaluation and small-step reduction agree (up to type annotations
    erased by closures).

    Approach: We prove the forward direction for CLOSED expressions
    evaluated in the EMPTY environment. This avoids close/subst
    commutation issues. The result uses [multi_step_equiv] which
    allows the target to differ from [val_to_expr v] only in
    type annotations on ELam, EFix, and EConstruct (which are
    computationally irrelevant).

    Main theorem:
      eval ENil e ENil v → multi_step_equiv e (val_to_expr v)
*)

From Stdlib Require Import ZArith.
From Stdlib Require Import Bool.
From Stdlib Require Import String.
From Stdlib Require Import List.
Import ListNotations.
From NanoCore Require Import Syntax.
From NanoCore Require Import Semantics.
From NanoCore Require Import Typing.
From NanoCore Require Import Progress.
Open Scope Z_scope.
Open Scope string_scope.

(** ** Value-to-expression conversion *)

Fixpoint val_to_expr (v : val) : expr :=
  match v with
  | VInt n => EInt n
  | VBool b => EBool b
  | VString s => EString s
  | VUnit => EUnit
  | VClos x body cenv => ELam x TUnit (close cenv body)
  | VFixClos f x body cenv => EFix f x TUnit TUnit (close cenv body)
  | VArray vs => EArray (map val_to_expr vs)
  | VRecord fvs => ERecord (map (fun '(f, v) => (f, val_to_expr v)) fvs)
  | VConstruct tag v => EConstruct tag (val_to_expr v) TUnit
  end

with close (renv : env) (e : expr) : expr :=
  match renv with
  | ENil => e
  | ECons x v rest => close rest (subst x (val_to_expr v) e)
  end.

(** ** Reflexive-transitive closure of step *)

Inductive multi_step : expr -> expr -> Prop :=
  | MS_Refl : forall e, multi_step e e
  | MS_Step : forall e1 e2 e3,
      step e1 e2 -> multi_step e2 e3 -> multi_step e1 e3.

(** ** Expression equivalence modulo type annotations.

    Type annotations on ELam, EFix, and EConstruct have no
    computational significance — they don't affect reduction.
    This relation captures "same expression modulo types". *)

Inductive expr_equiv : expr -> expr -> Prop :=
  | EQ_Int : forall n, expr_equiv (EInt n) (EInt n)
  | EQ_Bool : forall b, expr_equiv (EBool b) (EBool b)
  | EQ_String : forall s, expr_equiv (EString s) (EString s)
  | EQ_Unit : expr_equiv EUnit EUnit
  | EQ_Var : forall x, expr_equiv (EVar x) (EVar x)
  | EQ_BinOp : forall op e1 e1' e2 e2',
      expr_equiv e1 e1' -> expr_equiv e2 e2' ->
      expr_equiv (EBinOp op e1 e2) (EBinOp op e1' e2')
  | EQ_UnOp : forall op e e',
      expr_equiv e e' -> expr_equiv (EUnOp op e) (EUnOp op e')
  | EQ_If : forall e1 e1' e2 e2' e3 e3',
      expr_equiv e1 e1' -> expr_equiv e2 e2' -> expr_equiv e3 e3' ->
      expr_equiv (EIf e1 e2 e3) (EIf e1' e2' e3')
  | EQ_Let : forall x e1 e1' e2 e2',
      expr_equiv e1 e1' -> expr_equiv e2 e2' ->
      expr_equiv (ELet x e1 e2) (ELet x e1' e2')
  | EQ_Set : forall x e e',
      expr_equiv e e' -> expr_equiv (ESet x e) (ESet x e')
  | EQ_Seq : forall e1 e1' e2 e2',
      expr_equiv e1 e1' -> expr_equiv e2 e2' ->
      expr_equiv (ESeq e1 e2) (ESeq e1' e2')
  | EQ_While : forall e1 e1' e2 e2',
      expr_equiv e1 e1' -> expr_equiv e2 e2' ->
      expr_equiv (EWhile e1 e2) (EWhile e1' e2')
  | EQ_Lam : forall x t1 t2 body body',
      expr_equiv body body' ->
      expr_equiv (ELam x t1 body) (ELam x t2 body')
  | EQ_App : forall e1 e1' e2 e2',
      expr_equiv e1 e1' -> expr_equiv e2 e2' ->
      expr_equiv (EApp e1 e2) (EApp e1' e2')
  | EQ_Fix : forall f x t1 t1' t2 t2' body body',
      expr_equiv body body' ->
      expr_equiv (EFix f x t1 t2 body) (EFix f x t1' t2' body')
  | EQ_Array : forall es es',
      Forall2 expr_equiv es es' ->
      expr_equiv (EArray es) (EArray es')
  | EQ_Index : forall e1 e1' e2 e2',
      expr_equiv e1 e1' -> expr_equiv e2 e2' ->
      expr_equiv (EIndex e1 e2) (EIndex e1' e2')
  | EQ_ArraySet : forall e1 e1' e2 e2' e3 e3',
      expr_equiv e1 e1' -> expr_equiv e2 e2' -> expr_equiv e3 e3' ->
      expr_equiv (EArraySet e1 e2 e3) (EArraySet e1' e2' e3')
  | EQ_ArrayPush : forall e1 e1' e2 e2',
      expr_equiv e1 e1' -> expr_equiv e2 e2' ->
      expr_equiv (EArrayPush e1 e2) (EArrayPush e1' e2')
  | EQ_Record : forall fes fes',
      Forall2 (fun fe fe' => fst fe = fst fe' /\
               expr_equiv (snd fe) (snd fe')) fes fes' ->
      expr_equiv (ERecord fes) (ERecord fes')
  | EQ_Field : forall e e' f,
      expr_equiv e e' -> expr_equiv (EField e f) (EField e' f)
  | EQ_SetField : forall x f e e',
      expr_equiv e e' -> expr_equiv (ESetField x f e) (ESetField x f e')
  | EQ_Construct : forall tag e e' t1 t2,
      expr_equiv e e' ->
      expr_equiv (EConstruct tag e t1) (EConstruct tag e' t2)
  | EQ_Match : forall e e' branches branches',
      expr_equiv e e' ->
      Forall2 (fun b b' => fst (fst b) = fst (fst b')
               /\ snd (fst b) = snd (fst b')
               /\ expr_equiv (snd b) (snd b')) branches branches' ->
      expr_equiv (EMatch e branches) (EMatch e' branches')
  | EQ_StrIndex : forall e1 e1' e2 e2',
      expr_equiv e1 e1' -> expr_equiv e2 e2' ->
      expr_equiv (EStrIndex e1 e2) (EStrIndex e1' e2').

(** ** Multi-step equivalence: multi_step to an expr_equiv target *)

Definition multi_step_equiv (e1 e2 : expr) : Prop :=
  exists e2', multi_step e1 e2' /\ expr_equiv e2' e2.

(** ** Pure expression predicate *)

Inductive pure : expr -> Prop :=
  | P_Int : forall n, pure (EInt n)
  | P_Bool : forall b, pure (EBool b)
  | P_String : forall s, pure (EString s)
  | P_Unit : pure EUnit
  | P_Var : forall x, pure (EVar x)
  | P_BinOp : forall op e1 e2, pure e1 -> pure e2 -> pure (EBinOp op e1 e2)
  | P_UnOp : forall op e, pure e -> pure (EUnOp op e)
  | P_If : forall e1 e2 e3, pure e1 -> pure e2 -> pure e3 -> pure (EIf e1 e2 e3)
  | P_Let : forall x e1 e2, pure e1 -> pure e2 -> pure (ELet x e1 e2)
  | P_Seq : forall e1 e2, pure e1 -> pure e2 -> pure (ESeq e1 e2)
  | P_Lam : forall x t body, pure body -> pure (ELam x t body)
  | P_App : forall e1 e2, pure e1 -> pure e2 -> pure (EApp e1 e2)
  | P_Fix : forall f x t1 t2 body, pure body -> pure (EFix f x t1 t2 body)
  | P_Array : forall es, Forall pure es -> pure (EArray es)
  | P_Index : forall e1 e2, pure e1 -> pure e2 -> pure (EIndex e1 e2)
  | P_ArraySet : forall e1 e2 e3, pure e1 -> pure e2 -> pure e3 -> pure (EArraySet e1 e2 e3)
  | P_ArrayPush : forall e1 e2, pure e1 -> pure e2 -> pure (EArrayPush e1 e2)
  | P_Record : forall fes, Forall (fun fe => pure (snd fe)) fes -> pure (ERecord fes)
  | P_Field : forall e f, pure e -> pure (EField e f)
  | P_Construct : forall tag e t, pure e -> pure (EConstruct tag e t)
  | P_Match : forall e branches,
      pure e ->
      Forall (fun b => pure (snd b)) branches ->
      pure (EMatch e branches)
  | P_StrIndex : forall e1 e2, pure e1 -> pure e2 -> pure (EStrIndex e1 e2).

(** ** Helper: find_branch preserves Forall purity *)

Lemma find_branch_pure : forall tag branches x body,
  Forall (fun b => pure (snd b)) branches ->
  find_branch tag branches = Some (x, body) ->
  pure body.
Proof.
  intros tag branches x body HF Hfind.
  induction branches as [| [[t y] b] rest IH].
  - simpl in Hfind. discriminate.
  - simpl in *. destruct (String.eqb tag t).
    + inversion Hfind; subst. inversion HF; subst. simpl in *. assumption.
    + inversion HF; subst. apply IH; assumption.
Qed.

(** ** Pure expressions don't change the environment *)

Ltac pure_chain :=
  repeat match goal with
  | [ IH : pure ?e -> ?r = ?base, Hp : pure ?e |- _ ] =>
    let H := fresh "Hpu" in pose proof (IH Hp) as H; clear IH; subst
  end;
  try reflexivity.

Ltac pure_forall_inv :=
  match goal with
  | [ H : Forall _ (_ :: _) |- _ ] => inversion H; subst; simpl in *
  end.

Lemma pure_env_unchanged : forall renv e renv' v,
  eval renv e renv' v -> pure e -> renv' = renv.
Proof.
  intros renv e renv' v Heval.
  induction Heval; intro Hp; inversion Hp; subst;
    try reflexivity.
  1: pure_chain.   (* E_BinArith *)
  1: pure_chain.   (* E_BinCmp *)
  1: pure_chain.   (* E_BinEqBool *)
  1: pure_chain.   (* E_BinNeBool *)
  1: pure_chain.   (* E_StrCat *)
  1: pure_chain.   (* E_BinEqStr *)
  1: pure_chain.   (* E_BinNeStr *)
  1: pure_chain.   (* E_And_True *)
  1: { apply IHHeval. assumption. }   (* E_And_False *)
  1: { apply IHHeval. assumption. }   (* E_And_Short *)
  1: pure_chain.   (* E_Or_False *)
  1: { apply IHHeval. assumption. }   (* E_Or_Short *)
  1: pure_chain.   (* E_Neg *)
  1: pure_chain.   (* E_Not *)
  1: pure_chain.   (* E_StrLen *)
  1: pure_chain.   (* E_IfTrue *)
  1: pure_chain.   (* E_IfFalse *)
  1: { (* E_Let *)
    assert (renv1 = renv) by auto. subst.
    assert (ECons x vx renv_out = ECons x v1 renv) by auto.
    congruence. }
  1: pure_chain.   (* E_Seq *)
  1: pure_chain.   (* E_App *)
  1: { (* E_ArrayCons *)
    pure_forall_inv. pure_chain.
    apply IHHeval2. constructor. assumption. }
  1: pure_chain.   (* E_Index *)
  1: { apply IHHeval. assumption. }   (* E_ArrayLen *)
  1: { (* E_RecordCons *)
    pure_forall_inv. pure_chain.
    apply IHHeval2. constructor. assumption. }
  1: pure_chain.   (* E_Field *)
  1: pure_chain.   (* E_AppFix *)
  1: pure_chain.   (* E_Construct *)
  1: { (* E_Match *)
    assert (renv1 = renv) by auto. subst.
    assert (ECons x vx renv_out = ECons x v renv).
    { apply IHHeval2. eapply find_branch_pure; eassumption. }
    congruence. }
  1: pure_chain.   (* E_ArraySet *)
  1: pure_chain.   (* E_ArrayPush *)
  1: pure_chain.   (* E_StrIndex *)
Qed.

(** ** Multi-step transitivity *)

Lemma multi_step_trans : forall e1 e2 e3,
  multi_step e1 e2 -> multi_step e2 e3 -> multi_step e1 e3.
Proof.
  intros e1 e2 e3 H12 H23.
  induction H12.
  - assumption.
  - eapply MS_Step; [eassumption | apply IHmulti_step; assumption].
Qed.

(** ** val_to_expr produces values *)

Lemma val_to_expr_is_value : forall v, is_value (val_to_expr v).
Proof.
  fix IH 1.
  intros [n | b | s | | x body cenv | f x body cenv | vs | fvs | tag v0]; simpl.
  - constructor.
  - constructor.
  - constructor.
  - constructor.
  - constructor.
  - constructor.
  - constructor.
    induction vs as [| v rest IHrest]; [constructor |].
    simpl. constructor; [apply IH | apply IHrest].
  - constructor.
    induction fvs as [| [fn v] rest IHrest]; [constructor |].
    simpl. constructor; [simpl; apply IH | apply IHrest].
  - constructor. apply IH.
Qed.

(** ** expr_equiv is reflexive *)

Lemma expr_equiv_refl : forall e, expr_equiv e e.
Proof.
  fix IH 1. intro e. destruct e;
    (* Handle all leaf cases and simple recursive cases *)
    try solve [constructor];
    try solve [constructor; apply IH];
    try solve [constructor; apply IH; apply IH];
    try solve [constructor; [apply IH | apply IH | apply IH]].
  - (* Array *)
    constructor.
    induction l as [| hd tl IHl]; constructor; [apply IH | apply IHl].
  - (* Record *)
    constructor.
    induction l as [| [f e0] rest IHl]; constructor.
    + simpl. split; [reflexivity | apply IH].
    + apply IHl.
  - (* Match *)
    constructor; [apply IH |].
    induction l as [| [[t y] b] rest IHl]; constructor.
    + simpl. split; [reflexivity | split; [reflexivity | apply IH]].
    + apply IHl.
Qed.

(** ** expr_equiv preserves is_value *)

Lemma expr_equiv_is_value : forall e1 e2,
  expr_equiv e1 e2 -> is_value e1 -> is_value e2.
Proof.
  fix IH 3.
  intros e1 e2 Heq Hval.
  destruct Heq; inversion Hval; subst; try constructor.
  - (* Array *)
    rename H into HF2.
    match goal with HV : Forall is_value _ |- _ => clear Hval; rename HV into Hvals end.
    induction HF2; inversion Hvals; subst; constructor.
    + eapply IH; eassumption.
    + apply IHHF2. assumption.
  - (* Record *)
    rename H into HF2.
    match goal with HV : Forall (fun fe : string * expr => is_value (snd fe)) _ |- _ =>
      clear Hval; rename HV into Hvals end.
    induction HF2; inversion Hvals; subst; constructor.
    + destruct x; destruct y; simpl in *. destruct H as [_ Heq'].
      eapply IH; eassumption.
    + apply IHHF2. assumption.
  - (* Construct *)
    eapply IH; eassumption.
Qed.

(** ** Congruence lemmas: lifting multi_step through contexts *)

Lemma ms_binop1 : forall op e1 e1' e2,
  multi_step e1 e1' -> multi_step (EBinOp op e1 e2) (EBinOp op e1' e2).
Proof.
  intros. induction H.
  - apply MS_Refl.
  - eapply MS_Step; [apply S_BinOp1; eassumption | assumption].
Qed.

Lemma ms_binop2 : forall op v1 e2 e2',
  is_value v1 -> multi_step e2 e2' ->
  multi_step (EBinOp op v1 e2) (EBinOp op v1 e2').
Proof.
  intros. induction H0.
  - apply MS_Refl.
  - eapply MS_Step; [apply S_BinOp2; eassumption | assumption].
Qed.

Lemma ms_unop : forall op e e',
  multi_step e e' -> multi_step (EUnOp op e) (EUnOp op e').
Proof.
  intros. induction H.
  - apply MS_Refl.
  - eapply MS_Step; [apply S_UnOp1; eassumption | assumption].
Qed.

Lemma ms_if : forall e1 e1' e2 e3,
  multi_step e1 e1' -> multi_step (EIf e1 e2 e3) (EIf e1' e2 e3).
Proof.
  intros. induction H.
  - apply MS_Refl.
  - eapply MS_Step; [apply S_IfCond; eassumption | assumption].
Qed.

Lemma ms_let : forall x e1 e1' e2,
  multi_step e1 e1' -> multi_step (ELet x e1 e2) (ELet x e1' e2).
Proof.
  intros. induction H.
  - apply MS_Refl.
  - eapply MS_Step; [apply S_Let1; eassumption | assumption].
Qed.

Lemma ms_app1 : forall e1 e1' e2,
  multi_step e1 e1' -> multi_step (EApp e1 e2) (EApp e1' e2).
Proof.
  intros. induction H.
  - apply MS_Refl.
  - eapply MS_Step; [apply S_App1; eassumption | assumption].
Qed.

Lemma ms_app2 : forall v1 e2 e2',
  is_value v1 -> multi_step e2 e2' ->
  multi_step (EApp v1 e2) (EApp v1 e2').
Proof.
  intros. induction H0.
  - apply MS_Refl.
  - eapply MS_Step; [apply S_App2; eassumption | assumption].
Qed.

Lemma ms_index1 : forall e1 e1' e2,
  multi_step e1 e1' -> multi_step (EIndex e1 e2) (EIndex e1' e2).
Proof.
  intros. induction H.
  - apply MS_Refl.
  - eapply MS_Step; [apply S_Index1; eassumption | assumption].
Qed.

Lemma ms_index2 : forall v1 e2 e2',
  is_value v1 -> multi_step e2 e2' ->
  multi_step (EIndex v1 e2) (EIndex v1 e2').
Proof.
  intros. induction H0.
  - apply MS_Refl.
  - eapply MS_Step; [apply S_Index2; eassumption | assumption].
Qed.

Lemma ms_arrayset1 : forall e1 e1' e2 e3,
  multi_step e1 e1' -> multi_step (EArraySet e1 e2 e3) (EArraySet e1' e2 e3).
Proof.
  intros. induction H.
  - apply MS_Refl.
  - eapply MS_Step; [apply S_ArraySet1; eassumption | assumption].
Qed.

Lemma ms_arrayset2 : forall v1 e2 e2' e3,
  is_value v1 -> multi_step e2 e2' ->
  multi_step (EArraySet v1 e2 e3) (EArraySet v1 e2' e3).
Proof.
  intros. induction H0.
  - apply MS_Refl.
  - eapply MS_Step; [apply S_ArraySet2; eassumption | assumption].
Qed.

Lemma ms_arrayset3 : forall v1 v2 e3 e3',
  is_value v1 -> is_value v2 -> multi_step e3 e3' ->
  multi_step (EArraySet v1 v2 e3) (EArraySet v1 v2 e3').
Proof.
  intros. induction H1.
  - apply MS_Refl.
  - eapply MS_Step; [apply S_ArraySet3; eassumption | assumption].
Qed.

Lemma ms_arraypush1 : forall e1 e1' e2,
  multi_step e1 e1' -> multi_step (EArrayPush e1 e2) (EArrayPush e1' e2).
Proof.
  intros. induction H.
  - apply MS_Refl.
  - eapply MS_Step; [apply S_ArrayPush1; eassumption | assumption].
Qed.

Lemma ms_arraypush2 : forall v1 e2 e2',
  is_value v1 -> multi_step e2 e2' ->
  multi_step (EArrayPush v1 e2) (EArrayPush v1 e2').
Proof.
  intros. induction H0.
  - apply MS_Refl.
  - eapply MS_Step; [apply S_ArrayPush2; eassumption | assumption].
Qed.

Lemma ms_field : forall e e' f,
  multi_step e e' -> multi_step (EField e f) (EField e' f).
Proof.
  intros. induction H.
  - apply MS_Refl.
  - eapply MS_Step; [apply S_Field1; eassumption | assumption].
Qed.

Lemma ms_construct : forall tag e e' t,
  multi_step e e' -> multi_step (EConstruct tag e t) (EConstruct tag e' t).
Proof.
  intros. induction H.
  - apply MS_Refl.
  - eapply MS_Step; [apply S_Construct1; eassumption | assumption].
Qed.

Lemma ms_match : forall e e' branches,
  multi_step e e' -> multi_step (EMatch e branches) (EMatch e' branches).
Proof.
  intros. induction H.
  - apply MS_Refl.
  - eapply MS_Step; [apply S_Match1; eassumption | assumption].
Qed.

Lemma ms_seq1 : forall e1 e1' e2,
  multi_step e1 e1' -> multi_step (ESeq e1 e2) (ESeq e1' e2).
Proof.
  intros. induction H.
  - apply MS_Refl.
  - eapply MS_Step; [apply S_Seq1; eassumption | assumption].
Qed.

Lemma ms_strindex1 : forall e1 e1' e2,
  multi_step e1 e1' -> multi_step (EStrIndex e1 e2) (EStrIndex e1' e2).
Proof.
  intros. induction H.
  - apply MS_Refl.
  - eapply MS_Step; [apply S_StrIndex1; eassumption | assumption].
Qed.

Lemma ms_strindex2 : forall v1 e2 e2',
  is_value v1 -> multi_step e2 e2' ->
  multi_step (EStrIndex v1 e2) (EStrIndex v1 e2').
Proof.
  intros. induction H0.
  - apply MS_Refl.
  - eapply MS_Step; [apply S_StrIndex2; eassumption | assumption].
Qed.

Lemma ms_array_head : forall e e' es,
  multi_step e e' -> multi_step (EArray (e :: es)) (EArray (e' :: es)).
Proof.
  intros. induction H.
  - apply MS_Refl.
  - eapply MS_Step; [apply S_ArrayHead; eassumption | assumption].
Qed.

Lemma ms_array_tail : forall v es es',
  is_value v -> multi_step (EArray es) (EArray es') ->
  multi_step (EArray (v :: es)) (EArray (v :: es')).
Proof.
  intros v es es' Hv Hms.
  remember (EArray es) as ae. remember (EArray es') as ae'.
  revert es es' Heqae Heqae'.
  induction Hms; intros; subst.
  - injection Heqae'; intros; subst. apply MS_Refl.
  - destruct (step_array_form _ _ H) as [es'' Heq]; subst.
    eapply MS_Step.
    + apply S_ArrayTail; eassumption.
    + apply IHHms; reflexivity.
Qed.

Lemma ms_record_head : forall f e e' fes,
  multi_step e e' ->
  multi_step (ERecord ((f, e) :: fes)) (ERecord ((f, e') :: fes)).
Proof.
  intros. induction H.
  - apply MS_Refl.
  - eapply MS_Step; [apply S_RecordHead; eassumption | assumption].
Qed.

Lemma ms_record_tail : forall f v fes fes',
  is_value v -> multi_step (ERecord fes) (ERecord fes') ->
  multi_step (ERecord ((f, v) :: fes)) (ERecord ((f, v) :: fes')).
Proof.
  intros f v fes fes' Hv Hms.
  remember (ERecord fes) as re. remember (ERecord fes') as re'.
  revert fes fes' Heqre Heqre'.
  induction Hms; intros; subst.
  - injection Heqre'; intros; subst. apply MS_Refl.
  - destruct (step_record_form _ _ H) as [fes'' Heq]; subst.
    eapply MS_Step.
    + apply S_RecordTail; eassumption.
    + apply IHHms; reflexivity.
Qed.

(** ** Helpers for mapped values *)

Lemma forall_val_to_expr_is_value : forall vs,
  Forall is_value (map val_to_expr vs).
Proof.
  induction vs; simpl; constructor.
  - apply val_to_expr_is_value.
  - assumption.
Qed.

Lemma forall_val_to_expr_record_is_value : forall fvs,
  Forall (fun fe : string * expr => is_value (snd fe))
         (map (fun '(f, v) => (f, val_to_expr v)) fvs).
Proof.
  induction fvs as [| [f v] rest IH]; simpl; constructor.
  - simpl. apply val_to_expr_is_value.
  - assumption.
Qed.

(** ** close on ENil is identity *)

Lemma close_nil : forall e, close ENil e = e.
Proof. reflexivity. Qed.

(** ** Subst is compatible with expr_equiv *)

(* We use a Section + Fixpoint to get the guard checker to accept
   structural recursion on the expr_equiv proof. The fix tactic
   doesn't generate proof terms the guard checker accepts for this
   lemma because destruct on expr_equiv introduces aliases that
   obscure the subterm relationship. *)
Section SubstEquiv.
Variable x0 : string.
Variable s1 s2 : expr.
Variable Hs : expr_equiv s1 s2.

Fixpoint subst_equiv (e1 e2 : expr) (He : expr_equiv e1 e2) {struct He} :
  expr_equiv (subst x0 s1 e1) (subst x0 s2 e2).
Proof.
  destruct He; simpl;
    try solve [constructor];
    try solve [constructor; eapply subst_equiv; eassumption];
    try solve [constructor; [eapply subst_equiv; eassumption | eapply subst_equiv; eassumption]];
    try solve [constructor; [eapply subst_equiv; eassumption | eapply subst_equiv; eassumption | eapply subst_equiv; eassumption]].
  (* 7 goals remain: Var, Let, Lam, Fix, Array, Record, Match *)
  - (* Var *) destruct (String.eqb x0 x); [assumption | constructor].
  - (* Let *) constructor; [eapply subst_equiv; eassumption |].
    destruct (String.eqb x0 x); [assumption | eapply subst_equiv; eassumption].
  - (* Lam *) constructor. destruct (String.eqb x0 x); [assumption | eapply subst_equiv; eassumption].
  - (* Fix *) constructor. destruct (String.eqb x0 f || String.eqb x0 x)%bool; [assumption | eapply subst_equiv; eassumption].
  - (* Array *) constructor.
    induction H; simpl.
    + constructor.
    + constructor; [eapply subst_equiv; eassumption | apply IHForall2].
  - (* Record *) constructor.
    induction H; simpl.
    + constructor.
    + destruct x as [f1 e1']. destruct y as [f2 e2']. simpl in *.
      constructor.
      * destruct H as [Hname Hequiv].
        split; [assumption | eapply subst_equiv; eassumption].
      * apply IHForall2.
  - (* Match *)
    constructor; [eapply subst_equiv; eassumption |].
    match goal with HF : Forall2 _ _ _ |- _ => induction HF; simpl end.
    + constructor.
    + destruct x as [[t1 y1] b1]. destruct y as [[t2 y2] b2]. simpl in *.
      match goal with HD : _ /\ _ /\ _ |- _ => destruct HD as [Ht [Hy Hb]] end.
      subst t2. subst y2.
      constructor.
      * split; [reflexivity |]. split; [reflexivity |].
        destruct (String.eqb x0 y1); [exact Hb | eapply subst_equiv; eassumption].
      * apply IHForall2.
Defined.
End SubstEquiv.

Lemma subst_expr_equiv : forall x s1 s2 e1 e2,
  expr_equiv s1 s2 -> expr_equiv e1 e2 ->
  expr_equiv (subst x s1 e1) (subst x s2 e2).
Proof.
  intros. eapply subst_equiv; eassumption.
Qed.

(** ** Helper lemmas for step simulation *)

(** apply_binop only succeeds on base values (EInt/EBool/EString)
    where expr_equiv forces identity. *)
Lemma apply_binop_equiv : forall op v1 v2 r e1' e2',
  apply_binop op v1 v2 = Some r ->
  expr_equiv v1 e1' -> expr_equiv v2 e2' ->
  v1 = e1' /\ v2 = e2'.
Proof.
  intros op v1 v2 r e1' e2' Hab He1 He2.
  destruct op; simpl in Hab;
    repeat match goal with
    | [ H : match ?e with _ => _ end = Some _ |- _ ] => destruct e; try discriminate
    end;
    inversion He1; subst; inversion He2; subst;
    split; reflexivity.
Qed.

Lemma forall_value_transfer : forall es es',
  Forall is_value es -> Forall2 expr_equiv es es' -> Forall is_value es'.
Proof.
  intros es es' Hval HF2.
  revert es' HF2. induction Hval as [| ? ? ? ? IH]; intros es' HF2;
  inversion HF2; subst; constructor;
  [ eapply expr_equiv_is_value; eassumption | apply IH; assumption ].
Qed.

Lemma nth_equiv : forall k es es',
  Forall is_value es -> Forall2 expr_equiv es es' ->
  expr_equiv (nth k es EUnit) (nth k es' EUnit).
Proof.
  intros k es es' Hval HF2.
  revert es' HF2 k. induction Hval as [| ? ? ? ? IH]; intros es' HF2 k;
  inversion HF2; subst.
  - destruct k; simpl; apply EQ_Unit.
  - destruct k; simpl; [assumption | apply IH; assumption].
Qed.

Lemma length_forall2 : forall es es',
  Forall2 expr_equiv es es' -> length es' = length es.
Proof.
  intros es es' HF2. induction HF2; simpl; [reflexivity | f_equal; assumption].
Qed.

Lemma list_update_equiv : forall k v v' es es',
  Forall2 expr_equiv es es' -> expr_equiv v v' ->
  Forall2 expr_equiv (list_update k v es) (list_update k v' es').
Proof.
  intros k v v' es es' HF2 Hv.
  revert k. induction HF2; intros k; destruct k; simpl;
  try constructor; try assumption; try apply IHHF2.
Qed.

Lemma app_forall2_equiv : forall v v' es es',
  Forall2 expr_equiv es es' -> expr_equiv v v' ->
  Forall2 expr_equiv (es ++ [v]) (es' ++ [v']).
Proof.
  intros v v' es es' HF2 Hv.
  induction HF2; simpl.
  - constructor; [assumption | constructor].
  - constructor; [assumption | apply IHHF2].
Qed.

Lemma record_value_transfer : forall (fes fes' : list (string * expr)),
  Forall (fun p => is_value (snd p)) fes ->
  Forall2 (fun fe fe' => fst fe = fst fe' /\ expr_equiv (snd fe) (snd fe')) fes fes' ->
  Forall (fun p => is_value (snd p)) fes'.
Proof.
  intros fes fes' Hval HF2.
  revert fes' HF2. induction Hval as [| [f1 e1] ? ? ? IH]; intros fes' HF2;
  inversion HF2; subst; constructor.
  - match goal with
    | [ Hp : _ /\ expr_equiv _ _ |- _ ] =>
      destruct Hp as [_ Hequiv]; simpl in *; eapply expr_equiv_is_value; eassumption
    end.
  - apply IH; assumption.
Qed.

Lemma record_field_equiv : forall f (fes fes' : list (string * expr)),
  Forall (fun p => is_value (snd p)) fes ->
  Forall2 (fun fe fe' => fst fe = fst fe' /\ expr_equiv (snd fe) (snd fe')) fes fes' ->
  expr_equiv (match assoc_lookup f fes with Some v => v | None => EUnit end)
             (match assoc_lookup f fes' with Some v => v | None => EUnit end).
Proof.
  intros f fes fes' Hval HF2.
  revert fes' HF2. induction Hval as [| [f1 e1] ? ? ? IH]; intros fes' HF2;
  inversion HF2; subst.
  - simpl. apply EQ_Unit.
  - match goal with
    | [ Hp : fst (?f1p, ?e1p) = fst ?y /\ expr_equiv (snd (?f1p, ?e1p)) (snd ?y) |- _ ] =>
      destruct y as [f2 e2]; simpl in Hp; destruct Hp as [Hname Hequiv];
      subst f2; simpl;
      destruct (String.eqb f f1); [assumption | apply IH; assumption]
    end.
Qed.

Lemma find_branch_equiv : forall tag x body branches branches',
  find_branch tag branches = Some (x, body) ->
  Forall2 (fun b b' => fst (fst b) = fst (fst b')
           /\ snd (fst b) = snd (fst b')
           /\ expr_equiv (snd b) (snd b')) branches branches' ->
  exists body', find_branch tag branches' = Some (x, body') /\ expr_equiv body body'.
Proof.
  intros tag x body branches branches' Hfind HF2.
  induction HF2 as [| ? ? ? ? ? IH].
  - simpl in Hfind. discriminate.
  - destruct x0 as [[t1 y1] b1]. destruct y as [[t2 y2] b2].
    simpl in *. destruct H as [Ht [Hy Hb]].
    subst t2 y2. destruct (String.eqb tag t1).
    + injection Hfind; intros; subst. eexists. split; [reflexivity | assumption].
    + apply IHIH. assumption.
Qed.

(** ** Step simulation: expr_equiv expressions step in lockstep.

    The key insight is that the step relation never inspects type
    annotations. S_AppBeta matches on ELam x t body and produces
    subst x v body -- the type t is discarded.

    This proof uses [match goal] to find hypotheses by their types
    rather than auto-generated names, making it robust against
    changes in the [expr_equiv] inductive definition. *)

(** Tactic: use IH by finding a matching expr_equiv hypothesis *)
Local Ltac use_IH :=
  match goal with
  | [ IH : forall erhs0 : expr, expr_equiv ?e erhs0 -> _,
      He : expr_equiv ?e _ |- _ ] =>
    let erhs := fresh "erhs" in
    let Hs := fresh "Hs" in
    let He' := fresh "He" in
    destruct (IH _ He) as [erhs [Hs He']]
  end.

Lemma step_expr_equiv : forall e1 e2,
  expr_equiv e1 e2 -> forall e1',
  step e1 e1' ->
  exists e2', step e2 e2' /\ expr_equiv e1' e2'.
Proof.
  intros e1 erhs Heq e1' Hstep.
  revert erhs Heq.
  induction Hstep; intros erhs Heq; inversion Heq; subst; clear Heq.
  - (* S_BinOp1 *) use_IH.
    eexists. split; [apply S_BinOp1; exact Hs | constructor; assumption].
  - (* S_BinOp2 *) use_IH.
    eexists. split.
    + apply S_BinOp2; [eapply expr_equiv_is_value; eassumption | exact Hs].
    + constructor; assumption.
  - (* S_BinOp *)
    match goal with
    | [ Hab : apply_binop _ ?v1 ?v2 = Some _,
        He1 : expr_equiv ?v1 _,
        He2 : expr_equiv ?v2 _ |- _ ] =>
      let D := fresh in
      pose proof (apply_binop_equiv _ _ _ _ _ _ Hab He1 He2) as D;
      destruct D as [? ?]
    end.
    subst. eexists. split.
    + eapply S_BinOp; eassumption.
    + match goal with
      | [ Hab : apply_binop _ _ _ = Some _ |- _ ] =>
        destruct op; simpl in Hab;
        repeat match goal with
        | [ Hm : match ?e with _ => _ end = Some _ |- _ ] => destruct e; try discriminate
        end;
        injection Hab; intros; subst; apply expr_equiv_refl
      end.
  - (* S_AndFalse *)
    match goal with [ He : expr_equiv (EBool false) _ |- _ ] => inversion He; subst end.
    eexists. split; [apply S_AndFalse | constructor].
  - (* S_OrTrue *)
    match goal with [ He : expr_equiv (EBool true) _ |- _ ] => inversion He; subst end.
    eexists. split; [apply S_OrTrue | constructor].
  - (* S_UnOp1 *) use_IH.
    eexists. split; [apply S_UnOp1; exact Hs | constructor; assumption].
  - (* S_Neg *)
    match goal with [ He : expr_equiv (EInt _) _ |- _ ] => inversion He; subst end.
    eexists. split; [apply S_Neg | constructor].
  - (* S_Not *)
    match goal with [ He : expr_equiv (EBool _) _ |- _ ] => inversion He; subst end.
    eexists. split; [apply S_Not | constructor].
  - (* S_StrLen *)
    match goal with [ He : expr_equiv (EString _) _ |- _ ] => inversion He; subst end.
    eexists. split; [apply S_StrLen | constructor].
  - (* S_IfCond *) use_IH.
    eexists. split; [apply S_IfCond; exact Hs | constructor; assumption].
  - (* S_IfTrue *)
    match goal with [ He : expr_equiv (EBool true) _ |- _ ] => inversion He; subst end.
    eexists. split; [apply S_IfTrue | assumption].
  - (* S_IfFalse *)
    match goal with [ He : expr_equiv (EBool false) _ |- _ ] => inversion He; subst end.
    eexists. split; [apply S_IfFalse | assumption].
  - (* S_Let1 *) use_IH.
    eexists. split; [apply S_Let1; exact Hs | constructor; [assumption | assumption]].
  - (* S_LetVal *)
    eexists. split; [apply S_LetVal; eapply expr_equiv_is_value; eassumption |].
    apply subst_expr_equiv; assumption.
  - (* S_Set1 *) use_IH.
    eexists. split; [apply S_Set1; exact Hs | constructor; assumption].
  - (* S_SetVal *)
    eexists. split; [apply S_SetVal; eapply expr_equiv_is_value; eassumption | constructor].
  - (* S_Seq1 *) use_IH.
    eexists. split; [apply S_Seq1; exact Hs | constructor; assumption].
  - (* S_SeqVal *)
    eexists. split; [apply S_SeqVal; eapply expr_equiv_is_value; eassumption | assumption].
  - (* S_While *)
    eexists. split; [apply S_While |].
    apply EQ_If; [assumption | apply EQ_Seq; [assumption | apply EQ_While; assumption] | apply EQ_Unit].
  - (* S_App1 *) use_IH.
    eexists. split; [apply S_App1; exact Hs | constructor; assumption].
  - (* S_App2 *) use_IH.
    eexists. split; [apply S_App2; [eapply expr_equiv_is_value; eassumption | exact Hs] | constructor; [assumption | assumption]].
  - (* S_AppBeta *)
    match goal with [ He : expr_equiv (ELam _ _ _) _ |- _ ] => inversion He; subst end.
    eexists. split.
    + apply S_AppBeta. eapply expr_equiv_is_value; eassumption.
    + apply subst_expr_equiv; assumption.
  - (* S_AppFixBeta *)
    match goal with [ He : expr_equiv (EFix _ _ _ _ _) _ |- _ ] => inversion He; subst end.
    eexists. split.
    + apply S_AppFixBeta. eapply expr_equiv_is_value; eassumption.
    + apply subst_expr_equiv; [constructor; assumption |].
      apply subst_expr_equiv; assumption.
  - (* S_ArrayHead *)
    match goal with [ HF2 : Forall2 expr_equiv (_ :: _) _ |- _ ] => inversion HF2; subst end.
    use_IH.
    eexists. split; [apply S_ArrayHead; exact Hs |].
    constructor. constructor; [assumption |]. assumption.
  - (* S_ArrayTail *)
    match goal with [ HF2 : Forall2 expr_equiv (_ :: _) _ |- _ ] => inversion HF2; subst end.
    match goal with
    | [ HF2tail : Forall2 expr_equiv es ?es2 |- _ ] =>
      destruct (IHHstep _ (EQ_Array _ _ HF2tail)) as [erhs0 [Hs' He']]
    end.
    inversion He'; subst.
    eexists. split.
    + apply S_ArrayTail; [eapply expr_equiv_is_value; eassumption | exact Hs'].
    + constructor. constructor; assumption.
  - (* S_Index1 *) use_IH.
    eexists. split; [apply S_Index1; exact Hs | constructor; assumption].
  - (* S_Index2 *) use_IH.
    eexists. split; [apply S_Index2; [eapply expr_equiv_is_value; eassumption | exact Hs] | constructor; assumption].
  - (* S_IndexVal *)
    match goal with [ He : expr_equiv (EArray _) _ |- _ ] => inversion He; subst end.
    match goal with [ He : expr_equiv (EInt _) _ |- _ ] => inversion He; subst end.
    eexists. split.
    + apply S_IndexVal. eapply forall_value_transfer; eassumption.
    + apply nth_equiv; assumption.
  - (* S_ArrayLen *)
    match goal with [ He : expr_equiv (EArray _) _ |- _ ] => inversion He; subst end.
    eexists. split.
    + apply S_ArrayLen. eapply forall_value_transfer; eassumption.
    + match goal with
      | [ HF2 : Forall2 expr_equiv _ ?ys |- _ ] =>
        rewrite (length_forall2 _ _ HF2); apply EQ_Int
      end.
  - (* S_ArraySet1 *) use_IH.
    eexists. split; [apply S_ArraySet1; exact Hs | constructor; assumption].
  - (* S_ArraySet2 *) use_IH.
    eexists. split; [apply S_ArraySet2; [eapply expr_equiv_is_value; eassumption | exact Hs] | constructor; assumption].
  - (* S_ArraySet3 *) use_IH.
    eexists. split; [apply S_ArraySet3; [eapply expr_equiv_is_value; eassumption | eapply expr_equiv_is_value; eassumption | exact Hs] | constructor; [assumption | assumption | assumption]].
  - (* S_ArraySetVal *)
    match goal with [ He : expr_equiv (EArray _) _ |- _ ] => inversion He; subst end.
    match goal with [ He : expr_equiv (EInt _) _ |- _ ] => inversion He; subst end.
    eexists. split.
    + apply S_ArraySetVal.
      * eapply forall_value_transfer; eassumption.
      * eapply expr_equiv_is_value; eassumption.
    + constructor. apply list_update_equiv; assumption.
  - (* S_ArrayPush1 *) use_IH.
    eexists. split; [apply S_ArrayPush1; exact Hs | constructor; assumption].
  - (* S_ArrayPush2 *) use_IH.
    eexists. split; [apply S_ArrayPush2; [eapply expr_equiv_is_value; eassumption | exact Hs] | constructor; [assumption | assumption]].
  - (* S_ArrayPushVal *)
    match goal with [ He : expr_equiv (EArray _) _ |- _ ] => inversion He; subst end.
    eexists. split.
    + apply S_ArrayPushVal.
      * eapply forall_value_transfer; eassumption.
      * eapply expr_equiv_is_value; eassumption.
    + constructor. apply app_forall2_equiv; assumption.
  - (* S_RecordHead *)
    match goal with
    | [ HF2 : Forall2 _ ((?f0, ?e0) :: _) _ |- _ ] => inversion HF2; subst
    end.
    match goal with
    | [ Hp : fst _ = fst _ /\ expr_equiv (snd _) (snd _),
        y : string * expr |- _ ] =>
      destruct y as [fy ey]; simpl in Hp; destruct Hp as [Hname Hequiv]
    end.
    destruct (IHHstep _ Hequiv) as [erhs0 [Hs' He']].
    eexists. split; [apply S_RecordHead; exact Hs' |].
    constructor. constructor; [simpl; split; [assumption | assumption] | assumption].
  - (* S_RecordTail *)
    match goal with
    | [ HF2 : Forall2 _ ((?f0, ?e0) :: _) _ |- _ ] => inversion HF2; subst
    end.
    match goal with
    | [ Hp : fst _ = fst _ /\ expr_equiv (snd _) (snd _),
        y : string * expr |- _ ] =>
      destruct y as [fy ey]; simpl in Hp; destruct Hp as [Hname Hequiv]
    end.
    match goal with
    | [ HF2tail : Forall2 _ fes ?fes2 |- _ ] =>
      destruct (IHHstep _ (EQ_Record _ _ HF2tail)) as [erhs0 [Hs' He']]
    end.
    inversion He'; subst.
    eexists. split.
    + apply S_RecordTail; [simpl; eapply expr_equiv_is_value; eassumption | exact Hs'].
    + constructor. constructor; [simpl; split; [reflexivity | assumption] | assumption].
  - (* S_Field1 *) use_IH.
    eexists. split; [apply S_Field1; exact Hs | constructor; assumption].
  - (* S_FieldVal *)
    match goal with [ He : expr_equiv (ERecord _) _ |- _ ] => inversion He; subst end.
    eexists. split.
    + apply S_FieldVal. eapply record_value_transfer; eassumption.
    + apply record_field_equiv; assumption.
  - (* S_SetField1 *) use_IH.
    eexists. split; [apply S_SetField1; exact Hs | constructor; assumption].
  - (* S_SetFieldVal *)
    eexists. split; [apply S_SetFieldVal; eapply expr_equiv_is_value; eassumption | constructor].
  - (* S_Construct1 *) use_IH.
    eexists. split; [apply S_Construct1; exact Hs | constructor; assumption].
  - (* S_Match1 *) use_IH.
    eexists. split; [apply S_Match1; exact Hs | constructor; assumption].
  - (* S_MatchBeta *)
    match goal with [ He : expr_equiv (EConstruct _ _ _) _ |- _ ] => inversion He; subst end.
    match goal with
    | [ Hfind : find_branch _ _ = Some _,
        HF2 : Forall2 _ branches ?br2 |- _ ] =>
      destruct (find_branch_equiv _ _ _ _ _ Hfind HF2) as [body' [Hfind' Hbody']]
    end.
    eexists. split.
    + apply S_MatchBeta; [eapply expr_equiv_is_value; eassumption | exact Hfind'].
    + apply subst_expr_equiv; assumption.
  - (* S_StrIndex1 *) use_IH.
    eexists. split; [apply S_StrIndex1; exact Hs | constructor; assumption].
  - (* S_StrIndex2 *) use_IH.
    eexists. split; [apply S_StrIndex2; [eapply expr_equiv_is_value; eassumption | exact Hs] | constructor; assumption].
  - (* S_StrIndexVal *)
    match goal with [ He : expr_equiv (EString _) _ |- _ ] => inversion He; subst end.
    match goal with [ He : expr_equiv (EInt _) _ |- _ ] => inversion He; subst end.
    eexists. split; [apply S_StrIndexVal | constructor].
Qed.

(** ** Multi-step simulation *)

Lemma multi_step_expr_equiv : forall e1 e2 e3,
  expr_equiv e1 e2 -> multi_step e1 e3 ->
  exists e3', multi_step e2 e3' /\ expr_equiv e3 e3'.
Proof.
  intros e1 e2 e3 Heq Hms.
  revert e2 Heq.
  induction Hms; intros erhs Heq.
  - exists erhs. split; [apply MS_Refl | assumption].
  - destruct (step_expr_equiv _ _ Heq _ H) as [e2' [Hs' He']].
    destruct (IHHms _ He') as [e3' [Hms' He3']].
    exists e3'. split; [eapply MS_Step; eassumption | assumption].
Qed.

(** ** multi_step_equiv transitivity *)

Lemma multi_step_equiv_trans : forall e1 e2 e3,
  multi_step_equiv e1 e2 -> multi_step_equiv e2 e3 ->
  multi_step_equiv e1 e3.
Proof.
  intros e1 e2 e3 [e2' [Hms12 Heq12]] [e3' [Hms23 Heq23]].
  (* We have: e1 -->* e2' ≡ e2 -->* e3' ≡ e3 *)
  (* Need:     e1 -->* e3'' ≡ e3 *)
  (* Requires expr_equiv_sym to flip Heq12, then simulate Hms23 *)
  admit.
Admitted.

(** ** multi_step implies multi_step_equiv *)

Lemma multi_step_to_equiv : forall e1 e2,
  multi_step e1 e2 -> multi_step_equiv e1 e2.
Proof.
  intros. exists e2. split; [assumption | apply expr_equiv_refl].
Qed.

(** ** eval_arith_binop correspondence with apply_binop *)

Lemma arith_binop_apply : forall op n1 n2 v,
  eval_arith_binop op n1 n2 = Some v ->
  apply_binop op (EInt n1) (EInt n2) = Some (val_to_expr v).
Proof.
  intros op n1 n2 v H.
  destruct op; simpl in H; try discriminate.
  - (* OpAdd *) injection H as H; subst; reflexivity.
  - (* OpSub *) injection H as H; subst; reflexivity.
  - (* OpMul *) injection H as H; subst; reflexivity.
  - (* OpDiv *)
    destruct (Z.eqb n2 0) eqn:Hz; [discriminate |].
    injection H as H; subst; simpl; rewrite Hz; reflexivity.
  - (* OpMod *)
    destruct (Z.eqb n2 0) eqn:Hz; [discriminate |].
    injection H as H; subst; simpl; rewrite Hz; reflexivity.
Qed.

(** ** eval_cmp_binop correspondence with apply_binop *)

Lemma cmp_binop_apply : forall op n1 n2 v,
  eval_cmp_binop op n1 n2 = Some v ->
  apply_binop op (EInt n1) (EInt n2) = Some (val_to_expr v).
Proof.
  intros op n1 n2 v H.
  destruct op; simpl in *; try discriminate; injection H; intros; subst; reflexivity.
Qed.

(** ** Inversion: expr_equiv to lambda form *)

Lemma expr_equiv_lam_inv : forall e x t body,
  expr_equiv e (ELam x t body) ->
  exists t' body', e = ELam x t' body' /\ expr_equiv body' body.
Proof.
  intros e x t body H. inversion H; subst.
  eexists _, _. split; [reflexivity | assumption].
Qed.

Lemma expr_equiv_fix_inv : forall e f x t1 t2 body,
  expr_equiv e (EFix f x t1 t2 body) ->
  exists t1' t2' body', e = EFix f x t1' t2' body' /\ expr_equiv body' body.
Proof.
  intros e f x t1 t2 body H. inversion H; subst.
  eexists _, _, _. split; [reflexivity | assumption].
Qed.

(** ** Forward direction: eval ENil e ENil v → multi_step_equiv e (val_to_expr v)

    We prove this by induction on the eval derivation.
    The restriction to ENil avoids all close/subst commutation issues. *)

Theorem eval_to_multistep : forall e v,
  pure e ->
  eval ENil e ENil v ->
  multi_step_equiv e (val_to_expr v).
Proof.
  (* TODO: requires generalized induction over eval with renv = ENil equation.
     The proof is blocked on E_Let/E_App/E_Match cases that need a
     more general env theorem (not restricted to ENil). *)
  admit.
Admitted.

(* Original proof sketch preserved as comment for reference:
   intros e v Hpure Heval.
   cut (forall renv renv' e v, eval renv e renv' v -> pure e -> renv = ENil ->
        multi_step_equiv e (val_to_expr v)).
   Then induction on eval, with pure_env_unchanged to establish
   intermediate env = ENil for IH calls. *)

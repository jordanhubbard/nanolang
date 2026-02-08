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
  | VClos x body cenv => ELam x TUnit (close_except x cenv body)
  | VFixClos f x body cenv => EFix f x TUnit TUnit (close_except2 f x cenv body)
  | VArray vs => EArray (map val_to_expr vs)
  | VRecord fvs => ERecord (map (fun '(f, v) => (f, val_to_expr v)) fvs)
  | VConstruct tag v => EConstruct tag (val_to_expr v) TUnit
  end

with close (renv : env) (e : expr) : expr :=
  match renv with
  | ENil => e
  | ECons x v rest => close rest (subst x (val_to_expr v) e)
  end

with close_except (z : string) (renv : env) (e : expr) : expr :=
  match renv with
  | ENil => e
  | ECons x v rest =>
      if String.eqb x z then close_except z rest e
      else close_except z rest (subst x (val_to_expr v) e)
  end

with close_except2 (z1 z2 : string) (renv : env) (e : expr) : expr :=
  match renv with
  | ENil => e
  | ECons x v rest =>
      if (String.eqb x z1 || String.eqb x z2)%bool
      then close_except2 z1 z2 rest e
      else close_except2 z1 z2 rest (subst x (val_to_expr v) e)
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

(* forall_value_transfer_rev is defined later, after expr_equiv_sym *)

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

Lemma nth_error_map_nth : forall (A B : Type) (f : A -> B) (l : list A) (k : nat) (a : A) (d : B),
  nth_error l k = Some a -> nth k (map f l) d = f a.
Proof.
  intros A B f l. induction l as [| x xs IH]; intros k a d Hnth.
  - destruct k; discriminate.
  - destruct k; simpl in *.
    + injection Hnth; intros; subst. reflexivity.
    + apply IH. assumption.
Qed.

Lemma Forall_nth_error : forall (A : Type) (P : A -> Prop) (l : list A) (k : nat) (x : A),
  Forall P l -> nth_error l k = Some x -> P x.
Proof.
  intros A P l. induction l as [| a l' IH]; intros k x Hfa Hnth.
  - destruct k; simpl in Hnth; discriminate.
  - inversion Hfa; subst. destruct k; simpl in Hnth.
    + injection Hnth; intros; subst. assumption.
    + eapply IH; eassumption.
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

Lemma map_val_to_expr_list_update : forall k v vs,
  map val_to_expr (list_update k v vs) = list_update k (val_to_expr v) (map val_to_expr vs).
Proof.
  intros k v vs. revert k. induction vs as [| w vs' IH]; intros k; destruct k; simpl;
  try reflexivity.
  - f_equal. apply IH.
Qed.

Lemma list_update_forall : forall (A : Type) (P : A -> Prop) (k : nat) (v : A) (l : list A),
  Forall P l -> P v -> Forall P (list_update k v l).
Proof.
  intros A P k v l Hfa Hv. revert k.
  induction Hfa; intros k; destruct k; simpl;
  [constructor | constructor | constructor; assumption | constructor; [assumption | apply IHHfa]].
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

(* record_value_transfer_rev is defined later, after expr_equiv_sym *)

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

Lemma assoc_lookup_map_val_to_expr : forall f fvs v,
  assoc_lookup f fvs = Some v ->
  assoc_lookup f (map (fun '(fn, vn) => (fn, val_to_expr vn)) fvs) = Some (val_to_expr v).
Proof.
  intros f fvs. induction fvs as [| [fn vn] fvs' IH]; intros v Hlookup; simpl in *.
  - discriminate.
  - destruct (String.eqb f fn) eqn:Heq.
    + injection Hlookup; intros; subst. reflexivity.
    + apply IH. assumption.
Qed.

Lemma Forall_assoc_lookup : forall (A : Type) (P : A -> Prop) (f : string) (fvs : list (string * A)) (v : A),
  Forall (fun fv => P (snd fv)) fvs -> assoc_lookup f fvs = Some v -> P v.
Proof.
  intros A P f fvs v Hfa Hlookup.
  induction Hfa as [| [fn vn] fvs' Hvn Hfa' IH]; simpl in Hlookup.
  - discriminate.
  - simpl in Hvn. destruct (String.eqb f fn).
    + injection Hlookup; intros; subst. exact Hvn.
    + apply IH. exact Hlookup.
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

(** ** expr_equiv symmetry — proved via Fixpoint for guard checker *)

Fixpoint expr_equiv_sym (e1 e2 : expr) (H : expr_equiv e1 e2) {struct H} :
  expr_equiv e2 e1 :=
  match H in expr_equiv e1' e2' return expr_equiv e2' e1' with
  | EQ_Int n => EQ_Int n
  | EQ_Bool b => EQ_Bool b
  | EQ_String s => EQ_String s
  | EQ_Unit => EQ_Unit
  | EQ_Var x => EQ_Var x
  | EQ_BinOp op _ _ _ _ h1 h2 => EQ_BinOp op _ _ _ _ (expr_equiv_sym _ _ h1) (expr_equiv_sym _ _ h2)
  | EQ_UnOp op _ _ h => EQ_UnOp op _ _ (expr_equiv_sym _ _ h)
  | EQ_If _ _ _ _ _ _ h1 h2 h3 => EQ_If _ _ _ _ _ _ (expr_equiv_sym _ _ h1) (expr_equiv_sym _ _ h2) (expr_equiv_sym _ _ h3)
  | EQ_Let x _ _ _ _ h1 h2 => EQ_Let x _ _ _ _ (expr_equiv_sym _ _ h1) (expr_equiv_sym _ _ h2)
  | EQ_Set x _ _ h => EQ_Set x _ _ (expr_equiv_sym _ _ h)
  | EQ_Seq _ _ _ _ h1 h2 => EQ_Seq _ _ _ _ (expr_equiv_sym _ _ h1) (expr_equiv_sym _ _ h2)
  | EQ_While _ _ _ _ h1 h2 => EQ_While _ _ _ _ (expr_equiv_sym _ _ h1) (expr_equiv_sym _ _ h2)
  | EQ_Lam x _ _ _ _ h => EQ_Lam x _ _ _ _ (expr_equiv_sym _ _ h)
  | EQ_App _ _ _ _ h1 h2 => EQ_App _ _ _ _ (expr_equiv_sym _ _ h1) (expr_equiv_sym _ _ h2)
  | EQ_Fix f x _ _ _ _ _ _ h => EQ_Fix f x _ _ _ _ _ _ (expr_equiv_sym _ _ h)
  | EQ_Array _ _ hf =>
      EQ_Array _ _ ((fix go es1 es2 (hf : Forall2 expr_equiv es1 es2) :
                       Forall2 expr_equiv es2 es1 :=
                       match hf with
                       | Forall2_nil _ => Forall2_nil _
                       | Forall2_cons _ _ h ht => Forall2_cons _ _ (expr_equiv_sym _ _ h) (go _ _ ht)
                       end) _ _ hf)
  | EQ_Index _ _ _ _ h1 h2 => EQ_Index _ _ _ _ (expr_equiv_sym _ _ h1) (expr_equiv_sym _ _ h2)
  | EQ_ArraySet _ _ _ _ _ _ h1 h2 h3 => EQ_ArraySet _ _ _ _ _ _ (expr_equiv_sym _ _ h1) (expr_equiv_sym _ _ h2) (expr_equiv_sym _ _ h3)
  | EQ_ArrayPush _ _ _ _ h1 h2 => EQ_ArrayPush _ _ _ _ (expr_equiv_sym _ _ h1) (expr_equiv_sym _ _ h2)
  | @EQ_Record fes fes' hf =>
      @EQ_Record fes' fes
        ((fix go (l1 l2 : list (string * expr))
              (hf : Forall2 (fun fe fe' => fst fe = fst fe' /\ expr_equiv (snd fe) (snd fe')) l1 l2) :
              Forall2 (fun fe fe' => fst fe = fst fe' /\ expr_equiv (snd fe) (snd fe')) l2 l1 :=
            match hf with
            | Forall2_nil _ => Forall2_nil _
            | @Forall2_cons _ _ _ a b _ _ (conj heq hee) ht =>
                @Forall2_cons _ _ _ b a _ _ (conj (eq_sym heq) (expr_equiv_sym _ _ hee)) (go _ _ ht)
            end) _ _ hf)
  | EQ_Field _ _ f h => EQ_Field _ _ f (expr_equiv_sym _ _ h)
  | EQ_SetField x f _ _ h => EQ_SetField x f _ _ (expr_equiv_sym _ _ h)
  | EQ_Construct tag _ _ _ _ h => EQ_Construct tag _ _ _ _ (expr_equiv_sym _ _ h)
  | @EQ_Match e1' e2' _ _ he hf =>
      @EQ_Match e2' e1' _ _ (expr_equiv_sym _ _ he)
        ((fix go (l1 l2 : list (string * string * expr))
              (hf : Forall2 (fun b b' => fst (fst b) = fst (fst b') /\ snd (fst b) = snd (fst b') /\ expr_equiv (snd b) (snd b')) l1 l2) :
              Forall2 (fun b b' => fst (fst b) = fst (fst b') /\ snd (fst b) = snd (fst b') /\ expr_equiv (snd b) (snd b')) l2 l1 :=
            match hf with
            | Forall2_nil _ => Forall2_nil _
            | @Forall2_cons _ _ _ a b _ _ (conj ht (conj hy hb)) ht' =>
                @Forall2_cons _ _ _ b a _ _ (conj (eq_sym ht) (conj (eq_sym hy) (expr_equiv_sym _ _ hb))) (go _ _ ht')
            end) _ _ hf)
  | EQ_StrIndex _ _ _ _ h1 h2 => EQ_StrIndex _ _ _ _ (expr_equiv_sym _ _ h1) (expr_equiv_sym _ _ h2)
  end.

(** ** Reverse value transfer lemmas (need expr_equiv_sym) *)

Lemma forall2_expr_equiv_sym : forall es es',
  Forall2 expr_equiv es es' -> Forall2 expr_equiv es' es.
Proof.
  intros es es' HF2. induction HF2; constructor.
  - apply expr_equiv_sym. assumption.
  - assumption.
Qed.

Lemma forall_value_transfer_rev : forall es es',
  Forall is_value es' -> Forall2 expr_equiv es es' -> Forall is_value es.
Proof.
  intros es es' Hval HF2.
  eapply forall_value_transfer.
  - exact Hval.
  - apply forall2_expr_equiv_sym. exact HF2.
Qed.

Lemma forall2_record_equiv_sym : forall (fes fes' : list (string * expr)),
  Forall2 (fun fe fe' => fst fe = fst fe' /\ expr_equiv (snd fe) (snd fe')) fes fes' ->
  Forall2 (fun fe fe' => fst fe = fst fe' /\ expr_equiv (snd fe) (snd fe')) fes' fes.
Proof.
  intros fes fes' HF2.
  induction HF2 as [| [f1 e1] [f2 e2] ? ? [Hfst Hequiv] ? IH]; constructor.
  - split; [symmetry; exact Hfst | apply expr_equiv_sym; exact Hequiv].
  - exact IH.
Qed.

Lemma record_value_transfer_rev : forall (fes fes' : list (string * expr)),
  Forall (fun p => is_value (snd p)) fes' ->
  Forall2 (fun fe fe' => fst fe = fst fe' /\ expr_equiv (snd fe) (snd fe')) fes fes' ->
  Forall (fun p => is_value (snd p)) fes.
Proof.
  intros fes fes' Hval HF2.
  eapply record_value_transfer.
  - exact Hval.
  - apply forall2_record_equiv_sym. exact HF2.
Qed.

(** ** expr_equiv transitivity *)

Section ExprEquivTrans.
Fixpoint expr_equiv_trans (e1 e2 e3 : expr)
  (H12 : expr_equiv e1 e2) {struct H12} :
  expr_equiv e2 e3 -> expr_equiv e1 e3.
Proof.
  intro H23.
  destruct H12; inversion H23; subst;
    try solve [constructor];
    try solve [constructor; eapply expr_equiv_trans; eassumption];
    try solve [constructor; [eapply expr_equiv_trans; eassumption |
                              eapply expr_equiv_trans; eassumption]];
    try solve [constructor; [eapply expr_equiv_trans; eassumption |
                              eapply expr_equiv_trans; eassumption |
                              eapply expr_equiv_trans; eassumption]].
  - (* Array *)
    constructor. clear H23.
    match goal with
    | HF12 : Forall2 expr_equiv ?es1 ?es2,
      HF23 : Forall2 expr_equiv ?es2 ?es3 |- _ =>
      revert es3 HF23; induction HF12; intros es3 HF23; inversion HF23; subst;
      [constructor | constructor; [eapply expr_equiv_trans; eassumption | auto]]
    end.
  - (* Record *)
    constructor. clear H23.
    match goal with
    | HF12 : Forall2 _ ?fes1 ?fes2,
      HF23 : Forall2 _ ?fes2 ?fes3 |- _ =>
      revert fes3 HF23; induction HF12; intros fes3 HF23; inversion HF23; subst
    end.
    + constructor.
    + constructor.
      * match goal with
        | [ a : string * expr, b : string * expr, c : string * expr,
            H1 : fst ?a = fst ?b /\ _, H2 : fst ?b = fst ?c /\ _ |- _ ] =>
          destruct a, b, c; simpl in *;
          destruct H1 as [Hf1 He1]; destruct H2 as [Hf2 He2];
          split; [congruence | eapply expr_equiv_trans; eassumption]
        end.
      * match goal with IH : forall _, Forall2 _ _ _ -> _ |- _ => apply IH; assumption end.
  - (* Match *)
    constructor; [eapply expr_equiv_trans; eassumption |].
    clear H23.
    match goal with
    | HF12 : Forall2 _ ?br1 ?br2,
      HF23 : Forall2 _ ?br2 ?br3 |- _ =>
      revert br3 HF23; induction HF12; intros br3 HF23; inversion HF23; subst
    end.
    + constructor.
    + constructor.
      * match goal with
        | [ a : string * string * expr, b : string * string * expr, c : string * string * expr,
            H1 : _ /\ _ /\ expr_equiv (snd ?a) (snd ?b),
            H2 : _ /\ _ /\ expr_equiv (snd ?b) (snd ?c) |- _ ] =>
          destruct a as [[? ?] ?]; destruct b as [[? ?] ?]; destruct c as [[? ?] ?]; simpl in *;
          destruct H1 as [Ht1 [Hy1 Hb1]]; destruct H2 as [Ht2 [Hy2 Hb2]];
          split; [congruence | split; [congruence | eapply expr_equiv_trans; eassumption]]
        end.
      * match goal with IH : forall _, Forall2 _ _ _ -> _ |- _ => apply IH; assumption end.
Defined.
End ExprEquivTrans.

(** ** multi_step_equiv transitivity *)

Lemma multi_step_equiv_trans : forall e1 e2 e3,
  multi_step_equiv e1 e2 -> multi_step_equiv e2 e3 ->
  multi_step_equiv e1 e3.
Proof.
  intros e1 e2 e3 [e2' [Hms12 Heq12]] [e3' [Hms23 Heq23]].
  (* e1 -->* e2' ≡ e2 -->* e3' ≡ e3 *)
  (* Flip Heq12 to get e2 ≡ e2', simulate Hms23 starting from e2' *)
  pose proof (expr_equiv_sym _ _ Heq12) as Heq21.
  destruct (multi_step_expr_equiv _ _ _ Heq21 Hms23) as [e3'' [Hms23' Heq3']].
  exists e3''. split.
  - eapply multi_step_trans; eassumption.
  - eapply expr_equiv_trans.
    + apply expr_equiv_sym. exact Heq3'.
    + exact Heq23.
Qed.

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

(** ** Result type lemmas for eval operations *)

Lemma eval_arith_binop_int : forall op n1 n2 v,
  eval_arith_binop op n1 n2 = Some v -> exists n, v = VInt n.
Proof.
  intros op n1 n2 v H.
  destruct op; simpl in H; try discriminate;
    try (injection H; intros; subst; eexists; reflexivity).
  - destruct (Z.eqb n2 0); [discriminate | injection H; intros; subst; eexists; reflexivity].
  - destruct (Z.eqb n2 0); [discriminate | injection H; intros; subst; eexists; reflexivity].
Qed.

Lemma eval_cmp_binop_bool : forall op n1 n2 v,
  eval_cmp_binop op n1 n2 = Some v -> exists b, v = VBool b.
Proof.
  intros op n1 n2 v H.
  destruct op; simpl in H; try discriminate; injection H; intros; subst; eexists; reflexivity.
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

(** ** Closedness and substitution infrastructure *)

Definition eclosed (e : expr) : Prop := forall x s, subst x s e = e.

Fixpoint all_vals_closed (renv : env) : Prop :=
  match renv with
  | ENil => True
  | ECons _ v rest => eclosed (val_to_expr v) /\ all_vals_closed rest
  end.

Lemma eqb_refl : forall s, String.eqb s s = true.
Proof.
  intros. rewrite String.eqb_eq. reflexivity.
Qed.

Lemma eqb_neq : forall s1 s2, String.eqb s1 s2 = false -> s1 <> s2.
Proof.
  intros s1 s2 H Heq. subst. rewrite eqb_refl in H. discriminate.
Qed.

Lemma eqb_sym : forall s1 s2, String.eqb s1 s2 = String.eqb s2 s1.
Proof.
  intros. destruct (String.eqb s1 s2) eqn:H1.
  - apply String.eqb_eq in H1. subst. symmetry. apply eqb_refl.
  - destruct (String.eqb s2 s1) eqn:H2; auto.
    apply String.eqb_eq in H2. subst. rewrite eqb_refl in H1. discriminate.
Qed.

(** ** Strong induction principle for expr, handling nested list types *)
Fixpoint expr_strong_ind
  (P : expr -> Prop)
  (P_int : forall n, P (EInt n))
  (P_bool : forall b, P (EBool b))
  (P_string : forall s, P (EString s))
  (P_unit : P EUnit)
  (P_var : forall s, P (EVar s))
  (P_binop : forall op e1 e2, P e1 -> P e2 -> P (EBinOp op e1 e2))
  (P_unop : forall op e, P e -> P (EUnOp op e))
  (P_if : forall e1 e2 e3, P e1 -> P e2 -> P e3 -> P (EIf e1 e2 e3))
  (P_let : forall x e1 e2, P e1 -> P e2 -> P (ELet x e1 e2))
  (P_set : forall x e, P e -> P (ESet x e))
  (P_seq : forall e1 e2, P e1 -> P e2 -> P (ESeq e1 e2))
  (P_while : forall e1 e2, P e1 -> P e2 -> P (EWhile e1 e2))
  (P_lam : forall x t e, P e -> P (ELam x t e))
  (P_app : forall e1 e2, P e1 -> P e2 -> P (EApp e1 e2))
  (P_fix : forall f x t1 t2 e, P e -> P (EFix f x t1 t2 e))
  (P_array : forall l, (forall e, In e l -> P e) -> P (EArray l))
  (P_index : forall e1 e2, P e1 -> P e2 -> P (EIndex e1 e2))
  (P_arrayset : forall e1 e2 e3, P e1 -> P e2 -> P e3 -> P (EArraySet e1 e2 e3))
  (P_arraypush : forall e1 e2, P e1 -> P e2 -> P (EArrayPush e1 e2))
  (P_record : forall l, (forall e, In e (map snd l) -> P e) -> P (ERecord l))
  (P_field : forall e f, P e -> P (EField e f))
  (P_setfield : forall x f e, P e -> P (ESetField x f e))
  (P_construct : forall tag e t, P e -> P (EConstruct tag e t))
  (P_match : forall e branches,
    P e -> (forall b, In b branches -> P (snd b)) -> P (EMatch e branches))
  (P_strindex : forall e1 e2, P e1 -> P e2 -> P (EStrIndex e1 e2))
  (e : expr) {struct e} : P e :=
  let fix list_ind (l : list expr) : forall e, In e l -> P e :=
    match l return forall e, In e l -> P e with
    | [] => fun _ H => match H with end
    | h :: t => fun e' Hin =>
        match Hin with
        | or_introl Heq =>
            match Heq in (_ = e'') return P e'' with
            | eq_refl => expr_strong_ind P P_int P_bool P_string P_unit P_var
                P_binop P_unop P_if P_let P_set P_seq P_while P_lam P_app P_fix
                P_array P_index P_arrayset P_arraypush P_record P_field P_setfield
                P_construct P_match P_strindex h
            end
        | or_intror Hin' => list_ind t e' Hin'
        end
    end in
  let fix pair_list_ind (l : list (string * expr)) :
      forall e, In e (map snd l) -> P e :=
    match l return forall e, In e (map snd l) -> P e with
    | [] => fun _ H => match H with end
    | (_, h) :: t => fun e' Hin =>
        match Hin with
        | or_introl Heq =>
            match Heq in (_ = e'') return P e'' with
            | eq_refl => expr_strong_ind P P_int P_bool P_string P_unit P_var
                P_binop P_unop P_if P_let P_set P_seq P_while P_lam P_app P_fix
                P_array P_index P_arrayset P_arraypush P_record P_field P_setfield
                P_construct P_match P_strindex h
            end
        | or_intror Hin' => pair_list_ind t e' Hin'
        end
    end in
  let fix branch_list_ind (l : list (string * string * expr)) :
      forall b, In b l -> P (snd b) :=
    match l return forall b, In b l -> P (snd b) with
    | [] => fun _ H => match H with end
    | ((_, _), h) :: t => fun b' Hin =>
        match Hin with
        | or_introl Heq =>
            match Heq in (_ = b'') return P (snd b'') with
            | eq_refl => expr_strong_ind P P_int P_bool P_string P_unit P_var
                P_binop P_unop P_if P_let P_set P_seq P_while P_lam P_app P_fix
                P_array P_index P_arrayset P_arraypush P_record P_field P_setfield
                P_construct P_match P_strindex h
            end
        | or_intror Hin' => branch_list_ind t b' Hin'
        end
    end in
  let rec_ := expr_strong_ind P P_int P_bool P_string P_unit P_var
      P_binop P_unop P_if P_let P_set P_seq P_while P_lam P_app P_fix
      P_array P_index P_arrayset P_arraypush P_record P_field P_setfield
      P_construct P_match P_strindex in
  match e with
  | EInt n => P_int n
  | EBool b => P_bool b
  | EString s => P_string s
  | EUnit => P_unit
  | EVar s => P_var s
  | EBinOp op e1 e2 => P_binop op e1 e2 (rec_ e1) (rec_ e2)
  | EUnOp op e1 => P_unop op e1 (rec_ e1)
  | EIf e1 e2 e3 => P_if e1 e2 e3 (rec_ e1) (rec_ e2) (rec_ e3)
  | ELet x e1 e2 => P_let x e1 e2 (rec_ e1) (rec_ e2)
  | ESet x e1 => P_set x e1 (rec_ e1)
  | ESeq e1 e2 => P_seq e1 e2 (rec_ e1) (rec_ e2)
  | EWhile e1 e2 => P_while e1 e2 (rec_ e1) (rec_ e2)
  | ELam x t e1 => P_lam x t e1 (rec_ e1)
  | EApp e1 e2 => P_app e1 e2 (rec_ e1) (rec_ e2)
  | EFix f x t1 t2 e1 => P_fix f x t1 t2 e1 (rec_ e1)
  | EArray l => P_array l (list_ind l)
  | EIndex e1 e2 => P_index e1 e2 (rec_ e1) (rec_ e2)
  | EArraySet e1 e2 e3 => P_arrayset e1 e2 e3 (rec_ e1) (rec_ e2) (rec_ e3)
  | EArrayPush e1 e2 => P_arraypush e1 e2 (rec_ e1) (rec_ e2)
  | ERecord l => P_record l (pair_list_ind l)
  | EField e1 f => P_field e1 f (rec_ e1)
  | ESetField x f e1 => P_setfield x f e1 (rec_ e1)
  | EConstruct tag e1 t => P_construct tag e1 t (rec_ e1)
  | EMatch e1 branches => P_match e1 branches (rec_ e1) (branch_list_ind branches)
  | EStrIndex e1 e2 => P_strindex e1 e2 (rec_ e1) (rec_ e2)
  end.

(** Substitution commutativity for closed expressions (requires x <> y) *)
Lemma subst_comm_closed : forall x y sx sy e,
  x <> y ->
  eclosed sx -> eclosed sy ->
  subst x sx (subst y sy e) = subst y sy (subst x sx e).
Proof.
  intros x y sx sy e Hneq Hcx Hcy.
  revert e.
  fix IH 1. intro e. destruct e; simpl;
    try reflexivity;
    try (f_equal; try reflexivity; apply IH).
  - (* EVar *)
    destruct (String.eqb y s) eqn:Hys; destruct (String.eqb x s) eqn:Hxs; simpl;
      rewrite ?Hys, ?Hxs; try reflexivity.
    + exfalso. apply Hneq.
      apply String.eqb_eq in Hxs. apply String.eqb_eq in Hys. congruence.
    + apply Hcy.
    + symmetry. apply Hcx.
  - (* ELet *)
    f_equal; try apply IH.
    destruct (String.eqb y s) eqn:Hys; destruct (String.eqb x s) eqn:Hxs; simpl;
      rewrite ?Hys, ?Hxs; try reflexivity; try apply IH.
  - (* ELam *)
    f_equal; try reflexivity.
    destruct (String.eqb y s) eqn:Hys; destruct (String.eqb x s) eqn:Hxs; simpl;
      rewrite ?Hys, ?Hxs; try reflexivity; try apply IH.
  - (* EFix *)
    f_equal; try reflexivity.
    destruct (String.eqb y s || String.eqb y s0)%bool eqn:Hyf;
    destruct (String.eqb x s || String.eqb x s0)%bool eqn:Hxf; simpl;
      rewrite ?Hyf, ?Hxf; try reflexivity; try apply IH.
  - (* EArray *)
    f_equal.
    induction l; simpl; [reflexivity | f_equal; [apply IH | assumption]].
  - (* ERecord *)
    f_equal.
    induction l as [|[? ?] ?]; simpl; [reflexivity | f_equal; [f_equal; apply IH | assumption]].
  - (* EMatch *)
    f_equal; [apply IH |].
    induction l as [|[[t0 z] b] rest IHl]; simpl; [reflexivity |].
    f_equal; [|assumption].
    f_equal.
    destruct (String.eqb y z) eqn:Hyz; destruct (String.eqb x z) eqn:Hxz; simpl;
      rewrite ?Hyz, ?Hxz; try reflexivity; try apply IH.
Qed.

(** Substituting same variable twice *)
Lemma subst_subst_same : forall x s1 s2 e,
  subst x s1 (subst x s2 e) = subst x (subst x s1 s2) e.
Proof.
  intros x s1 s2 e.
  induction e using expr_strong_ind; simpl;
    try reflexivity;
    try (f_equal; auto; fail).
  - (* EVar *)
    destruct (String.eqb x s) eqn:Hxs; simpl.
    + reflexivity.
    + simpl. rewrite Hxs. reflexivity.
  - (* ELet *)
    f_equal; [auto |].
    destruct (String.eqb x x0) eqn:Hxs.
    + reflexivity.
    + auto.
  - (* ELam *)
    f_equal.
    destruct (String.eqb x x0) eqn:Hxs.
    + reflexivity.
    + auto.
  - (* EFix *)
    f_equal.
    destruct (String.eqb x f || String.eqb x x0)%bool eqn:Hxs.
    + reflexivity.
    + auto.
  - (* EArray *)
    f_equal.
    induction l as [|e0 rest IHl']; simpl; [reflexivity|].
    f_equal; [apply H; left; reflexivity |].
    apply IHl'. intros. apply H. right. assumption.
  - (* ERecord *)
    f_equal.
    induction l as [|[fn e0] rest IHl']; simpl; [reflexivity|].
    f_equal; [f_equal; apply H; simpl; left; reflexivity |].
    apply IHl'. intros. apply H. simpl. right. assumption.
  - (* EMatch *)
    match goal with
    | H : forall b, In b ?l -> _ |- _ => rename H into IHbrs
    end.
    f_equal; [auto |].
    induction branches as [|[[t0 z] b] rest IHl']; simpl; [reflexivity|].
    f_equal.
    + f_equal.
      destruct (String.eqb x z) eqn:Hxz.
      * reflexivity.
      * apply (IHbrs (t0, z, b)). left. reflexivity.
    + apply IHl'. intros b0 Hin. apply IHbrs. right. assumption.
Qed.

(** Substituting into a closed expression is identity *)
Lemma subst_eclosed : forall x s e, eclosed e -> subst x s e = e.
Proof.
  intros x s e Hcl. apply Hcl.
Qed.

(** ** val_good / env_good — well-formedness invariants *)

Inductive val_good : val -> Prop :=
  | VG_Int : forall n, val_good (VInt n)
  | VG_Bool : forall b, val_good (VBool b)
  | VG_String : forall s, val_good (VString s)
  | VG_Unit : val_good VUnit
  | VG_Clos : forall x body cenv,
      pure body -> env_good cenv -> val_good (VClos x body cenv)
  | VG_FixClos : forall f x body cenv,
      pure body -> env_good cenv -> val_good (VFixClos f x body cenv)
  | VG_Array : forall vs, Forall val_good vs -> val_good (VArray vs)
  | VG_Record : forall fvs,
      Forall (fun fv => val_good (snd fv)) fvs -> val_good (VRecord fvs)
  | VG_Construct : forall tag v, val_good v -> val_good (VConstruct tag v)
with env_good : env -> Prop :=
  | EG_Nil : env_good ENil
  | EG_Cons : forall x v renv,
      val_good v -> env_good renv -> env_good (ECons x v renv).

(** env_good lookup *)
Lemma env_good_lookup : forall renv x v,
  env_good renv -> env_lookup x renv = Some v -> val_good v.
Proof.
  intros renv x v Hgood Hlook.
  induction Hgood.
  - simpl in Hlook. discriminate.
  - simpl in Hlook. destruct (String.eqb x x0) eqn:?.
    + injection Hlook; intros; subst. assumption.
    + apply IHHgood. assumption.
Qed.

(** ** Close distribution lemmas *)

Lemma close_int : forall renv n, close renv (EInt n) = EInt n.
Proof. intros. induction renv; simpl; auto. Qed.

Lemma close_bool : forall renv b, close renv (EBool b) = EBool b.
Proof. intros. induction renv; simpl; auto. Qed.

Lemma close_string : forall renv s, close renv (EString s) = EString s.
Proof. intros. induction renv; simpl; auto. Qed.

Lemma close_unit : forall renv, close renv EUnit = EUnit.
Proof. intros. induction renv; simpl; auto. Qed.

Lemma close_var : forall renv x,
  all_vals_closed renv ->
  close renv (EVar x) =
  match env_lookup x renv with
  | Some v => val_to_expr v
  | None => EVar x
  end.
Proof.
  intros renv x Havc.
  induction renv as [|y vy rest IH]; simpl in *.
  - reflexivity.
  - destruct Havc as [Hcvy Hrest].
    destruct (String.eqb x y) eqn:Hxy.
    + apply String.eqb_eq in Hxy. subst y.
      simpl. rewrite eqb_refl.
      clear IH.
      induction rest as [|z vz rest' IH']; simpl in *; auto.
      destruct Hrest as [Hcvz Hrest'].
      rewrite Hcvy. apply IH'. assumption.
    + simpl. rewrite (eqb_sym y x). rewrite Hxy. apply IH. assumption.
Qed.

Lemma close_binop : forall renv op e1 e2,
  close renv (EBinOp op e1 e2) = EBinOp op (close renv e1) (close renv e2).
Proof.
  intro renv; induction renv as [|y vy rest IH]; intros; simpl; auto.
Qed.

Lemma close_unop : forall renv op e,
  close renv (EUnOp op e) = EUnOp op (close renv e).
Proof.
  intro renv; induction renv as [|y vy rest IH]; intros; simpl; auto.
Qed.

Lemma close_if : forall renv e1 e2 e3,
  close renv (EIf e1 e2 e3) = EIf (close renv e1) (close renv e2) (close renv e3).
Proof.
  intro renv; induction renv as [|y vy rest IH]; intros; simpl; auto.
Qed.

Lemma close_seq : forall renv e1 e2,
  close renv (ESeq e1 e2) = ESeq (close renv e1) (close renv e2).
Proof.
  intro renv; induction renv as [|y vy rest IH]; intros; simpl; auto.
Qed.

Lemma close_app : forall renv e1 e2,
  close renv (EApp e1 e2) = EApp (close renv e1) (close renv e2).
Proof.
  intro renv; induction renv as [|y vy rest IH]; intros; simpl; auto.
Qed.

Lemma close_index : forall renv e1 e2,
  close renv (EIndex e1 e2) = EIndex (close renv e1) (close renv e2).
Proof.
  intro renv; induction renv as [|y vy rest IH]; intros; simpl; auto.
Qed.

Lemma close_arrayset : forall renv e1 e2 e3,
  close renv (EArraySet e1 e2 e3) =
  EArraySet (close renv e1) (close renv e2) (close renv e3).
Proof.
  intro renv; induction renv as [|y vy rest IH]; intros; simpl; auto.
Qed.

Lemma close_arraypush : forall renv e1 e2,
  close renv (EArrayPush e1 e2) = EArrayPush (close renv e1) (close renv e2).
Proof.
  intro renv; induction renv as [|y vy rest IH]; intros; simpl; auto.
Qed.

Lemma close_field : forall renv e f,
  close renv (EField e f) = EField (close renv e) f.
Proof.
  intro renv; induction renv as [|y vy rest IH]; intros; simpl; auto.
Qed.

Lemma close_construct : forall renv tag e t,
  close renv (EConstruct tag e t) = EConstruct tag (close renv e) t.
Proof.
  intro renv; induction renv as [|y vy rest IH]; intros; simpl; auto.
Qed.

Lemma close_strindex : forall renv e1 e2,
  close renv (EStrIndex e1 e2) = EStrIndex (close renv e1) (close renv e2).
Proof.
  intro renv; induction renv as [|y vy rest IH]; intros; simpl; auto.
Qed.

Lemma close_array : forall renv es,
  close renv (EArray es) = EArray (map (close renv) es).
Proof.
  intro renv. induction renv as [|y vy rest IH]; intros; simpl.
  - f_equal. induction es; simpl; auto. f_equal. assumption.
  - rewrite IH. f_equal.
    induction es as [|e0 rest' IH']; simpl; auto.
    f_equal. assumption.
Qed.

Lemma close_record : forall renv fes,
  close renv (ERecord fes) = ERecord (map (fun '(f, e) => (f, close renv e)) fes).
Proof.
  intro renv. induction renv as [|y vy rest IH]; intros; simpl.
  - f_equal. induction fes as [|[f0 e0] rest' IH']; simpl; auto.
    f_equal; auto.
  - rewrite IH. f_equal.
    induction fes as [|[f0 e0] rest' IH']; simpl; auto.
    f_equal; auto.
Qed.

(** ** close_except / close_except2 properties *)
(* close_except and close_except2 are defined in the mutual fixpoint with val_to_expr/close *)

Lemma close_let : forall renv x e1 e2,
  close renv (ELet x e1 e2) = ELet x (close renv e1) (close_except x renv e2).
Proof.
  intro renv. induction renv as [|y vy rest IH]; intros; simpl; auto.
  rewrite IH. f_equal.
  destruct (String.eqb y x) eqn:Hyx; simpl; auto.
Qed.

Lemma close_lam : forall renv x t body,
  close renv (ELam x t body) = ELam x t (close_except x renv body).
Proof.
  intro renv. induction renv as [|y vy rest IH]; intros; simpl; auto.
  rewrite IH. f_equal.
  destruct (String.eqb y x) eqn:Hyx; simpl; auto.
Qed.

Lemma close_fix : forall renv f x t1 t2 body,
  close renv (EFix f x t1 t2 body) = EFix f x t1 t2 (close_except2 f x renv body).
Proof.
  intro renv. induction renv as [|y vy rest IH]; intros; simpl; auto.
  rewrite IH. f_equal.
  destruct (String.eqb y f || String.eqb y x)%bool eqn:Hyf; simpl; auto.
Qed.

(** close distributes over match branches *)
Fixpoint close_match_branches (renv : env) (branches : list (string * string * expr))
  : list (string * string * expr) :=
  match branches with
  | [] => []
  | (tag, x, body) :: rest =>
      (tag, x, close_except x renv body) :: close_match_branches renv rest
  end.

Lemma close_match : forall renv e branches,
  close renv (EMatch e branches) =
  EMatch (close renv e) (close_match_branches renv branches).
Proof.
  intro renv. induction renv as [|y vy rest IH]; intros; simpl; auto.
  - f_equal. induction branches as [|[[t0 z] b] rest' IH']; simpl; auto.
    f_equal; auto.
  - rewrite IH. f_equal.
    induction branches as [|[[t0 z] b] rest' IH']; simpl; auto.
    f_equal; [| assumption].
    f_equal.
    destruct (String.eqb y z) eqn:Hyz; simpl; auto.
Qed.

Lemma find_branch_close : forall tag renv branches z body,
  find_branch tag branches = Some (z, body) ->
  find_branch tag (close_match_branches renv branches) =
  Some (z, close_except z renv body).
Proof.
  intros tag renv branches z body Hfind.
  induction branches as [|[[t0 y] b] rest IH]; simpl in *.
  - discriminate.
  - destruct (String.eqb tag t0) eqn:Htag.
    + injection Hfind; intros; subst. reflexivity.
    + apply IH. assumption.
Qed.

(** ** Substitution / close_except commutation (same variable) *)

Lemma subst_close_except : forall x v renv e,
  all_vals_closed renv ->
  eclosed v ->
  subst x v (close_except x renv e) = close renv (subst x v e).
Proof.
  intros x v renv.
  induction renv as [|y vy rest IH]; intros e Havc Hcv; simpl in *.
  - reflexivity.
  - destruct Havc as [Hcvy Hrest].
    destruct (String.eqb y x) eqn:Hyx.
    + apply String.eqb_eq in Hyx. subst y.
      rewrite IH; auto.
      f_equal. rewrite subst_subst_same.
      f_equal. symmetry. apply Hcv.
    + rewrite IH; auto.
      rewrite subst_comm_closed; auto.
      intro Heq. subst. rewrite eqb_refl in Hyx. discriminate.
Qed.

Lemma subst_close_except2 : forall f x vf vx renv e,
  all_vals_closed renv ->
  eclosed vf -> eclosed vx ->
  subst f vf (subst x vx (close_except2 f x renv e)) =
  close renv (subst f vf (subst x vx e)).
Proof.
  intros f x vf vx renv.
  induction renv as [|y vy rest IH]; intros e Havc Hcvf Hcvx; simpl in *.
  - reflexivity.
  - destruct Havc as [Hcvy Hrest].
    destruct (String.eqb y f || String.eqb y x)%bool eqn:Hyf.
    + rewrite Bool.orb_true_iff in Hyf. destruct Hyf as [Hyf | Hyx].
      * (* y = f *)
        apply String.eqb_eq in Hyf. subst y.
        rewrite IH; auto.
        (* RHS: close rest (subst f vte_vy (subst f vf (subst x vx e))) *)
        (* subst f vte_vy (subst f vf X) = subst f (subst f vte_vy vf) X *)
        (*   = subst f vf X  because eclosed vf *)
        rewrite subst_subst_same.
        f_equal. f_equal. symmetry. apply Hcvf.
      * (* y = x *)
        apply String.eqb_eq in Hyx. subst y.
        rewrite IH; auto.
        destruct (String.eqb x f) eqn:Hxf.
        { apply String.eqb_eq in Hxf. subst f.
          f_equal. symmetry. rewrite subst_subst_same.
          f_equal. apply Hcvf. }
        { (* x <> f case *)
          f_equal. symmetry.
          rewrite (subst_comm_closed x f); auto;
            [| intro Heq; subst; rewrite eqb_refl in Hxf; discriminate].
          rewrite subst_subst_same.
          replace (subst x (val_to_expr vy) vx) with vx by (symmetry; apply Hcvx).
          reflexivity. }
    + rewrite Bool.orb_false_iff in Hyf. destruct Hyf as [Hyf Hyx].
      rewrite IH; auto. f_equal.
      rewrite (subst_comm_closed x y); auto;
        [| intro Heq; subst; rewrite eqb_refl in Hyx; discriminate].
      rewrite (subst_comm_closed f y); auto.
      intro Heq; subst; rewrite eqb_refl in Hyf; discriminate.
Qed.

(** ** Multi-step reduction helpers *)

Lemma let_reduce : forall x v e2,
  is_value v ->
  multi_step (ELet x v e2) (subst x v e2).
Proof.
  intros. eapply MS_Step; [apply S_LetVal; assumption | apply MS_Refl].
Qed.

Lemma app_reduce_lam : forall x t body v,
  is_value v ->
  multi_step (EApp (ELam x t body) v) (subst x v body).
Proof.
  intros. eapply MS_Step; [apply S_AppBeta; assumption | apply MS_Refl].
Qed.

Lemma app_reduce_fix : forall f x t1 t2 body v,
  is_value v ->
  multi_step (EApp (EFix f x t1 t2 body) v)
             (subst f (EFix f x t1 t2 body) (subst x v body)).
Proof.
  intros. eapply MS_Step; [apply S_AppFixBeta; assumption | apply MS_Refl].
Qed.

(** ** val_good implies eclosed (val_to_expr v)
    For closure values, this requires knowing close renv body is closed
    when env_good renv and body only uses variables from renv.
    We admit this for now — it's a property of well-scoped programs. *)

Axiom val_good_eclosed : forall v, val_good v -> eclosed (val_to_expr v).

Lemma env_good_all_vals_closed : forall renv,
  env_good renv -> all_vals_closed renv.
Proof.
  intros renv Hgood. induction Hgood; simpl.
  - exact I.
  - split; [apply val_good_eclosed; assumption | assumption].
Qed.

(** ** Main generalized theorem *)

Ltac pure_env_eq :=
  repeat match goal with
  | [ Heval : eval ?renv ?e ?renv' _, Hpure : pure ?e |- _ ] =>
    match renv' with
    | renv => fail 1
    | _ =>
      let H := fresh "Henv" in
      assert (H : renv' = renv) by (eapply pure_env_unchanged; eassumption);
      subst renv'
    end
  end.

(** Helper: invert an expr_equiv for an EArray value and name the result es' *)
Ltac inv_array_equiv H :=
  inversion H; subst;
  match goal with
  | [ H2 : Forall2 expr_equiv ?x (map val_to_expr _) |- _ ] => rename x into es'
  end.

(** Helper: invert an expr_equiv for an ERecord value and name the result fes' *)
Ltac inv_record_equiv H :=
  inversion H; subst;
  match goal with
  | [ H2 : Forall2 _ ?x (map _ _) |- _ ] => rename x into fes'
  end.

(** Helper: apply an IH to get multi_step_equiv and val_good.
    We use `pose proof` to mark the IH as used, then clear the original.
    The match picks the most recent unused IH (bottom-up = last subexpr first).
    After two calls with pattern `apply_IH. apply_IH.`:
      Hms = last subexpr, Hms0 = second-to-last subexpr.
    The proof cases use Hms/Hms0 names accordingly. *)
Ltac apply_IH :=
  match goal with
  | [ IH : pure ?e -> env_good ?renv -> all_vals_closed ?renv ->
           multi_step_equiv (close ?renv ?e) (val_to_expr ?v) /\ val_good ?v |- _ ] =>
    let H := fresh "IH_result" in
    pose proof (IH ltac:(assumption) ltac:(assumption) ltac:(assumption)) as H;
    clear IH;
    let Hms := fresh "Hms" in
    let Hvg := fresh "Hvg" in
    destruct H as [Hms Hvg]
  end.

(** Helper: build multi_step_equiv through binary operator *)
Ltac solve_binop_mse :=
  match goal with
  | [ Hms1 : multi_step_equiv _ _, Hms2 : multi_step_equiv _ _ |- multi_step_equiv (EBinOp ?op _ _) _ ] =>
    destruct Hms1 as [?e1' [?Hms1' ?Heq1']];
    destruct Hms2 as [?e2' [?Hms2' ?Heq2']];
    exists (EBinOp op e1' e2'); split;
    [ eapply multi_step_trans;
      [ apply ms_binop1; exact Hms1'
      | apply ms_binop2;
        [eapply expr_equiv_is_value; [apply expr_equiv_sym; exact Heq1' | apply val_to_expr_is_value]
        | exact Hms2'] ]
    | constructor; assumption ]
  end.

Theorem eval_to_multistep_gen : forall renv e renv' v,
  pure e -> eval renv e renv' v ->
  env_good renv -> all_vals_closed renv ->
  multi_step_equiv (close renv e) (val_to_expr v)
  /\ val_good v.
Proof.
  intros renv e renv' v Hpure Heval Hgood Havc.
  revert Hpure Hgood Havc.
  induction Heval; intros Hpure Hgood Havc; inversion Hpure; subst;
    try (pure_env_eq).

  (* E_Int *)
  - split; [| constructor].
    rewrite close_int. apply multi_step_to_equiv. apply MS_Refl.

  (* E_Bool *)
  - split; [| constructor].
    rewrite close_bool. apply multi_step_to_equiv. apply MS_Refl.

  (* E_String *)
  - split; [| constructor].
    rewrite close_string. apply multi_step_to_equiv. apply MS_Refl.

  (* E_Unit *)
  - split; [| constructor].
    rewrite close_unit. apply multi_step_to_equiv. apply MS_Refl.

  (* E_Var *)
  - split; [| eapply env_good_lookup; eassumption].
    rewrite close_var by assumption. rewrite H.
    apply multi_step_to_equiv. apply MS_Refl.

  (* E_BinArith: eval_arith_binop op n1 n2 = Some v *)
  - apply_IH. apply_IH.
    destruct (eval_arith_binop_int _ _ _ _ H) as [nr ?]; subst.
    split; [| constructor]. rewrite close_binop.
    destruct Hms0 as [e1' [Hms1' Heq1']]. destruct Hms as [e2' [Hms2' Heq2']].
    simpl in Heq1'. inversion Heq1'; subst.
    simpl in Heq2'. inversion Heq2'; subst.
    exists (EInt nr). split; [| apply expr_equiv_refl].
    eapply multi_step_trans. eapply multi_step_trans.
    + apply ms_binop1. exact Hms1'.
    + apply ms_binop2; [constructor | exact Hms2'].
    + eapply MS_Step; [| apply MS_Refl].
      apply S_BinOp; [constructor | constructor |].
      change (EInt nr) with (val_to_expr (VInt nr)).
      apply arith_binop_apply. assumption.

  (* E_BinCmp: eval_cmp_binop op n1 n2 = Some v *)
  - apply_IH. apply_IH.
    destruct (eval_cmp_binop_bool _ _ _ _ H) as [br ?]; subst.
    split; [| constructor]. rewrite close_binop.
    destruct Hms0 as [e1' [Hms1' Heq1']]. destruct Hms as [e2' [Hms2' Heq2']].
    simpl in Heq1'. inversion Heq1'; subst.
    simpl in Heq2'. inversion Heq2'; subst.
    exists (EBool br). split; [| apply expr_equiv_refl].
    eapply multi_step_trans. eapply multi_step_trans.
    + apply ms_binop1; eassumption.
    + apply ms_binop2; [constructor | eassumption].
    + eapply MS_Step; [| apply MS_Refl].
      apply S_BinOp; [constructor | constructor |].
      change (EBool br) with (val_to_expr (VBool br)).
      apply cmp_binop_apply. assumption.

  (* E_BinEqBool: op = OpEq, result = VBool (Bool.eqb b1 b2) *)
  - apply_IH. apply_IH.
    split; [| constructor]. rewrite close_binop.
    destruct Hms0 as [e1' [Hms1' Heq1']]. destruct Hms as [e2' [Hms2' Heq2']].
    simpl in Heq1'. inversion Heq1'; subst.
    simpl in Heq2'. inversion Heq2'; subst.
    exists (EBool (Bool.eqb b1 b2)). split; [| apply expr_equiv_refl].
    eapply multi_step_trans. eapply multi_step_trans.
    + apply ms_binop1; eassumption.
    + apply ms_binop2; [constructor | eassumption].
    + eapply MS_Step; [| apply MS_Refl].
      apply S_BinOp; [constructor | constructor |]. simpl. reflexivity.

  (* E_BinNeBool: op = OpNe, result = VBool (negb (Bool.eqb b1 b2)) *)
  - apply_IH. apply_IH.
    split; [| constructor]. rewrite close_binop.
    destruct Hms0 as [e1' [Hms1' Heq1']]. destruct Hms as [e2' [Hms2' Heq2']].
    simpl in Heq1'. inversion Heq1'; subst.
    simpl in Heq2'. inversion Heq2'; subst.
    exists (EBool (negb (Bool.eqb b1 b2))). split; [| apply expr_equiv_refl].
    eapply multi_step_trans. eapply multi_step_trans.
    + apply ms_binop1; eassumption.
    + apply ms_binop2; [constructor | eassumption].
    + eapply MS_Step; [| apply MS_Refl].
      apply S_BinOp; [constructor | constructor |]. simpl. reflexivity.

  (* E_StrCat: result = VString (append s1 s2) *)
  - apply_IH. apply_IH.
    split; [| constructor]. rewrite close_binop.
    destruct Hms0 as [e1' [Hms1' Heq1']]. destruct Hms as [e2' [Hms2' Heq2']].
    simpl in Heq1'. inversion Heq1'; subst.
    simpl in Heq2'. inversion Heq2'; subst.
    exists (EString (String.append s1 s2)). split; [| apply expr_equiv_refl].
    eapply multi_step_trans. eapply multi_step_trans.
    + apply ms_binop1; eassumption.
    + apply ms_binop2; [constructor | eassumption].
    + eapply MS_Step; [| apply MS_Refl].
      apply S_BinOp; [constructor | constructor |]. simpl. reflexivity.

  (* E_BinEqStr: op = OpEq, result = VBool (String.eqb s1 s2) *)
  - apply_IH. apply_IH.
    split; [| constructor]. rewrite close_binop.
    destruct Hms0 as [e1' [Hms1' Heq1']]. destruct Hms as [e2' [Hms2' Heq2']].
    simpl in Heq1'. inversion Heq1'; subst.
    simpl in Heq2'. inversion Heq2'; subst.
    exists (EBool (String.eqb s1 s2)). split; [| apply expr_equiv_refl].
    eapply multi_step_trans. eapply multi_step_trans.
    + apply ms_binop1; eassumption.
    + apply ms_binop2; [constructor | eassumption].
    + eapply MS_Step; [| apply MS_Refl].
      apply S_BinOp; [constructor | constructor |]. simpl. reflexivity.

  (* E_BinNeStr: op = OpNe, result = VBool (negb (String.eqb s1 s2)) *)
  - apply_IH. apply_IH.
    split; [| constructor]. rewrite close_binop.
    destruct Hms0 as [e1' [Hms1' Heq1']]. destruct Hms as [e2' [Hms2' Heq2']].
    simpl in Heq1'. inversion Heq1'; subst.
    simpl in Heq2'. inversion Heq2'; subst.
    exists (EBool (negb (String.eqb s1 s2))). split; [| apply expr_equiv_refl].
    eapply multi_step_trans. eapply multi_step_trans.
    + apply ms_binop1; eassumption.
    + apply ms_binop2; [constructor | eassumption].
    + eapply MS_Step; [| apply MS_Refl].
      apply S_BinOp; [constructor | constructor |]. simpl. reflexivity.

  (* E_And_True: eval e1 => true, eval e2 => VBool v2 *)
  - apply_IH. apply_IH.
    split; [| constructor]. rewrite close_binop.
    destruct Hms0 as [e1' [Hms1' Heq1']]. destruct Hms as [e2' [Hms2' Heq2']].
    simpl in Heq1'. inversion Heq1'; subst.
    simpl in Heq2'. inversion Heq2'; subst.
    exists (EBool v2). split; [| apply expr_equiv_refl].
    eapply multi_step_trans. eapply multi_step_trans.
    + apply ms_binop1; eassumption.
    + apply ms_binop2; [constructor | eassumption].
    + eapply MS_Step; [| apply MS_Refl].
      apply S_BinOp; [constructor | constructor |]. simpl. reflexivity.

  (* E_And_False: eval e1 => false, e2 = EBool false, result = VBool false *)
  - apply_IH.
    split; [| constructor]. rewrite close_binop.
    destruct Hms as [e1' [Hms1' Heq1']].
    simpl in Heq1'. inversion Heq1'; subst.
    exists (EBool false). split; [| apply expr_equiv_refl].
    eapply multi_step_trans.
    + apply ms_binop1; eassumption.
    + eapply MS_Step; [apply S_AndFalse | apply MS_Refl].

  (* E_And_Short: eval e1 => false, result = VBool false *)
  - apply_IH.
    split; [| constructor]. rewrite close_binop.
    destruct Hms as [e1' [Hms1' Heq1']].
    simpl in Heq1'. inversion Heq1'; subst.
    exists (EBool false). split; [| apply expr_equiv_refl].
    eapply multi_step_trans.
    + apply ms_binop1; eassumption.
    + eapply MS_Step; [apply S_AndFalse | apply MS_Refl].

  (* E_Or_False: eval e1 => false, eval e2 => VBool v2 *)
  - apply_IH. apply_IH.
    split; [| constructor]. rewrite close_binop.
    destruct Hms0 as [e1' [Hms1' Heq1']]. destruct Hms as [e2' [Hms2' Heq2']].
    simpl in Heq1'. inversion Heq1'; subst.
    simpl in Heq2'. inversion Heq2'; subst.
    exists (EBool v2). split; [| apply expr_equiv_refl].
    eapply multi_step_trans. eapply multi_step_trans.
    + apply ms_binop1; eassumption.
    + apply ms_binop2; [constructor | eassumption].
    + eapply MS_Step; [| apply MS_Refl].
      apply S_BinOp; [constructor | constructor |]. simpl. reflexivity.

  (* E_Or_Short: eval e1 => true, result = VBool true *)
  - apply_IH.
    split; [| constructor]. rewrite close_binop.
    destruct Hms as [e1' [Hms1' Heq1']].
    simpl in Heq1'. inversion Heq1'; subst.
    exists (EBool true). split; [| apply expr_equiv_refl].
    eapply multi_step_trans.
    + apply ms_binop1; eassumption.
    + eapply MS_Step; [apply S_OrTrue | apply MS_Refl].

  (* E_Neg: eval e => VInt n, result = VInt (- n) *)
  - apply_IH.
    split; [| constructor]. rewrite close_unop.
    destruct Hms as [e' [Hms' Heq']].
    simpl in Heq'. inversion Heq'; subst.
    exists (EInt (- n)). split; [| apply expr_equiv_refl].
    eapply multi_step_trans.
    + apply ms_unop; eassumption.
    + eapply MS_Step; [apply S_Neg | apply MS_Refl].

  (* E_Not: eval e => VBool b, result = VBool (negb b) *)
  - apply_IH.
    split; [| constructor]. rewrite close_unop.
    destruct Hms as [e' [Hms' Heq']].
    simpl in Heq'. inversion Heq'; subst.
    exists (EBool (negb b)). split; [| apply expr_equiv_refl].
    eapply multi_step_trans.
    + apply ms_unop; eassumption.
    + eapply MS_Step; [apply S_Not | apply MS_Refl].

  (* E_StrLen: eval e => VString s, result = VInt (length s) *)
  - apply_IH.
    split; [| constructor]. rewrite close_unop.
    destruct Hms as [e' [Hms' Heq']].
    simpl in Heq'. inversion Heq'; subst.
    exists (EInt (Z.of_nat (String.length s))). split; [| apply expr_equiv_refl].
    eapply multi_step_trans.
    + apply ms_unop; eassumption.
    + eapply MS_Step; [apply S_StrLen | apply MS_Refl].

  (* E_IfTrue: eval e1 => true, eval e2 => v *)
  - apply_IH. apply_IH.
    split; [| assumption]. rewrite close_if.
    destruct Hms0 as [e1' [Hms1' Heq1']].
    simpl in Heq1'. inversion Heq1'; subst.
    eapply multi_step_equiv_trans.
    + exists (EIf (EBool true) (close renv e2) (close renv e3)). split.
      * apply ms_if. exact Hms1'.
      * apply expr_equiv_refl.
    + eapply multi_step_equiv_trans.
      * exists (close renv e2). split.
        -- eapply MS_Step; [apply S_IfTrue | apply MS_Refl].
        -- apply expr_equiv_refl.
      * exact Hms.

  (* E_IfFalse: eval e1 => false, eval e3 => v *)
  - apply_IH. apply_IH.
    split; [| assumption]. rewrite close_if.
    destruct Hms0 as [e1' [Hms1' Heq1']].
    simpl in Heq1'. inversion Heq1'; subst.
    eapply multi_step_equiv_trans.
    + exists (EIf (EBool false) (close renv e2) (close renv e3)). split.
      * apply ms_if. exact Hms1'.
      * apply expr_equiv_refl.
    + eapply multi_step_equiv_trans.
      * exists (close renv e3). split.
        -- eapply MS_Step; [apply S_IfFalse | apply MS_Refl].
        -- apply expr_equiv_refl.
      * exact Hms.

  (* E_Let: eval e1 => v1, eval (ECons x v1 renv) e2 => v2 *)
  - apply_IH.
    assert (Hcv1 : eclosed (val_to_expr v1)) by (apply val_good_eclosed; assumption).
    assert (Havc2 : all_vals_closed (ECons x v1 renv)).
    { simpl. split; [exact Hcv1 | assumption]. }
    match goal with
    | [ IH2 : pure ?e2 -> env_good (ECons x v1 renv) -> all_vals_closed (ECons x v1 renv) ->
              multi_step_equiv (close (ECons x v1 renv) ?e2) (val_to_expr ?v2) /\ val_good ?v2 |- _ ] =>
      destruct (IH2 ltac:(assumption) ltac:(constructor; assumption) Havc2) as [Hms2 Hvg2]
    end.
    split; [| assumption]. rewrite close_let.
    assert (Hkey : close (ECons x v1 renv) e2 =
                   subst x (val_to_expr v1) (close_except x renv e2)).
    { simpl. symmetry. apply subst_close_except; assumption. }
    rewrite Hkey in Hms2.
    destruct Hms as [e1' [Hms1' Heq1']].
    assert (Hv1' : is_value e1').
    { eapply expr_equiv_is_value. apply expr_equiv_sym; eassumption. apply val_to_expr_is_value. }
    eapply multi_step_equiv_trans.
    + exists (subst x e1' (close_except x renv e2)). split.
      * eapply multi_step_trans.
        -- apply ms_let. exact Hms1'.
        -- eapply MS_Step; [apply S_LetVal; exact Hv1' | apply MS_Refl].
      * apply expr_equiv_refl.
    + eapply multi_step_equiv_trans; [| exact Hms2].
      exists (subst x e1' (close_except x renv e2)). split.
      * apply MS_Refl.
      * apply subst_expr_equiv; [exact Heq1' | apply expr_equiv_refl].

  (* E_Seq: eval e1 => v1, eval e2 => v2 *)
  - apply_IH. apply_IH.
    split; [| assumption]. rewrite close_seq.
    destruct Hms0 as [e1' [Hms1' Heq1']].
    assert (Hv1' : is_value e1').
    { eapply expr_equiv_is_value. apply expr_equiv_sym; eassumption. apply val_to_expr_is_value. }
    eapply multi_step_equiv_trans.
    + exists (ESeq e1' (close renv e2)). split.
      * apply ms_seq1. exact Hms1'.
      * apply expr_equiv_refl.
    + eapply multi_step_equiv_trans.
      * exists (close renv e2). split.
        -- eapply MS_Step; [apply S_SeqVal; exact Hv1' | apply MS_Refl].
        -- apply expr_equiv_refl.
      * exact Hms.

  (* E_Lam: result = VClos x body renv *)
  - split; [| constructor; assumption].
    rewrite close_lam.
    exists (ELam x t (close_except x renv body)). split.
    + apply MS_Refl.
    + apply EQ_Lam. apply expr_equiv_refl.

  (* E_App: eval e1 => VClos x body clos_env, eval e2 => v2, eval body => v *)
  - apply_IH. apply_IH.
    inversion Hvg0; subst.
    assert (Hcv2 : eclosed (val_to_expr v2)) by (apply val_good_eclosed; assumption).
    assert (Havc_clos : all_vals_closed clos_env) by (apply env_good_all_vals_closed; assumption).
    match goal with
    | [ IH3 : pure body -> env_good (ECons x v2 clos_env) -> all_vals_closed (ECons x v2 clos_env) ->
              multi_step_equiv (close (ECons x v2 clos_env) body) (val_to_expr ?vr) /\ val_good ?vr |- _ ] =>
      destruct (IH3 ltac:(assumption) ltac:(constructor; assumption) ltac:(simpl; split; [exact Hcv2 | exact Havc_clos])) as [Hms3 Hvg3]
    end.
    split; [| assumption]. rewrite close_app.
    assert (Hkey : subst x (val_to_expr v2) (close_except x clos_env body) =
                   close (ECons x v2 clos_env) body).
    { simpl. apply subst_close_except; assumption. }
    rewrite <- Hkey in Hms3.
    destruct Hms0 as [e1' [Hms1' Heq1']]. destruct Hms as [e2' [Hms2' Heq2']].
    assert (Hv2' : is_value e2').
    { eapply expr_equiv_is_value. apply expr_equiv_sym; eassumption. apply val_to_expr_is_value. }
    destruct (expr_equiv_lam_inv _ _ _ _ Heq1') as [t' [body' [? Hbody'eq]]]; subst e1'.
    eapply multi_step_equiv_trans.
    + exists (EApp (ELam x t' body') e2'). split.
      * eapply multi_step_trans.
        -- apply ms_app1; eassumption.
        -- apply ms_app2; [constructor | eassumption].
      * apply expr_equiv_refl.
    + eapply multi_step_equiv_trans.
      * exists (subst x e2' body'). split.
        -- eapply MS_Step; [apply S_AppBeta; exact Hv2' | apply MS_Refl].
        -- apply expr_equiv_refl.
      * eapply multi_step_equiv_trans; [| exact Hms3].
        exists (subst x e2' body'). split; [apply MS_Refl |].
        apply subst_expr_equiv; [exact Heq2' | exact Hbody'eq].

  (* E_ArrayNil: result = VArray [] *)
  - split; [| constructor; constructor].
    rewrite close_array. simpl.
    apply multi_step_to_equiv. apply MS_Refl.

  (* E_ArrayCons: eval e => v, eval (EArray es) => VArray vs *)
  - match goal with
    | [ H : Forall pure (_ :: _) |- _ ] => inversion H; subst; clear H
    end.
    (* Manually equate environments since pure_env_eq couldn't *)
    match goal with
    | [ Heval : eval ?r ?e ?r' _, Hp : pure ?e |- _ ] =>
      match r' with
      | r => idtac
      | _ => let Heq := fresh "Heq" in
             assert (Heq : r' = r) by (eapply pure_env_unchanged; eassumption); subst r'
      end
    end.
    apply_IH.
    match goal with
    | [ IH2 : pure (EArray ?es) -> env_good ?renv -> all_vals_closed ?renv ->
              multi_step_equiv (close ?renv (EArray ?es)) (val_to_expr (VArray ?vs)) /\ val_good (VArray ?vs) |- _ ] =>
      destruct (IH2 ltac:(constructor; assumption) ltac:(assumption) ltac:(assumption)) as [Hms2 Hvg2]
    end.
    inversion Hvg2; subst.
    split.
    + (* multi_step_equiv *)
      rewrite close_array. simpl.
      destruct Hms as [e' [Hms' Heq']].
      assert (Hv' : is_value e').
      { eapply expr_equiv_is_value. apply expr_equiv_sym; eassumption. apply val_to_expr_is_value. }
      destruct Hms2 as [ea' [Hms2' Heq2']].
      (* ea' ≡ EArray (map val_to_expr vs), so ea' = EArray es' with Forall2 expr_equiv *)
      simpl in Heq2'. inversion Heq2'; subst.
      match goal with
      | [ H : Forall2 expr_equiv ?es' (map val_to_expr vs) |- _ ] =>
        exists (EArray (e' :: es')); split;
        [ eapply multi_step_trans;
          [ apply ms_array_head; exact Hms'
          | apply ms_array_tail; [exact Hv' |];
            rewrite close_array in Hms2'; exact Hms2' ]
        | constructor; constructor; assumption ]
      end.
    + (* val_good *)
      constructor. constructor; assumption.

  (* E_Index: eval e1 => VArray vs, eval e2 => VInt n, nth_error vs n = Some v *)
  - apply_IH. apply_IH.
    split.
    + rewrite close_index.
      destruct Hms0 as [e1' [Hms1' Heq1']]. destruct Hms as [e2' [Hms2' Heq2']].
      simpl in Heq2'. inversion Heq2'; subst.
      (* e1' ≡ EArray (map val_to_expr vs), so e1' = EArray es' *)
      simpl in Heq1'. inv_array_equiv Heq1'.
      assert (Hv1' : Forall is_value es').
      { eapply forall_value_transfer_rev; [apply forall_val_to_expr_is_value | eassumption]. }
      exists (nth (Z.to_nat n) es' EUnit). split.
      * eapply multi_step_trans. eapply multi_step_trans.
        -- apply ms_index1; eassumption.
        -- apply ms_index2; [constructor; assumption | eassumption].
        -- eapply MS_Step; [| apply MS_Refl].
           apply S_IndexVal. exact Hv1'.
      * eapply expr_equiv_trans.
        -- apply nth_equiv; [exact Hv1' | eassumption].
        -- rewrite (nth_error_map_nth _ _ val_to_expr vs (Z.to_nat n) v EUnit H).
           apply expr_equiv_refl.
    + (* val_good: nth_error vs (Z.to_nat n) = Some v *)
      eapply Forall_nth_error.
      * inversion Hvg0; subst. eassumption.
      * eassumption.

  (* E_ArrayLen: eval e => VArray vs, result = VInt (length vs) *)
  - apply_IH.
    split; [| constructor]. rewrite close_unop.
    destruct Hms as [e' [Hms' Heq']].
    simpl in Heq'. inv_array_equiv Heq'.
    assert (Hv' : Forall is_value es').
    { eapply forall_value_transfer_rev; [apply forall_val_to_expr_is_value | eassumption]. }
    exists (EInt (Z.of_nat (length es'))). split.
    + eapply multi_step_trans.
      * apply ms_unop; eassumption.
      * eapply MS_Step; [apply S_ArrayLen; exact Hv' | apply MS_Refl].
    + simpl.
      assert (Hlen : length es' = length vs).
      { pose proof (length_forall2 _ _ H2) as Htmp.
        rewrite map_length in Htmp. symmetry. exact Htmp. }
      rewrite Hlen. apply expr_equiv_refl.

  (* E_RecordNil: result = VRecord [] *)
  - split; [| constructor; constructor].
    rewrite close_record. simpl.
    apply multi_step_to_equiv. apply MS_Refl.

  (* E_RecordCons: eval e => v, eval (ERecord es) => VRecord vs *)
  - match goal with
    | [ H : Forall (fun fe => pure (snd fe)) ((_ , _) :: _) |- _ ] => inversion H; subst; clear H
    end.
    (* Manually equate environments *)
    match goal with
    | [ Heval : eval ?r ?e ?r' _, Hp : pure ?e |- _ ] =>
      match r' with
      | r => idtac
      | _ => let Heq := fresh "Heq" in
             assert (Heq : r' = r) by (eapply pure_env_unchanged; eassumption); subst r'
      end
    end.
    apply_IH.
    match goal with
    | [ IH2 : Forall (fun fe => pure (snd fe)) ?es0 -> env_good ?r -> all_vals_closed ?r ->
              multi_step_equiv (close ?r (ERecord ?es0)) (val_to_expr (VRecord ?vs0)) /\ val_good (VRecord ?vs0) |- _ ] =>
      destruct (IH2 ltac:(assumption) ltac:(assumption) ltac:(assumption)) as [Hms2 Hvg2]
    | [ IH2 : ?P -> env_good ?r -> all_vals_closed ?r ->
              multi_step_equiv _ (val_to_expr (VRecord ?vs0)) /\ val_good (VRecord ?vs0) |- _ ] =>
      destruct (IH2 ltac:(try assumption; try (constructor; assumption)) ltac:(assumption) ltac:(assumption)) as [Hms2 Hvg2]
    end.
    inversion Hvg2; subst.
    split.
    + rewrite close_record. simpl.
      destruct Hms as [e' [Hms' Heq']].
      assert (Hv' : is_value e').
      { eapply expr_equiv_is_value. apply expr_equiv_sym; eassumption. apply val_to_expr_is_value. }
      destruct Hms2 as [er' [Hms2' Heq2']].
      simpl in Heq2'. inv_record_equiv Heq2'.
      exists (ERecord ((f, e') :: fes')). split.
      * eapply multi_step_trans.
        -- apply ms_record_head. exact Hms'.
        -- apply ms_record_tail; [exact Hv' |].
           rewrite close_record in Hms2'. exact Hms2'.
      * constructor. constructor.
        -- simpl. split; [reflexivity | assumption].
        -- assumption.
    + constructor. constructor; [simpl; assumption | assumption].

  (* E_Field: eval e => VRecord fvs, assoc_lookup f fvs = Some v *)
  - apply_IH.
    split.
    + rewrite close_field.
      destruct Hms as [e' [Hms' Heq']].
      simpl in Heq'. inv_record_equiv Heq'.
      assert (Hv' : Forall (fun p => is_value (snd p)) fes').
      { eapply record_value_transfer_rev; [apply forall_val_to_expr_record_is_value | eassumption]. }
      eapply multi_step_equiv_trans.
      * exists (EField (ERecord fes') f). split.
        -- apply ms_field; eassumption.
        -- apply expr_equiv_refl.
      * exists (match assoc_lookup f fes' with Some v => v | None => EUnit end). split.
        -- eapply MS_Step; [apply S_FieldVal; exact Hv' | apply MS_Refl].
        -- eapply expr_equiv_trans.
           ++ apply record_field_equiv; [exact Hv' | eassumption].
           ++ rewrite (assoc_lookup_map_val_to_expr f fvs v H).
              apply expr_equiv_refl.
    + (* val_good *)
      eapply Forall_assoc_lookup.
      * inversion Hvg; subst. eassumption.
      * eassumption.

  (* E_Fix: result = VFixClos f x body renv *)
  - split; [| constructor; assumption].
    rewrite close_fix.
    exists (EFix f x t1 t2 (close_except2 f x renv body)). split.
    + apply MS_Refl.
    + apply EQ_Fix. apply expr_equiv_refl.

  (* E_AppFix: eval e1 => VFixClos f x body clos_env, eval e2 => v2, eval body => v *)
  - apply_IH. apply_IH.
    inversion Hvg0; subst.
    assert (Hvgfix : val_good (VFixClos f x body clos_env)) by (constructor; assumption).
    assert (Hcv2 : eclosed (val_to_expr v2)) by (apply val_good_eclosed; assumption).
    assert (Hcfix : eclosed (val_to_expr (VFixClos f x body clos_env))) by (apply val_good_eclosed; assumption).
    assert (Havc_clos : all_vals_closed clos_env) by (apply env_good_all_vals_closed; assumption).
    match goal with
    | [ IH3 : pure body -> env_good (ECons x v2 (ECons f (VFixClos f x body clos_env) clos_env)) ->
              all_vals_closed (ECons x v2 (ECons f (VFixClos f x body clos_env) clos_env)) ->
              multi_step_equiv _ (val_to_expr ?vr) /\ val_good ?vr |- _ ] =>
      destruct (IH3 ltac:(assumption)
                    ltac:(constructor; [assumption | constructor; [exact Hvgfix | assumption]])
                    ltac:(simpl; split; [exact Hcv2 | split; [exact Hcfix | exact Havc_clos]])) as [Hms3 Hvg3]
    end.
    split; [| assumption]. rewrite close_app.
    assert (Hkey : subst f (val_to_expr (VFixClos f x body clos_env))
                     (subst x (val_to_expr v2) (close_except2 f x clos_env body)) =
                   close (ECons x v2 (ECons f (VFixClos f x body clos_env) clos_env)) body).
    { simpl. apply subst_close_except2; auto. }
    rewrite <- Hkey in Hms3.
    destruct Hms0 as [e1' [Hms1' Heq1']]. destruct Hms as [e2' [Hms2' Heq2']].
    assert (Hv2' : is_value e2').
    { eapply expr_equiv_is_value. apply expr_equiv_sym; eassumption. apply val_to_expr_is_value. }
    destruct (expr_equiv_fix_inv _ _ _ _ _ _ Heq1') as [t1' [t2' [body' [? Hbody'eq]]]]; subst e1'.
    eapply multi_step_equiv_trans.
    + exists (EApp (EFix f x t1' t2' body') e2'). split.
      * eapply multi_step_trans.
        -- apply ms_app1; eassumption.
        -- apply ms_app2; [constructor | eassumption].
      * apply expr_equiv_refl.
    + eapply multi_step_equiv_trans.
      * exists (subst f (EFix f x t1' t2' body') (subst x e2' body')). split.
        -- eapply MS_Step; [apply S_AppFixBeta; exact Hv2' | apply MS_Refl].
        -- apply expr_equiv_refl.
      * eapply multi_step_equiv_trans; [| exact Hms3].
        exists (subst f (EFix f x t1' t2' body') (subst x e2' body')). split; [apply MS_Refl |].
        apply subst_expr_equiv.
        -- apply EQ_Fix. exact Hbody'eq.
        -- apply subst_expr_equiv; [exact Heq2' | exact Hbody'eq].

  (* E_Construct: eval e => v, result = VConstruct tag v *)
  - apply_IH.
    split.
    + rewrite close_construct.
      destruct Hms as [e' [Hms' Heq']].
      exists (EConstruct tag e' t). split.
      * apply ms_construct. exact Hms'.
      * apply EQ_Construct. assumption.
    + constructor. assumption.

  (* E_Match: eval e => VConstruct tag v, find_branch => body, eval body => v_result *)
  - apply_IH.
    inversion Hvg; subst.
    assert (Hcv : eclosed (val_to_expr v)) by (apply val_good_eclosed; assumption).
    assert (Havc2 : all_vals_closed (ECons x v renv)).
    { simpl. split; [exact Hcv | assumption]. }
    match goal with
    | [ IH2 : pure body -> env_good (ECons x v renv) -> all_vals_closed (ECons x v renv) ->
              multi_step_equiv (close (ECons x v renv) body) (val_to_expr ?vr) /\ val_good ?vr |- _ ] =>
      destruct (IH2 ltac:(eapply find_branch_pure; eassumption) ltac:(constructor; assumption) Havc2) as [Hms2 Hvg2]
    end.
    split; [| assumption]. rewrite close_match.
    assert (Hkey : close (ECons x v renv) body =
                   subst x (val_to_expr v) (close_except x renv body)).
    { simpl. symmetry. apply subst_close_except; assumption. }
    rewrite Hkey in Hms2.
    destruct Hms as [em' [Hms' Heq']].
    simpl in Heq'.
    assert (Hv' : is_value em').
    { eapply expr_equiv_is_value. apply expr_equiv_sym; eassumption.
      simpl. constructor. apply val_to_expr_is_value. }
    (* Step 1: reduce to EMatch em' branches' *)
    eapply multi_step_equiv_trans.
    + exists (EMatch em' (close_match_branches renv branches)). split.
      * apply ms_match. exact Hms'.
      * apply expr_equiv_refl.
    + (* Step 2: invert to get the inner expression *)
      inversion Heq'; subst.
      match goal with
      | [ Hfb : find_branch tag branches = Some (x, body) |- _ ] =>
        pose proof (find_branch_close tag renv branches x body Hfb) as Hfind'
      end.
      eapply multi_step_equiv_trans.
      * match goal with
        | [ Heqc_ : expr_equiv ?ec_ (val_to_expr _) |- _ ] =>
          exists (subst x ec_ (close_except x renv body)); split;
          [ eapply MS_Step; [| apply MS_Refl];
            apply S_MatchBeta; [inversion Hv'; assumption | exact Hfind']
          | apply expr_equiv_refl ]
        end.
      * eapply multi_step_equiv_trans; [| exact Hms2].
        match goal with
        | [ Heqc_ : expr_equiv ?ec_ (val_to_expr _) |- _ ] =>
          exists (subst x ec_ (close_except x renv body)); split; [apply MS_Refl |];
          apply subst_expr_equiv; [assumption | apply expr_equiv_refl]
        end.

  (* E_ArraySet: eval e1 => VArray vs, eval e2 => VInt n, eval e3 => v *)
  - apply_IH. apply_IH. apply_IH.
    split.
    + rewrite close_arrayset.
      (* IH order: Hms=e2(VInt n), Hms0=e3(v), Hms1=e1(VArray vs) *)
      destruct Hms1 as [e1' [Hms1' Heq1']].
      destruct Hms as [e2' [Hms2' Heq2']].
      destruct Hms0 as [e3' [Hms3' Heq3']].
      simpl in Heq1'. inv_array_equiv Heq1'.
      simpl in Heq2'. inversion Heq2'; subst.
      assert (Hv1' : Forall is_value es').
      { eapply forall_value_transfer_rev; [apply forall_val_to_expr_is_value | eassumption]. }
      assert (Hv3' : is_value e3').
      { eapply expr_equiv_is_value. apply expr_equiv_sym; eassumption. apply val_to_expr_is_value. }
      exists (EArray (list_update (Z.to_nat n) e3' es')). split.
      * eapply multi_step_trans. eapply multi_step_trans. eapply multi_step_trans.
        -- apply ms_arrayset1; eassumption.
        -- apply ms_arrayset2; [constructor; assumption | eassumption].
        -- apply ms_arrayset3; [constructor; assumption | constructor | eassumption].
        -- eapply MS_Step; [| apply MS_Refl].
           apply S_ArraySetVal; assumption.
      * simpl. rewrite map_val_to_expr_list_update.
        constructor. apply list_update_equiv; assumption.
    + constructor.
      apply list_update_forall; [inversion Hvg1; subst; eassumption | exact Hvg0].

  (* E_ArrayPush: eval e1 => VArray vs, eval e2 => v *)
  - apply_IH. apply_IH.
    split.
    + rewrite close_arraypush.
      destruct Hms0 as [e1' [Hms1' Heq1']].
      destruct Hms as [e2' [Hms2' Heq2']].
      simpl in Heq1'. inv_array_equiv Heq1'.
      assert (Hv1' : Forall is_value es').
      { eapply forall_value_transfer_rev; [apply forall_val_to_expr_is_value | eassumption]. }
      assert (Hv2' : is_value e2').
      { eapply expr_equiv_is_value. apply expr_equiv_sym; eassumption. apply val_to_expr_is_value. }
      exists (EArray (es' ++ [e2'])). split.
      * eapply multi_step_trans. eapply multi_step_trans.
        -- apply ms_arraypush1; eassumption.
        -- apply ms_arraypush2; [constructor; assumption | eassumption].
        -- eapply MS_Step; [| apply MS_Refl].
           apply S_ArrayPushVal; assumption.
      * simpl. rewrite map_app. simpl.
        constructor. apply app_forall2_equiv; assumption.
    + constructor.
      inversion Hvg0; subst.
      apply Forall_app. split; [eassumption | constructor; [assumption | constructor]].

  (* E_StrIndex: eval e1 => VString s, eval e2 => VInt n *)
  - apply_IH. apply_IH.
    split; [| constructor]. rewrite close_strindex.
    destruct Hms0 as [e1' [Hms1' Heq1']].
    destruct Hms as [e2' [Hms2' Heq2']].
    simpl in Heq1'. inversion Heq1'; subst.
    simpl in Heq2'. inversion Heq2'; subst.
    exists (EString (String.substring (Z.to_nat n) 1 s)). split; [| apply expr_equiv_refl].
    eapply multi_step_trans. eapply multi_step_trans.
    + apply ms_strindex1; eassumption.
    + apply ms_strindex2; [constructor | eassumption].
    + eapply MS_Step; [apply S_StrIndexVal | apply MS_Refl].
Qed.

(** ** Wrapper: the original theorem follows from the generalized one *)

Theorem eval_to_multistep : forall e v,
  pure e ->
  eval ENil e ENil v ->
  multi_step_equiv e (val_to_expr v).
Proof.
  intros e v Hpure Heval.
  assert (env_good ENil) by constructor.
  assert (all_vals_closed ENil) by (simpl; exact I).
  rewrite <- (close_nil e).
  eapply eval_to_multistep_gen; eassumption.
Qed.

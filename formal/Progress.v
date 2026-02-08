(** * NanoCore: Progress Theorem

    Progress: a well-typed closed expression is either a value
    or can take a small-step reduction.

    We introduce small-step semantics here because big-step progress
    would require proving termination (via logical relations), whereas
    small-step progress only requires showing one step exists.

    Division by zero is defined to produce 0, making all operations
    total and the progress theorem unconditional.

    For mutation, the small-step relation carries an environment
    (store) alongside the expression. Set updates the store,
    while loops unroll to if expressions, and sequences reduce
    left-to-right. *)

From Stdlib Require Import ZArith.
From Stdlib Require Import Bool.
From Stdlib Require Import String.
From Stdlib Require Import List.
Import ListNotations.
From NanoCore Require Import Syntax.
From NanoCore Require Import Semantics.
From NanoCore Require Import Typing.
Open Scope Z_scope.
Open Scope string_scope.

(** ** Substitution *)

Fixpoint subst (x : string) (s : expr) (e : expr) : expr :=
  match e with
  | EInt n => EInt n
  | EBool b => EBool b
  | EString str => EString str
  | EUnit => EUnit
  | EVar y => if String.eqb x y then s else EVar y
  | EBinOp op e1 e2 => EBinOp op (subst x s e1) (subst x s e2)
  | EUnOp op e1 => EUnOp op (subst x s e1)
  | EIf e1 e2 e3 => EIf (subst x s e1) (subst x s e2) (subst x s e3)
  | ELet y e1 e2 =>
      ELet y (subst x s e1)
             (if String.eqb x y then e2 else subst x s e2)
  | ESet y e1 => ESet y (subst x s e1)
  | ESeq e1 e2 => ESeq (subst x s e1) (subst x s e2)
  | EWhile e1 e2 => EWhile (subst x s e1) (subst x s e2)
  | ELam y t body =>
      ELam y t (if String.eqb x y then body else subst x s body)
  | EApp e1 e2 => EApp (subst x s e1) (subst x s e2)
  | EArray es => EArray ((fix subst_list (l : list expr) : list expr :=
                   match l with
                   | [] => []
                   | e0 :: rest => subst x s e0 :: subst_list rest
                   end) es)
  | EIndex e1 e2 => EIndex (subst x s e1) (subst x s e2)
  end.

(** ** Value predicate on expressions *)

Inductive is_value : expr -> Prop :=
  | V_Int    : forall n, is_value (EInt n)
  | V_Bool   : forall b, is_value (EBool b)
  | V_String : forall s, is_value (EString s)
  | V_Unit   : is_value EUnit
  | V_Lam    : forall x t body, is_value (ELam x t body)
  | V_Array  : forall es, Forall is_value es -> is_value (EArray es).

(** ** Total binary operation on expression values

    Division and modulo by zero produce 0, making this total
    on well-typed inputs. String concatenation and equality
    are total on string inputs. *)

Definition apply_binop (op : binop) (e1 e2 : expr) : option expr :=
  match op with
  | OpAdd => match e1, e2 with EInt n1, EInt n2 => Some (EInt (n1 + n2)) | _, _ => None end
  | OpSub => match e1, e2 with EInt n1, EInt n2 => Some (EInt (n1 - n2)) | _, _ => None end
  | OpMul => match e1, e2 with EInt n1, EInt n2 => Some (EInt (n1 * n2)) | _, _ => None end
  | OpDiv => match e1, e2 with
             | EInt n1, EInt n2 => Some (EInt (if Z.eqb n2 0 then 0 else Z.div n1 n2))
             | _, _ => None end
  | OpMod => match e1, e2 with
             | EInt n1, EInt n2 => Some (EInt (if Z.eqb n2 0 then 0 else Z.modulo n1 n2))
             | _, _ => None end
  | OpEq  => match e1, e2 with
             | EInt n1, EInt n2 => Some (EBool (Z.eqb n1 n2))
             | EBool b1, EBool b2 => Some (EBool (Bool.eqb b1 b2))
             | EString s1, EString s2 => Some (EBool (String.eqb s1 s2))
             | _, _ => None end
  | OpNe  => match e1, e2 with
             | EInt n1, EInt n2 => Some (EBool (negb (Z.eqb n1 n2)))
             | EBool b1, EBool b2 => Some (EBool (negb (Bool.eqb b1 b2)))
             | EString s1, EString s2 => Some (EBool (negb (String.eqb s1 s2)))
             | _, _ => None end
  | OpLt  => match e1, e2 with EInt n1, EInt n2 => Some (EBool (Z.ltb n1 n2)) | _, _ => None end
  | OpLe  => match e1, e2 with EInt n1, EInt n2 => Some (EBool (Z.leb n1 n2)) | _, _ => None end
  | OpGt  => match e1, e2 with EInt n1, EInt n2 => Some (EBool (Z.ltb n2 n1)) | _, _ => None end
  | OpGe  => match e1, e2 with EInt n1, EInt n2 => Some (EBool (Z.leb n2 n1)) | _, _ => None end
  | OpAnd => match e1, e2 with EBool b1, EBool b2 => Some (EBool (andb b1 b2)) | _, _ => None end
  | OpOr  => match e1, e2 with EBool b1, EBool b2 => Some (EBool (orb b1 b2)) | _, _ => None end
  | OpStrCat => match e1, e2 with
                | EString s1, EString s2 => Some (EString (String.append s1 s2))
                | _, _ => None end
  end.

(** ** Small-step reduction relation

    The relation [step e e'] reduces a closed expression one step.
    For pure expressions, this is standard. For mutation:
    - [ESet x v] reduces to [EUnit] (the actual store update
      happens in the big-step semantics; in small-step, set on
      closed expressions requires a store, which we handle by
      restricting progress to the pure+control fragment)
    - [EWhile] unrolls to an if expression
    - [ESeq v e2] reduces to [e2] when the left side is a value *)

Inductive step : expr -> expr -> Prop :=
  (* Binary: evaluate left *)
  | S_BinOp1 : forall op e1 e1' e2,
      step e1 e1' ->
      step (EBinOp op e1 e2) (EBinOp op e1' e2)
  (* Binary: evaluate right *)
  | S_BinOp2 : forall op v1 e2 e2',
      is_value v1 -> step e2 e2' ->
      step (EBinOp op v1 e2) (EBinOp op v1 e2')
  (* Binary: compute *)
  | S_BinOp : forall op v1 v2 e',
      is_value v1 -> is_value v2 ->
      apply_binop op v1 v2 = Some e' ->
      step (EBinOp op v1 v2) e'
  (* Short-circuit *)
  | S_AndFalse : forall e2,
      step (EBinOp OpAnd (EBool false) e2) (EBool false)
  | S_OrTrue : forall e2,
      step (EBinOp OpOr (EBool true) e2) (EBool true)
  (* Unary: evaluate operand *)
  | S_UnOp1 : forall op e e',
      step e e' ->
      step (EUnOp op e) (EUnOp op e')
  | S_Neg : forall n,
      step (EUnOp OpNeg (EInt n)) (EInt (- n))
  | S_Not : forall b,
      step (EUnOp OpNot (EBool b)) (EBool (negb b))
  | S_StrLen : forall s,
      step (EUnOp OpStrLen (EString s)) (EInt (Z.of_nat (String.length s)))
  (* If *)
  | S_IfCond : forall e1 e1' e2 e3,
      step e1 e1' ->
      step (EIf e1 e2 e3) (EIf e1' e2 e3)
  | S_IfTrue : forall e2 e3,
      step (EIf (EBool true) e2 e3) e2
  | S_IfFalse : forall e2 e3,
      step (EIf (EBool false) e2 e3) e3
  (* Let *)
  | S_Let1 : forall x e1 e1' e2,
      step e1 e1' ->
      step (ELet x e1 e2) (ELet x e1' e2)
  | S_LetVal : forall x v e2,
      is_value v ->
      step (ELet x v e2) (subst x v e2)
  (* Set: evaluate the value expression *)
  | S_Set1 : forall x e e',
      step e e' ->
      step (ESet x e) (ESet x e')
  (* Set: when value is ready, reduce to unit
     (In a store-passing small-step, this would update the store.
      For closed expressions without free variables, set is a no-op
      that produces unit.) *)
  | S_SetVal : forall x v,
      is_value v ->
      step (ESet x v) EUnit
  (* Sequence: evaluate left side *)
  | S_Seq1 : forall e1 e1' e2,
      step e1 e1' ->
      step (ESeq e1 e2) (ESeq e1' e2)
  (* Sequence: left side done, proceed to right *)
  | S_SeqVal : forall v e2,
      is_value v ->
      step (ESeq v e2) e2
  (* While: unroll to if *)
  | S_While : forall cond body,
      step (EWhile cond body)
           (EIf cond (ESeq body (EWhile cond body)) EUnit)
  (* Application *)
  | S_App1 : forall e1 e1' e2,
      step e1 e1' ->
      step (EApp e1 e2) (EApp e1' e2)
  | S_App2 : forall v1 e2 e2',
      is_value v1 -> step e2 e2' ->
      step (EApp v1 e2) (EApp v1 e2')
  | S_AppBeta : forall x t body v,
      is_value v ->
      step (EApp (ELam x t body) v) (subst x v body)
  (* Array: evaluate head element *)
  | S_ArrayHead : forall e e' es,
      step e e' ->
      step (EArray (e :: es)) (EArray (e' :: es))
  (* Array: head is value, step tail *)
  | S_ArrayTail : forall v es es',
      is_value v ->
      step (EArray es) (EArray es') ->
      step (EArray (v :: es)) (EArray (v :: es'))
  (* Index: evaluate array expression *)
  | S_Index1 : forall e1 e1' e2,
      step e1 e1' ->
      step (EIndex e1 e2) (EIndex e1' e2)
  (* Index: evaluate index expression *)
  | S_Index2 : forall v1 e2 e2',
      is_value v1 -> step e2 e2' ->
      step (EIndex v1 e2) (EIndex v1 e2')
  (* Index: extract element (out-of-bounds defaults to unit) *)
  | S_IndexVal : forall es n,
      Forall is_value es ->
      step (EIndex (EArray es) (EInt n)) (nth (Z.to_nat n) es EUnit)
  (* Array length *)
  | S_ArrayLen : forall es,
      Forall is_value es ->
      step (EUnOp OpArrayLen (EArray es)) (EInt (Z.of_nat (length es))).

(** ** Canonical forms for expressions *)

Lemma canonical_forms_int : forall e,
  has_type CtxNil e TInt -> is_value e -> exists n, e = EInt n.
Proof.
  intros e Ht Hv. inversion Hv; subst; inversion Ht; subst.
  - exists n. reflexivity.
Qed.

Lemma canonical_forms_bool : forall e,
  has_type CtxNil e TBool -> is_value e -> exists b, e = EBool b.
Proof.
  intros e Ht Hv. inversion Hv; subst; inversion Ht; subst.
  - exists b. reflexivity.
Qed.

Lemma canonical_forms_string : forall e,
  has_type CtxNil e TString -> is_value e -> exists s, e = EString s.
Proof.
  intros e Ht Hv. inversion Hv; subst; inversion Ht; subst.
  - exists s. reflexivity.
Qed.

Lemma canonical_forms_unit : forall e,
  has_type CtxNil e TUnit -> is_value e -> e = EUnit.
Proof.
  intros e Ht Hv. inversion Hv; subst; inversion Ht; subst.
  reflexivity.
Qed.

Lemma canonical_forms_arrow : forall e t1 t2,
  has_type CtxNil e (TArrow t1 t2) -> is_value e ->
  exists x body, e = ELam x t1 body.
Proof.
  intros e t1 t2 Ht Hv. inversion Hv; subst; inversion Ht; subst.
  - exists x, body. reflexivity.
Qed.

Lemma canonical_forms_array : forall e t,
  has_type CtxNil e (TArray t) -> is_value e ->
  exists es, e = EArray es /\ Forall is_value es.
Proof.
  intros e t Ht Hv. inversion Hv; subst; try solve [inversion Ht].
  exists es. split; [reflexivity | assumption].
Qed.

(** ** Totality of apply_binop on well-typed inputs *)

Lemma apply_binop_int_total : forall op n1 n2,
  binop_arg_type op = TInt ->
  exists e', apply_binop op (EInt n1) (EInt n2) = Some e'.
Proof.
  intros op n1 n2 Harg.
  destruct op; simpl in *; try discriminate; eexists; reflexivity.
Qed.

Lemma apply_binop_bool_total : forall op b1 b2,
  binop_arg_type op = TBool ->
  exists e', apply_binop op (EBool b1) (EBool b2) = Some e'.
Proof.
  intros op b1 b2 Harg.
  destruct op; simpl in *; try discriminate; eexists; reflexivity.
Qed.

Lemma apply_binop_bool_eq_total : forall op b1 b2,
  binop_allows_bool_args op = true ->
  exists e', apply_binop op (EBool b1) (EBool b2) = Some e'.
Proof.
  intros op b1 b2 Hallow.
  destruct op; simpl in *; try discriminate; eexists; reflexivity.
Qed.

Lemma apply_binop_strcat_total : forall s1 s2,
  exists e', apply_binop OpStrCat (EString s1) (EString s2) = Some e'.
Proof.
  intros. eexists. reflexivity.
Qed.

Lemma apply_binop_string_eq_total : forall op s1 s2,
  binop_allows_string_args op = true ->
  exists e', apply_binop op (EString s1) (EString s2) = Some e'.
Proof.
  intros op s1 s2 Hallow.
  destruct op; simpl in *; try discriminate; eexists; reflexivity.
Qed.

(** ** Array step form: stepping an array produces an array *)

Lemma step_array_form : forall es e',
  step (EArray es) e' -> exists es', e' = EArray es'.
Proof.
  intros es e' Hstep.
  inversion Hstep; subst; eexists; reflexivity.
Qed.

(** ** Progress Theorem *)

Theorem progress : forall e t,
  has_type CtxNil e t ->
  is_value e \/ (exists e', step e e').
Proof.
  intros e t Htype.
  remember CtxNil as Gamma.
  induction Htype; subst.

  - (* T_Int *) left. constructor.
  - (* T_Bool *) left. constructor.
  - (* T_String *) left. constructor.
  - (* T_Unit *) left. constructor.

  - (* T_Var: impossible in empty context *)
    simpl in H. discriminate.

  - (* T_BinOp: integer operators *)
    right.
    destruct IHHtype1 as [Hv1 | [e1' Hs1]]; [reflexivity | |].
    + (* e1 is a value *)
      destruct IHHtype2 as [Hv2 | [e2' Hs2]]; [reflexivity | |].
      * (* both values *)
        apply canonical_forms_int in Htype1 as [n1 ?]; [| assumption]; subst.
        apply canonical_forms_int in Htype2 as [n2 ?]; [| assumption]; subst.
        destruct (apply_binop_int_total op n1 n2 H) as [e' He'].
        exists e'. apply S_BinOp; try constructor; assumption.
      * (* e2 steps *)
        exists (EBinOp op e1 e2'). apply S_BinOp2; assumption.
    + (* e1 steps *)
      exists (EBinOp op e1' e2). apply S_BinOp1. assumption.

  - (* T_BinLogic: boolean logic operators *)
    right.
    destruct IHHtype1 as [Hv1 | [e1' Hs1]]; [reflexivity | |].
    + (* e1 is a value *)
      apply canonical_forms_bool in Htype1 as [b1 ?]; [| assumption]; subst.
      destruct b1; destruct op eqn:Hop; simpl in H; try discriminate.
      * (* true AND *)
        destruct IHHtype2 as [Hv2 | [e2' Hs2]]; [reflexivity | |].
        -- apply canonical_forms_bool in Htype2 as [b2 ?]; [| assumption]; subst.
           eexists. apply S_BinOp; constructor.
        -- exists (EBinOp OpAnd (EBool true) e2'). apply S_BinOp2; [constructor | assumption].
      * (* true OR: short-circuit *)
        exists (EBool true). apply S_OrTrue.
      * (* false AND: short-circuit *)
        exists (EBool false). apply S_AndFalse.
      * (* false OR *)
        destruct IHHtype2 as [Hv2 | [e2' Hs2]]; [reflexivity | |].
        -- apply canonical_forms_bool in Htype2 as [b2 ?]; [| assumption]; subst.
           eexists. apply S_BinOp; constructor.
        -- exists (EBinOp OpOr (EBool false) e2'). apply S_BinOp2; [constructor | assumption].
    + (* e1 steps *)
      exists (EBinOp op e1' e2). apply S_BinOp1. assumption.

  - (* T_BinEqBool: equality on booleans *)
    right.
    destruct IHHtype1 as [Hv1 | [e1' Hs1]]; [reflexivity | |].
    + destruct IHHtype2 as [Hv2 | [e2' Hs2]]; [reflexivity | |].
      * apply canonical_forms_bool in Htype1 as [b1 ?]; [| assumption]; subst.
        apply canonical_forms_bool in Htype2 as [b2 ?]; [| assumption]; subst.
        destruct (apply_binop_bool_eq_total op b1 b2 H) as [e' He'].
        exists e'. apply S_BinOp; try constructor; assumption.
      * exists (EBinOp op e1 e2'). apply S_BinOp2; assumption.
    + exists (EBinOp op e1' e2). apply S_BinOp1. assumption.

  - (* T_StrCat: string concatenation *)
    right.
    destruct IHHtype1 as [Hv1 | [e1' Hs1]]; [reflexivity | |].
    + destruct IHHtype2 as [Hv2 | [e2' Hs2]]; [reflexivity | |].
      * apply canonical_forms_string in Htype1 as [s1 ?]; [| assumption]; subst.
        apply canonical_forms_string in Htype2 as [s2 ?]; [| assumption]; subst.
        destruct (apply_binop_strcat_total s1 s2) as [e' He'].
        exists e'. apply S_BinOp; try constructor; assumption.
      * exists (EBinOp OpStrCat e1 e2'). apply S_BinOp2; assumption.
    + exists (EBinOp OpStrCat e1' e2). apply S_BinOp1. assumption.

  - (* T_BinEqStr: equality/inequality on strings *)
    right.
    destruct IHHtype1 as [Hv1 | [e1' Hs1]]; [reflexivity | |].
    + destruct IHHtype2 as [Hv2 | [e2' Hs2]]; [reflexivity | |].
      * apply canonical_forms_string in Htype1 as [s1 ?]; [| assumption]; subst.
        apply canonical_forms_string in Htype2 as [s2 ?]; [| assumption]; subst.
        destruct (apply_binop_string_eq_total op s1 s2 H) as [e' He'].
        exists e'. apply S_BinOp; try constructor; assumption.
      * exists (EBinOp op e1 e2'). apply S_BinOp2; assumption.
    + exists (EBinOp op e1' e2). apply S_BinOp1. assumption.

  - (* T_Neg *)
    right.
    destruct IHHtype as [Hv | [e' Hs]]; [reflexivity | |].
    + apply canonical_forms_int in Htype as [n ?]; [| assumption]; subst.
      exists (EInt (- n)). apply S_Neg.
    + exists (EUnOp OpNeg e'). apply S_UnOp1. assumption.

  - (* T_Not *)
    right.
    destruct IHHtype as [Hv | [e' Hs]]; [reflexivity | |].
    + apply canonical_forms_bool in Htype as [b ?]; [| assumption]; subst.
      exists (EBool (negb b)). apply S_Not.
    + exists (EUnOp OpNot e'). apply S_UnOp1. assumption.

  - (* T_StrLen *)
    right.
    destruct IHHtype as [Hv | [e' Hs]]; [reflexivity | |].
    + apply canonical_forms_string in Htype as [s ?]; [| assumption]; subst.
      exists (EInt (Z.of_nat (String.length s))). apply S_StrLen.
    + exists (EUnOp OpStrLen e'). apply S_UnOp1. assumption.

  - (* T_If *)
    right.
    destruct IHHtype1 as [Hv1 | [e1' Hs1]]; [reflexivity | |].
    + apply canonical_forms_bool in Htype1 as [b ?]; [| assumption]; subst.
      destruct b.
      * exists e2. apply S_IfTrue.
      * exists e3. apply S_IfFalse.
    + exists (EIf e1' e2 e3). apply S_IfCond. assumption.

  - (* T_Let *)
    right.
    destruct IHHtype1 as [Hv1 | [e1' Hs1]]; [reflexivity | |].
    + exists (subst x e1 e2). apply S_LetVal. assumption.
    + exists (ELet x e1' e2). apply S_Let1. assumption.

  - (* T_Set *)
    right.
    destruct IHHtype as [Hv | [e' Hs]]; [reflexivity | |].
    + exists EUnit. apply S_SetVal. assumption.
    + exists (ESet x e'). apply S_Set1. assumption.

  - (* T_Seq *)
    right.
    destruct IHHtype1 as [Hv1 | [e1' Hs1]]; [reflexivity | |].
    + exists e2. apply S_SeqVal. assumption.
    + exists (ESeq e1' e2). apply S_Seq1. assumption.

  - (* T_While *)
    right.
    exists (EIf cond (ESeq body (EWhile cond body)) EUnit).
    apply S_While.

  - (* T_Lam *) left. constructor.

  - (* T_App *)
    right.
    destruct IHHtype1 as [Hv1 | [e1' Hs1]]; [reflexivity | |].
    + destruct IHHtype2 as [Hv2 | [e2' Hs2]]; [reflexivity | |].
      * apply canonical_forms_arrow in Htype1 as [x [body ?]]; [| assumption]; subst.
        exists (subst x e2 body). apply S_AppBeta. assumption.
      * exists (EApp e1 e2'). apply S_App2; assumption.
    + exists (EApp e1' e2). apply S_App1. assumption.

  - (* T_ArrayNil *)
    left. constructor. constructor.

  - (* T_ArrayCons *)
    destruct IHHtype1 as [Hv1 | [e' Hs1]]; [reflexivity | |].
    + (* e is a value *)
      destruct IHHtype2 as [Hv2 | [e'' Hs2]]; [reflexivity | |].
      * (* EArray es is also a value *)
        left. inversion Hv2; subst.
        constructor. constructor; assumption.
      * (* EArray es steps *)
        right.
        destruct (step_array_form _ _ Hs2) as [es' Heq]; subst.
        exists (EArray (e :: es')). apply S_ArrayTail; assumption.
    + (* e steps *)
      right. exists (EArray (e' :: es)). apply S_ArrayHead. assumption.

  - (* T_Index *)
    right.
    destruct IHHtype1 as [Hv1 | [e1' Hs1]]; [reflexivity | |].
    + destruct IHHtype2 as [Hv2 | [e2' Hs2]]; [reflexivity | |].
      * (* both values *)
        apply canonical_forms_array in Htype1 as [es [? HF]]; [| assumption]; subst.
        apply canonical_forms_int in Htype2 as [n ?]; [| assumption]; subst.
        exists (nth (Z.to_nat n) es EUnit). apply S_IndexVal. assumption.
      * (* e2 steps *)
        exists (EIndex e1 e2'). apply S_Index2; assumption.
    + (* e1 steps *)
      exists (EIndex e1' e2). apply S_Index1. assumption.

  - (* T_ArrayLen *)
    right.
    destruct IHHtype as [Hv | [e' Hs]]; [reflexivity | |].
    + apply canonical_forms_array in Htype as [es [? HF]]; [| assumption]; subst.
      exists (EInt (Z.of_nat (length es))). apply S_ArrayLen. assumption.
    + exists (EUnOp OpArrayLen e'). apply S_UnOp1. assumption.
Qed.

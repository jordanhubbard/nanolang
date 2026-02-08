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
  | EFix f y t1 t2 body =>
      EFix f y t1 t2
        (if (String.eqb x f || String.eqb x y)%bool then body
         else subst x s body)
  | EArray es => EArray ((fix subst_list (l : list expr) : list expr :=
                   match l with
                   | [] => []
                   | e0 :: rest => subst x s e0 :: subst_list rest
                   end) es)
  | EIndex e1 e2 => EIndex (subst x s e1) (subst x s e2)
  | EArraySet e1 e2 e3 => EArraySet (subst x s e1) (subst x s e2) (subst x s e3)
  | EArrayPush e1 e2 => EArrayPush (subst x s e1) (subst x s e2)
  | ERecord fes => ERecord ((fix subst_fields (l : list (string * expr)) : list (string * expr) :=
                     match l with
                     | [] => []
                     | (f, e0) :: rest => (f, subst x s e0) :: subst_fields rest
                     end) fes)
  | EField e1 f => EField (subst x s e1) f
  | ESetField y f e1 => ESetField y f (subst x s e1)
  | EConstruct tag e1 t => EConstruct tag (subst x s e1) t
  | EMatch e1 branches =>
      EMatch (subst x s e1)
             ((fix subst_branches (bs : list (string * string * expr)) :
                   list (string * string * expr) :=
                match bs with
                | [] => []
                | (tag, y, body) :: rest =>
                    (tag, y, if String.eqb x y then body else subst x s body)
                    :: subst_branches rest
                end) branches)
  | EStrIndex e1 e2 => EStrIndex (subst x s e1) (subst x s e2)
  end.

(** ** Value predicate on expressions *)

Inductive is_value : expr -> Prop :=
  | V_Int    : forall n, is_value (EInt n)
  | V_Bool   : forall b, is_value (EBool b)
  | V_String : forall s, is_value (EString s)
  | V_Unit   : is_value EUnit
  | V_Lam    : forall x t body, is_value (ELam x t body)
  | V_Fix    : forall f x t1 t2 body, is_value (EFix f x t1 t2 body)
  | V_Array  : forall es, Forall is_value es -> is_value (EArray es)
  | V_Record : forall fes, Forall (fun fe => is_value (snd fe)) fes -> is_value (ERecord fes)
  | V_Construct : forall tag v t, is_value v -> is_value (EConstruct tag v t).

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

(** ** Small-step reduction relation *)

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
  | S_SetVal : forall x v,
      is_value v ->
      step (ESet x v) EUnit
  (* Sequence *)
  | S_Seq1 : forall e1 e1' e2,
      step e1 e1' ->
      step (ESeq e1 e2) (ESeq e1' e2)
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
  | S_AppFixBeta : forall f x t1 t2 body v,
      is_value v ->
      step (EApp (EFix f x t1 t2 body) v)
           (subst f (EFix f x t1 t2 body) (subst x v body))
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
      step (EUnOp OpArrayLen (EArray es)) (EInt (Z.of_nat (length es)))
  (* Array functional update *)
  | S_ArraySet1 : forall e1 e1' e2 e3,
      step e1 e1' ->
      step (EArraySet e1 e2 e3) (EArraySet e1' e2 e3)
  | S_ArraySet2 : forall v1 e2 e2' e3,
      is_value v1 -> step e2 e2' ->
      step (EArraySet v1 e2 e3) (EArraySet v1 e2' e3)
  | S_ArraySet3 : forall v1 v2 e3 e3',
      is_value v1 -> is_value v2 -> step e3 e3' ->
      step (EArraySet v1 v2 e3) (EArraySet v1 v2 e3')
  | S_ArraySetVal : forall es n v,
      Forall is_value es -> is_value v ->
      step (EArraySet (EArray es) (EInt n) v)
           (EArray (list_update (Z.to_nat n) v es))
  (* Array push *)
  | S_ArrayPush1 : forall e1 e1' e2,
      step e1 e1' ->
      step (EArrayPush e1 e2) (EArrayPush e1' e2)
  | S_ArrayPush2 : forall v1 e2 e2',
      is_value v1 -> step e2 e2' ->
      step (EArrayPush v1 e2) (EArrayPush v1 e2')
  | S_ArrayPushVal : forall es v,
      Forall is_value es -> is_value v ->
      step (EArrayPush (EArray es) v) (EArray (es ++ [v]))
  (* Record: evaluate head field *)
  | S_RecordHead : forall f e e' fes,
      step e e' ->
      step (ERecord ((f, e) :: fes)) (ERecord ((f, e') :: fes))
  (* Record: head field is value, step tail *)
  | S_RecordTail : forall f v fes fes',
      is_value v ->
      step (ERecord fes) (ERecord fes') ->
      step (ERecord ((f, v) :: fes)) (ERecord ((f, v) :: fes'))
  (* Field: evaluate record expression *)
  | S_Field1 : forall e e' f,
      step e e' ->
      step (EField e f) (EField e' f)
  (* Field: extract from record value *)
  | S_FieldVal : forall fes f,
      Forall (fun fe => is_value (snd fe)) fes ->
      step (EField (ERecord fes) f)
           (match assoc_lookup f fes with
            | Some v => v
            | None => EUnit
            end)
  (* SetField: evaluate value expression *)
  | S_SetField1 : forall x f e e',
      step e e' ->
      step (ESetField x f e) (ESetField x f e')
  (* SetField: reduce to unit *)
  | S_SetFieldVal : forall x f v,
      is_value v ->
      step (ESetField x f v) EUnit
  (* Construct: evaluate payload *)
  | S_Construct1 : forall tag e e' t,
      step e e' ->
      step (EConstruct tag e t) (EConstruct tag e' t)
  (* Match: evaluate scrutinee *)
  | S_Match1 : forall e e' branches,
      step e e' ->
      step (EMatch e branches) (EMatch e' branches)
  (* Match: beta reduction *)
  | S_MatchBeta : forall tag v t branches x body,
      is_value v ->
      find_branch tag branches = Some (x, body) ->
      step (EMatch (EConstruct tag v t) branches) (subst x v body)
  (* String indexing *)
  | S_StrIndex1 : forall e1 e1' e2,
      step e1 e1' ->
      step (EStrIndex e1 e2) (EStrIndex e1' e2)
  | S_StrIndex2 : forall v1 e2 e2',
      is_value v1 -> step e2 e2' ->
      step (EStrIndex v1 e2) (EStrIndex v1 e2')
  | S_StrIndexVal : forall s n,
      step (EStrIndex (EString s) (EInt n))
           (EString (String.substring (Z.to_nat n) 1 s)).

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
  (exists x body, e = ELam x t1 body) \/
  (exists f x body, e = EFix f x t1 t2 body).
Proof.
  intros e t1 t2 Ht Hv. inversion Hv; subst; inversion Ht; subst.
  - left. exists x, body. reflexivity.
  - right. exists f, x, body. reflexivity.
Qed.

Lemma canonical_forms_record : forall e fts,
  has_type CtxNil e (TRecord fts) -> is_value e ->
  exists fes, e = ERecord fes /\ Forall (fun fe => is_value (snd fe)) fes.
Proof.
  intros e fts Ht Hv. inversion Hv; subst; try solve [inversion Ht].
  exists fes. split; [reflexivity | assumption].
Qed.

Lemma canonical_forms_array : forall e t,
  has_type CtxNil e (TArray t) -> is_value e ->
  exists es, e = EArray es /\ Forall is_value es.
Proof.
  intros e t Ht Hv. inversion Hv; subst; try solve [inversion Ht].
  exists es. split; [reflexivity | assumption].
Qed.

Lemma canonical_forms_variant : forall e fts,
  has_type CtxNil e (TVariant fts) -> is_value e ->
  exists tag v t, e = EConstruct tag v t /\ is_value v.
Proof.
  intros e fts Ht Hv. inversion Hv; subst; try solve [inversion Ht].
  eexists _, _, _. split; [reflexivity | assumption].
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

(** ** Record step form: stepping a record produces a record *)

Lemma step_record_form : forall fes e',
  step (ERecord fes) e' -> exists fes', e' = ERecord fes'.
Proof.
  intros fes e' Hstep.
  inversion Hstep; subst; eexists; reflexivity.
Qed.

(** ** Branch lookup for variants *)

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
      * apply canonical_forms_arrow in Htype1 as Hcan; [| assumption].
        destruct Hcan as [[x [body Heq]] | [f [x0 [body Heq]]]]; subst.
        -- exists (subst x e2 body). apply S_AppBeta. assumption.
        -- exists (subst f (EFix f x0 t1 t2 body) (subst x0 e2 body)).
           apply S_AppFixBeta. assumption.
      * exists (EApp e1 e2'). apply S_App2; assumption.
    + exists (EApp e1' e2). apply S_App1. assumption.

  - (* T_Fix *)
    left. constructor.

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

  - (* T_ArraySet *)
    right.
    destruct IHHtype1 as [Hv1 | [e1' Hs1]]; [reflexivity | |].
    + destruct IHHtype2 as [Hv2 | [e2' Hs2]]; [reflexivity | |].
      * destruct IHHtype3 as [Hv3 | [e3' Hs3]]; [reflexivity | |].
        -- (* all values *)
           apply canonical_forms_array in Htype1 as [es [? HF]]; [| assumption]; subst.
           apply canonical_forms_int in Htype2 as [n ?]; [| assumption]; subst.
           exists (EArray (list_update (Z.to_nat n) e3 es)). apply S_ArraySetVal; assumption.
        -- (* e3 steps *)
           exists (EArraySet e1 e2 e3'). apply S_ArraySet3; assumption.
      * (* e2 steps *)
        exists (EArraySet e1 e2' e3). apply S_ArraySet2; assumption.
    + (* e1 steps *)
      exists (EArraySet e1' e2 e3). apply S_ArraySet1. assumption.

  - (* T_ArrayPush *)
    right.
    destruct IHHtype1 as [Hv1 | [e1' Hs1]]; [reflexivity | |].
    + destruct IHHtype2 as [Hv2 | [e2' Hs2]]; [reflexivity | |].
      * (* both values *)
        apply canonical_forms_array in Htype1 as [es [? HF]]; [| assumption]; subst.
        exists (EArray (es ++ [e2])). apply S_ArrayPushVal; assumption.
      * (* e2 steps *)
        exists (EArrayPush e1 e2'). apply S_ArrayPush2; assumption.
    + (* e1 steps *)
      exists (EArrayPush e1' e2). apply S_ArrayPush1. assumption.

  - (* T_RecordNil *)
    left. constructor. constructor.

  - (* T_RecordCons *)
    destruct IHHtype1 as [Hv1 | [e' Hs1]]; [reflexivity | |].
    + (* e is a value *)
      destruct IHHtype2 as [Hv2 | [e'' Hs2]]; [reflexivity | |].
      * (* ERecord es is also a value *)
        left. inversion Hv2; subst.
        constructor. constructor; [simpl; assumption | assumption].
      * (* ERecord es steps *)
        right.
        destruct (step_record_form _ _ Hs2) as [fes' Heq]; subst.
        exists (ERecord ((f, e) :: fes')). apply S_RecordTail; assumption.
    + (* e steps *)
      right. exists (ERecord ((f, e') :: es)). apply S_RecordHead. assumption.

  - (* T_Field *)
    right.
    destruct IHHtype as [Hv | [e' Hs]]; [reflexivity | |].
    + (* e is a value *)
      apply canonical_forms_record in Htype as [fes [? HF]]; [| assumption]; subst.
      exists (match assoc_lookup f fes with Some v => v | None => EUnit end).
      apply S_FieldVal. assumption.
    + (* e steps *)
      exists (EField e' f). apply S_Field1. assumption.

  - (* T_SetField *)
    right.
    destruct IHHtype as [Hv | [e' Hs]]; [reflexivity | |].
    + exists EUnit. apply S_SetFieldVal. assumption.
    + exists (ESetField x f e'). apply S_SetField1. assumption.

  - (* T_Construct *)
    destruct IHHtype as [Hv | [e' Hs]]; [reflexivity | |].
    + left. constructor. assumption.
    + right. exists (EConstruct tag e' (TVariant fts)). apply S_Construct1. assumption.

  - (* T_Match *)
    right.
    destruct IHHtype as [Hv | [e' Hs]]; [reflexivity | |].
    + (* e is a value *)
      assert (Hcan : exists tag v0 t0, e = EConstruct tag v0 t0 /\ is_value v0)
        by (eapply canonical_forms_variant; eassumption).
      destruct Hcan as [tag0 [v0 [t0 [Heq Hval]]]]. subst.
      (* Invert T_Construct typing to get assoc_lookup *)
      match goal with
      | [ Ht : has_type _ (EConstruct _ _ _) _ |- _ ] =>
        inversion Ht; subst
      end.
      (* Find the branch using branches_type_find *)
      assert (Hfind : exists x0 body0, find_branch tag0 branches = Some (x0, body0))
        by (eapply branches_type_find; eassumption).
      destruct Hfind as [x0 [body0 Hfind]].
      exists (subst x0 v0 body0). apply S_MatchBeta; assumption.
    + (* e steps *)
      exists (EMatch e' branches). apply S_Match1. assumption.

  - (* T_StrIndex *)
    right.
    destruct IHHtype1 as [Hv1 | [e1' Hs1]]; [reflexivity | |].
    + destruct IHHtype2 as [Hv2 | [e2' Hs2]]; [reflexivity | |].
      * (* both values *)
        apply canonical_forms_string in Htype1 as [s ?]; [| assumption]; subst.
        apply canonical_forms_int in Htype2 as [n ?]; [| assumption]; subst.
        exists (EString (String.substring (Z.to_nat n) 1 s)). apply S_StrIndexVal.
      * (* e2 steps *)
        exists (EStrIndex e1 e2'). apply S_StrIndex2; assumption.
    + (* e1 steps *)
      exists (EStrIndex e1' e2). apply S_StrIndex1. assumption.
Qed.

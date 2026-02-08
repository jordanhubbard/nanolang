(** * NanoCore: Type System

    This file defines the typing rules for NanoCore.
    The typing judgment is:
      has_type ctx e t  ===  "expression e has type t in typing context ctx"

    The type system ensures:
    - Arithmetic operators require integer operands and produce integers
    - Comparison operators require integer operands and produce booleans
    - Logical operators require boolean operands and produce booleans
    - If-then-else requires boolean condition and matching branch types
    - Function application checks argument against parameter type
    - Set requires variable in scope and value of matching type
    - While requires boolean condition and unit-typed body
*)

From Stdlib Require Import ZArith.
From Stdlib Require Import String.
From Stdlib Require Import List.
Import ListNotations.
From NanoCore Require Import Syntax.
Open Scope string_scope.

(** ** Typing context

    A typing context maps variable names to their types.
    Uses a simple association list, like environments. *)

Inductive ctx : Type :=
  | CtxNil  : ctx
  | CtxCons : string -> ty -> ctx -> ctx.

(** Context lookup *)

Fixpoint ctx_lookup (x : string) (c : ctx) : option ty :=
  match c with
  | CtxNil => None
  | CtxCons y t rest =>
    if String.eqb x y then Some t
    else ctx_lookup x rest
  end.

(** ** Operator typing

    Classify what types a binary operator expects and produces. *)

(** Type of a binary operator's operands and result *)

Definition binop_arg_type (op : binop) : ty :=
  match op with
  | OpAdd | OpSub | OpMul | OpDiv | OpMod => TInt
  | OpEq | OpNe | OpLt | OpLe | OpGt | OpGe => TInt
  | OpAnd | OpOr => TBool
  | OpStrCat => TString
  end.

Definition binop_res_type (op : binop) : ty :=
  match op with
  | OpAdd | OpSub | OpMul | OpDiv | OpMod => TInt
  | OpEq | OpNe | OpLt | OpLe | OpGt | OpGe => TBool
  | OpAnd | OpOr => TBool
  | OpStrCat => TString
  end.

(** For equality on booleans, we need a separate rule *)

Definition binop_allows_bool_args (op : binop) : bool :=
  match op with
  | OpEq | OpNe => true
  | _ => false
  end.

(** For equality on strings, we need a separate rule *)

Definition binop_allows_string_args (op : binop) : bool :=
  match op with
  | OpEq | OpNe => true
  | _ => false
  end.

(** ** Typing relation *)

Inductive has_type : ctx -> expr -> ty -> Prop :=

  (** Integer literal *)
  | T_Int : forall ctx n,
      has_type ctx (EInt n) TInt

  (** Boolean literal *)
  | T_Bool : forall ctx b,
      has_type ctx (EBool b) TBool

  (** String literal *)
  | T_String : forall ctx s,
      has_type ctx (EString s) TString

  (** Unit literal *)
  | T_Unit : forall ctx,
      has_type ctx EUnit TUnit

  (** Variable *)
  | T_Var : forall ctx x t,
      ctx_lookup x ctx = Some t ->
      has_type ctx (EVar x) t

  (** Binary operator on integers *)
  | T_BinOp : forall ctx op e1 e2,
      binop_arg_type op = TInt ->
      has_type ctx e1 TInt ->
      has_type ctx e2 TInt ->
      has_type ctx (EBinOp op e1 e2) (binop_res_type op)

  (** Logical operators on booleans *)
  | T_BinLogic : forall ctx op e1 e2,
      binop_arg_type op = TBool ->
      has_type ctx e1 TBool ->
      has_type ctx e2 TBool ->
      has_type ctx (EBinOp op e1 e2) TBool

  (** Equality/inequality on booleans *)
  | T_BinEqBool : forall ctx op e1 e2,
      binop_allows_bool_args op = true ->
      has_type ctx e1 TBool ->
      has_type ctx e2 TBool ->
      has_type ctx (EBinOp op e1 e2) TBool

  (** String concatenation *)
  | T_StrCat : forall ctx e1 e2,
      has_type ctx e1 TString ->
      has_type ctx e2 TString ->
      has_type ctx (EBinOp OpStrCat e1 e2) TString

  (** Equality/inequality on strings *)
  | T_BinEqStr : forall ctx op e1 e2,
      binop_allows_string_args op = true ->
      has_type ctx e1 TString ->
      has_type ctx e2 TString ->
      has_type ctx (EBinOp op e1 e2) TBool

  (** Unary negation *)
  | T_Neg : forall ctx e,
      has_type ctx e TInt ->
      has_type ctx (EUnOp OpNeg e) TInt

  (** Logical not *)
  | T_Not : forall ctx e,
      has_type ctx e TBool ->
      has_type ctx (EUnOp OpNot e) TBool

  (** String length *)
  | T_StrLen : forall ctx e,
      has_type ctx e TString ->
      has_type ctx (EUnOp OpStrLen e) TInt

  (** If-then-else *)
  | T_If : forall ctx e1 e2 e3 t,
      has_type ctx e1 TBool ->
      has_type ctx e2 t ->
      has_type ctx e3 t ->
      has_type ctx (EIf e1 e2 e3) t

  (** Let binding *)
  | T_Let : forall ctx x e1 e2 t1 t2,
      has_type ctx e1 t1 ->
      has_type (CtxCons x t1 ctx) e2 t2 ->
      has_type ctx (ELet x e1 e2) t2

  (** Set: mutable variable assignment *)
  | T_Set : forall ctx x e t,
      ctx_lookup x ctx = Some t ->
      has_type ctx e t ->
      has_type ctx (ESet x e) TUnit

  (** Sequence *)
  | T_Seq : forall ctx e1 e2 t1 t2,
      has_type ctx e1 t1 ->
      has_type ctx e2 t2 ->
      has_type ctx (ESeq e1 e2) t2

  (** While loop *)
  | T_While : forall ctx cond body,
      has_type ctx cond TBool ->
      has_type ctx body TUnit ->
      has_type ctx (EWhile cond body) TUnit

  (** Lambda abstraction *)
  | T_Lam : forall ctx x t1 body t2,
      has_type (CtxCons x t1 ctx) body t2 ->
      has_type ctx (ELam x t1 body) (TArrow t1 t2)

  (** Function application *)
  | T_App : forall ctx e1 e2 t1 t2,
      has_type ctx e1 (TArrow t1 t2) ->
      has_type ctx e2 t1 ->
      has_type ctx (EApp e1 e2) t2

  (** Empty array literal: [] has type array<T> for any T *)
  | T_ArrayNil : forall ctx t,
      has_type ctx (EArray []) (TArray t)

  (** Non-empty array literal: [e, ...es] *)
  | T_ArrayCons : forall ctx e es t,
      has_type ctx e t ->
      has_type ctx (EArray es) (TArray t) ->
      has_type ctx (EArray (e :: es)) (TArray t)

  (** Array indexing: (at arr i) *)
  | T_Index : forall ctx e1 e2 t,
      has_type ctx e1 (TArray t) ->
      has_type ctx e2 TInt ->
      has_type ctx (EIndex e1 e2) t

  (** Array length *)
  | T_ArrayLen : forall ctx e t,
      has_type ctx e (TArray t) ->
      has_type ctx (EUnOp OpArrayLen e) TInt

  (** Empty record literal *)
  | T_RecordNil : forall ctx,
      has_type ctx (ERecord []) (TRecord [])

  (** Non-empty record literal *)
  | T_RecordCons : forall ctx f e es t fts,
      has_type ctx e t ->
      has_type ctx (ERecord es) (TRecord fts) ->
      has_type ctx (ERecord ((f, e) :: es)) (TRecord ((f, t) :: fts))

  (** Record field access *)
  | T_Field : forall ctx e f fts t,
      has_type ctx e (TRecord fts) ->
      assoc_lookup f fts = Some t ->
      has_type ctx (EField e f) t.

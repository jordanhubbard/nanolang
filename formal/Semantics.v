(** * NanoCore: Big-Step Operational Semantics

    This file defines the big-step (natural) evaluation semantics for NanoCore.
    The evaluation relation is:
      eval env e v  ===  "expression e evaluates to value v in environment env"

    Key properties:
    - Eager evaluation (call-by-value)
    - Deterministic (each expression has at most one value)
    - Pure (no side effects in this subset)
    - Lexical scoping via closures
*)

From Stdlib Require Import ZArith.
From Stdlib Require Import Bool.
From Stdlib Require Import String.
From NanoCore Require Import Syntax.
Open Scope Z_scope.

(** ** Arithmetic operator application *)

Definition eval_arith_binop (op : binop) (n1 n2 : Z) : option val :=
  match op with
  | OpAdd => Some (VInt (n1 + n2))
  | OpSub => Some (VInt (n1 - n2))
  | OpMul => Some (VInt (n1 * n2))
  | OpDiv => if Z.eqb n2 0 then None else Some (VInt (Z.div n1 n2))
  | OpMod => if Z.eqb n2 0 then None else Some (VInt (Z.modulo n1 n2))
  | _ => None
  end.

(** ** Comparison operator application *)

Definition eval_cmp_binop (op : binop) (n1 n2 : Z) : option val :=
  match op with
  | OpEq => Some (VBool (Z.eqb n1 n2))
  | OpNe => Some (VBool (negb (Z.eqb n1 n2)))
  | OpLt => Some (VBool (Z.ltb n1 n2))
  | OpLe => Some (VBool (Z.leb n1 n2))
  | OpGt => Some (VBool (Z.ltb n2 n1))
  | OpGe => Some (VBool (Z.leb n2 n1))
  | _ => None
  end.

(** ** Classify operator kinds *)

Definition is_arith_op (op : binop) : bool :=
  match op with
  | OpAdd | OpSub | OpMul | OpDiv | OpMod => true
  | _ => false
  end.

Definition is_cmp_op (op : binop) : bool :=
  match op with
  | OpEq | OpNe | OpLt | OpLe | OpGt | OpGe => true
  | _ => false
  end.

Definition is_logic_op (op : binop) : bool :=
  match op with
  | OpAnd | OpOr => true
  | _ => false
  end.

(** ** Big-step evaluation relation

    [eval env e v] means that expression [e] evaluates to value [v]
    in environment [env]. *)

Inductive eval : env -> expr -> val -> Prop :=

  (** Integer literal *)
  | E_Int : forall env n,
      eval env (EInt n) (VInt n)

  (** Boolean literal *)
  | E_Bool : forall env b,
      eval env (EBool b) (VBool b)

  (** Variable lookup *)
  | E_Var : forall env x v,
      env_lookup x env = Some v ->
      eval env (EVar x) v

  (** Arithmetic binary operations: int op int -> int *)
  | E_BinArith : forall env op e1 e2 n1 n2 v,
      eval env e1 (VInt n1) ->
      eval env e2 (VInt n2) ->
      eval_arith_binop op n1 n2 = Some v ->
      eval env (EBinOp op e1 e2) v

  (** Comparison operations: int op int -> bool *)
  | E_BinCmp : forall env op e1 e2 n1 n2 v,
      eval env e1 (VInt n1) ->
      eval env e2 (VInt n2) ->
      eval_cmp_binop op n1 n2 = Some v ->
      eval env (EBinOp op e1 e2) v

  (** Equality on booleans: bool op bool -> bool *)
  | E_BinEqBool : forall env op e1 e2 b1 b2,
      eval env e1 (VBool b1) ->
      eval env e2 (VBool b2) ->
      op = OpEq ->
      eval env (EBinOp op e1 e2) (VBool (Bool.eqb b1 b2))

  (** Inequality on booleans *)
  | E_BinNeBool : forall env op e1 e2 b1 b2,
      eval env e1 (VBool b1) ->
      eval env e2 (VBool b2) ->
      op = OpNe ->
      eval env (EBinOp op e1 e2) (VBool (negb (Bool.eqb b1 b2)))

  (** Logical AND (short-circuit) *)
  | E_And_True : forall env e1 e2 v2,
      eval env e1 (VBool true) ->
      eval env e2 (VBool v2) ->
      eval env (EBinOp OpAnd e1 e2) (VBool v2)

  | E_And_False : forall env e1,
      eval env e1 (VBool false) ->
      eval env (EBinOp OpAnd e1 (EBool false)) (VBool false)

  (** Short-circuit AND: if left is false, result is false regardless of right *)
  | E_And_Short : forall env e1 e2,
      eval env e1 (VBool false) ->
      eval env (EBinOp OpAnd e1 e2) (VBool false)

  (** Logical OR (short-circuit) *)
  | E_Or_False : forall env e1 e2 v2,
      eval env e1 (VBool false) ->
      eval env e2 (VBool v2) ->
      eval env (EBinOp OpOr e1 e2) (VBool v2)

  | E_Or_Short : forall env e1 e2,
      eval env e1 (VBool true) ->
      eval env (EBinOp OpOr e1 e2) (VBool true)

  (** Unary negation: -n *)
  | E_Neg : forall env e n,
      eval env e (VInt n) ->
      eval env (EUnOp OpNeg e) (VInt (- n))

  (** Logical not *)
  | E_Not : forall env e b,
      eval env e (VBool b) ->
      eval env (EUnOp OpNot e) (VBool (negb b))

  (** If-then-else: true branch *)
  | E_IfTrue : forall env e1 e2 e3 v,
      eval env e1 (VBool true) ->
      eval env e2 v ->
      eval env (EIf e1 e2 e3) v

  (** If-then-else: false branch *)
  | E_IfFalse : forall env e1 e2 e3 v,
      eval env e1 (VBool false) ->
      eval env e3 v ->
      eval env (EIf e1 e2 e3) v

  (** Let binding: let x = e1 in e2 *)
  | E_Let : forall env x e1 e2 v1 v2,
      eval env e1 v1 ->
      eval (ECons x v1 env) e2 v2 ->
      eval env (ELet x e1 e2) v2

  (** Lambda abstraction: creates a closure *)
  | E_Lam : forall env x t body,
      eval env (ELam x t body) (VClos x body env)

  (** Function application *)
  | E_App : forall env e1 e2 x body clos_env v2 v,
      eval env e1 (VClos x body clos_env) ->
      eval env e2 v2 ->
      eval (ECons x v2 clos_env) body v ->
      eval env (EApp e1 e2) v.

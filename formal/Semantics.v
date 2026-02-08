(** * NanoCore: Big-Step Operational Semantics

    This file defines the big-step (natural) evaluation semantics for NanoCore.
    The evaluation relation is:
      eval env e env' v  ===  "expression e evaluates to value v,
                               transforming environment env into env'"

    Key properties:
    - Eager evaluation (call-by-value)
    - Deterministic (each expression has at most one value)
    - Lexical scoping via closures
    - Mutable variables via environment update (store-passing style)
    - While loops via inductive unfolding
*)

From Stdlib Require Import ZArith.
From Stdlib Require Import Bool.
From Stdlib Require Import String.
From Stdlib Require Import List.
Import ListNotations.
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

Definition is_string_op (op : binop) : bool :=
  match op with
  | OpStrCat => true
  | _ => false
  end.

(** ** Big-step evaluation relation

    [eval env e env' v] means that expression [e] evaluates to value [v]
    in environment [env], producing updated environment [env'].

    Pure expressions leave the environment unchanged (env' = env).
    Mutation (set) and sequencing thread environment changes. *)

Inductive eval : env -> expr -> env -> val -> Prop :=

  (** Integer literal *)
  | E_Int : forall renv n,
      eval renv (EInt n) renv (VInt n)

  (** Boolean literal *)
  | E_Bool : forall renv b,
      eval renv (EBool b) renv (VBool b)

  (** String literal *)
  | E_String : forall renv s,
      eval renv (EString s) renv (VString s)

  (** Unit literal *)
  | E_Unit : forall renv,
      eval renv EUnit renv VUnit

  (** Variable lookup *)
  | E_Var : forall renv x v,
      env_lookup x renv = Some v ->
      eval renv (EVar x) renv v

  (** Arithmetic binary operations: int op int -> int *)
  | E_BinArith : forall renv renv1 renv2 op e1 e2 n1 n2 v,
      eval renv e1 renv1 (VInt n1) ->
      eval renv1 e2 renv2 (VInt n2) ->
      eval_arith_binop op n1 n2 = Some v ->
      eval renv (EBinOp op e1 e2) renv2 v

  (** Comparison operations: int op int -> bool *)
  | E_BinCmp : forall renv renv1 renv2 op e1 e2 n1 n2 v,
      eval renv e1 renv1 (VInt n1) ->
      eval renv1 e2 renv2 (VInt n2) ->
      eval_cmp_binop op n1 n2 = Some v ->
      eval renv (EBinOp op e1 e2) renv2 v

  (** Equality on booleans: bool op bool -> bool *)
  | E_BinEqBool : forall renv renv1 renv2 op e1 e2 b1 b2,
      eval renv e1 renv1 (VBool b1) ->
      eval renv1 e2 renv2 (VBool b2) ->
      op = OpEq ->
      eval renv (EBinOp op e1 e2) renv2 (VBool (Bool.eqb b1 b2))

  (** Inequality on booleans *)
  | E_BinNeBool : forall renv renv1 renv2 op e1 e2 b1 b2,
      eval renv e1 renv1 (VBool b1) ->
      eval renv1 e2 renv2 (VBool b2) ->
      op = OpNe ->
      eval renv (EBinOp op e1 e2) renv2 (VBool (negb (Bool.eqb b1 b2)))

  (** String concatenation *)
  | E_StrCat : forall renv renv1 renv2 e1 e2 s1 s2,
      eval renv e1 renv1 (VString s1) ->
      eval renv1 e2 renv2 (VString s2) ->
      eval renv (EBinOp OpStrCat e1 e2) renv2 (VString (String.append s1 s2))

  (** String equality *)
  | E_BinEqStr : forall renv renv1 renv2 op e1 e2 s1 s2,
      eval renv e1 renv1 (VString s1) ->
      eval renv1 e2 renv2 (VString s2) ->
      op = OpEq ->
      eval renv (EBinOp op e1 e2) renv2 (VBool (String.eqb s1 s2))

  (** String inequality *)
  | E_BinNeStr : forall renv renv1 renv2 op e1 e2 s1 s2,
      eval renv e1 renv1 (VString s1) ->
      eval renv1 e2 renv2 (VString s2) ->
      op = OpNe ->
      eval renv (EBinOp op e1 e2) renv2 (VBool (negb (String.eqb s1 s2)))

  (** Logical AND (short-circuit) *)
  | E_And_True : forall renv renv1 renv2 e1 e2 v2,
      eval renv e1 renv1 (VBool true) ->
      eval renv1 e2 renv2 (VBool v2) ->
      eval renv (EBinOp OpAnd e1 e2) renv2 (VBool v2)

  | E_And_False : forall renv renv1 e1,
      eval renv e1 renv1 (VBool false) ->
      eval renv (EBinOp OpAnd e1 (EBool false)) renv1 (VBool false)

  (** Short-circuit AND: if left is false, result is false regardless of right *)
  | E_And_Short : forall renv renv1 e1 e2,
      eval renv e1 renv1 (VBool false) ->
      eval renv (EBinOp OpAnd e1 e2) renv1 (VBool false)

  (** Logical OR (short-circuit) *)
  | E_Or_False : forall renv renv1 renv2 e1 e2 v2,
      eval renv e1 renv1 (VBool false) ->
      eval renv1 e2 renv2 (VBool v2) ->
      eval renv (EBinOp OpOr e1 e2) renv2 (VBool v2)

  | E_Or_Short : forall renv renv1 e1 e2,
      eval renv e1 renv1 (VBool true) ->
      eval renv (EBinOp OpOr e1 e2) renv1 (VBool true)

  (** Unary negation: -n *)
  | E_Neg : forall renv renv1 e n,
      eval renv e renv1 (VInt n) ->
      eval renv (EUnOp OpNeg e) renv1 (VInt (- n))

  (** Logical not *)
  | E_Not : forall renv renv1 e b,
      eval renv e renv1 (VBool b) ->
      eval renv (EUnOp OpNot e) renv1 (VBool (negb b))

  (** String length *)
  | E_StrLen : forall renv renv1 e s,
      eval renv e renv1 (VString s) ->
      eval renv (EUnOp OpStrLen e) renv1 (VInt (Z.of_nat (String.length s)))

  (** If-then-else: true branch *)
  | E_IfTrue : forall renv renv1 renv2 e1 e2 e3 v,
      eval renv e1 renv1 (VBool true) ->
      eval renv1 e2 renv2 v ->
      eval renv (EIf e1 e2 e3) renv2 v

  (** If-then-else: false branch *)
  | E_IfFalse : forall renv renv1 renv2 e1 e2 e3 v,
      eval renv e1 renv1 (VBool false) ->
      eval renv1 e3 renv2 v ->
      eval renv (EIf e1 e2 e3) renv2 v

  (** Let binding: let x = e1 in e2.
      The body evaluates in the extended environment.
      The output environment pops the let binding, preserving
      any mutations to variables in the enclosing scope. *)
  | E_Let : forall renv renv1 x e1 e2 v1 v2 vx renv_out,
      eval renv e1 renv1 v1 ->
      eval (ECons x v1 renv1) e2 (ECons x vx renv_out) v2 ->
      eval renv (ELet x e1 e2) renv_out v2

  (** Set: mutable variable update *)
  | E_Set : forall renv renv1 x e v v_old,
      eval renv e renv1 v ->
      env_lookup x renv1 = Some v_old ->
      eval renv (ESet x e) (env_update x v renv1) VUnit

  (** Sequence: e1; e2 *)
  | E_Seq : forall renv renv1 renv2 e1 e2 v1 v2,
      eval renv e1 renv1 v1 ->
      eval renv1 e2 renv2 v2 ->
      eval renv (ESeq e1 e2) renv2 v2

  (** While loop: condition true, execute body and loop *)
  | E_WhileTrue : forall renv renv1 renv2 renv3 cond body v_body v,
      eval renv cond renv1 (VBool true) ->
      eval renv1 body renv2 v_body ->
      eval renv2 (EWhile cond body) renv3 v ->
      eval renv (EWhile cond body) renv3 v

  (** While loop: condition false, stop *)
  | E_WhileFalse : forall renv renv1 cond body,
      eval renv cond renv1 (VBool false) ->
      eval renv (EWhile cond body) renv1 VUnit

  (** Lambda abstraction: creates a closure *)
  | E_Lam : forall renv x t body,
      eval renv (ELam x t body) renv (VClos x body renv)

  (** Function application.
      The body evaluates in the closure's environment (lexical scoping).
      Mutations inside the body don't affect the caller's environment.
      The output environment is the caller's env after evaluating
      the function and argument. *)
  | E_App : forall renv renv1 renv2 renv3 e1 e2 x body clos_env v2 v,
      eval renv e1 renv1 (VClos x body clos_env) ->
      eval renv1 e2 renv2 v2 ->
      eval (ECons x v2 clos_env) body renv3 v ->
      eval renv (EApp e1 e2) renv2 v

  (** Array literal: empty *)
  | E_ArrayNil : forall renv,
      eval renv (EArray []) renv (VArray [])

  (** Array literal: evaluate elements left-to-right *)
  | E_ArrayCons : forall renv renv1 renv2 e es v vs,
      eval renv e renv1 v ->
      eval renv1 (EArray es) renv2 (VArray vs) ->
      eval renv (EArray (e :: es)) renv2 (VArray (v :: vs))

  (** Array indexing: (at arr i) *)
  | E_Index : forall renv renv1 renv2 e1 e2 vs n v,
      eval renv e1 renv1 (VArray vs) ->
      eval renv1 e2 renv2 (VInt n) ->
      nth_error vs (Z.to_nat n) = Some v ->
      eval renv (EIndex e1 e2) renv2 v

  (** Array length *)
  | E_ArrayLen : forall renv renv1 e vs,
      eval renv e renv1 (VArray vs) ->
      eval renv (EUnOp OpArrayLen e) renv1 (VInt (Z.of_nat (length vs))).

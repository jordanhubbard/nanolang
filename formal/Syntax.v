(** * NanoCore: Abstract Syntax

    This file defines the abstract syntax for NanoCore, a minimal subset
    of NanoLang used for formal verification. NanoCore includes:
    - Integer, boolean, and unit literals
    - Binary operators (arithmetic, comparison, logical)
    - Unary operators (negation, logical not)
    - If/then/else expressions
    - Let bindings
    - Mutable variable assignment (set)
    - Sequential composition and while loops
    - Lambda abstractions and function application
*)

From Stdlib Require Import ZArith.
From Stdlib Require Import String.
Open Scope string_scope.

(** ** Types *)

Inductive ty : Type :=
  | TInt   : ty                    (* int *)
  | TBool  : ty                    (* bool *)
  | TUnit  : ty                    (* unit *)
  | TArrow : ty -> ty -> ty.       (* function type: T1 -> T2 *)

(** ** Binary operators *)

Inductive binop : Type :=
  (* Arithmetic *)
  | OpAdd   : binop    (* + *)
  | OpSub   : binop    (* - *)
  | OpMul   : binop    (* * *)
  | OpDiv   : binop    (* / *)
  | OpMod   : binop    (* % *)
  (* Comparison *)
  | OpEq    : binop    (* == *)
  | OpNe    : binop    (* != *)
  | OpLt    : binop    (* < *)
  | OpLe    : binop    (* <= *)
  | OpGt    : binop    (* > *)
  | OpGe    : binop    (* >= *)
  (* Logical *)
  | OpAnd   : binop    (* and *)
  | OpOr    : binop.   (* or *)

(** ** Unary operators *)

Inductive unop : Type :=
  | OpNeg   : unop     (* unary minus *)
  | OpNot   : unop.    (* not *)

(** ** Expressions *)

Inductive expr : Type :=
  | EInt    : Z -> expr                         (* integer literal *)
  | EBool   : bool -> expr                      (* boolean literal *)
  | EUnit   : expr                              (* unit literal *)
  | EVar    : string -> expr                     (* variable reference *)
  | EBinOp  : binop -> expr -> expr -> expr      (* binary operation *)
  | EUnOp   : unop -> expr -> expr               (* unary operation *)
  | EIf     : expr -> expr -> expr -> expr       (* if cond then e1 else e2 *)
  | ELet    : string -> expr -> expr -> expr     (* let x = e1 in e2 *)
  | ESet    : string -> expr -> expr             (* set x = e *)
  | ESeq    : expr -> expr -> expr               (* e1; e2 *)
  | EWhile  : expr -> expr -> expr               (* while cond do body *)
  | ELam    : string -> ty -> expr -> expr       (* fun (x : T) => body *)
  | EApp    : expr -> expr -> expr.              (* function application *)

(** ** Values

    Values are the results of evaluation. They form a subset of expressions
    that cannot be reduced further. *)

Inductive val : Type :=
  | VInt    : Z -> val                           (* integer value *)
  | VBool   : bool -> val                        (* boolean value *)
  | VUnit   : val                                (* unit value *)
  | VClos   : string -> expr -> env -> val       (* closure: param, body, captured env *)

(** ** Environments

    An environment maps variable names to values. We use a simple
    association list representation. *)

with env : Type :=
  | ENil    : env                                (* empty environment *)
  | ECons   : string -> val -> env -> env.       (* binding: x = v, rest *)

(** ** Environment lookup *)

Fixpoint env_lookup (x : string) (e : env) : option val :=
  match e with
  | ENil => None
  | ECons y v rest =>
    if String.eqb x y then Some v
    else env_lookup x rest
  end.

(** ** Environment update (for mutable variables)

    Updates the most recent binding of [x] in the environment.
    If [x] is not found, the environment is unchanged. *)

Fixpoint env_update (x : string) (v : val) (e : env) : env :=
  match e with
  | ENil => ENil
  | ECons y v' rest =>
    if String.eqb x y then ECons y v rest
    else ECons y v' (env_update x v rest)
  end.

(** ** Type equality decidability *)

Lemma ty_eq_dec : forall (t1 t2 : ty), {t1 = t2} + {t1 <> t2}.
Proof.
  decide equality.
Defined.

(** ** Decidable equality for binop *)

Lemma binop_eq_dec : forall (b1 b2 : binop), {b1 = b2} + {b1 <> b2}.
Proof.
  decide equality.
Defined.

(** ** Decidable equality for unop *)

Lemma unop_eq_dec : forall (u1 u2 : unop), {u1 = u2} + {u1 <> u2}.
Proof.
  decide equality.
Defined.

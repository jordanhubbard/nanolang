(** * NanoCore: Abstract Syntax

    This file defines the abstract syntax for NanoCore, a minimal subset
    of NanoLang used for formal verification. NanoCore includes:
    - Integer and boolean literals
    - Binary operators (arithmetic, comparison, logical)
    - Unary operators (negation, logical not)
    - If/then/else expressions
    - Let bindings (immutable)
    - Lambda abstractions and function application
*)

From Stdlib Require Import ZArith.
From Stdlib Require Import String.
Open Scope string_scope.

(** ** Types *)

Inductive ty : Type :=
  | TInt   : ty                    (* int *)
  | TBool  : ty                    (* bool *)
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
  | EVar    : string -> expr                     (* variable reference *)
  | EBinOp  : binop -> expr -> expr -> expr      (* binary operation *)
  | EUnOp   : unop -> expr -> expr               (* unary operation *)
  | EIf     : expr -> expr -> expr -> expr       (* if cond then e1 else e2 *)
  | ELet    : string -> expr -> expr -> expr     (* let x = e1 in e2 *)
  | ELam    : string -> ty -> expr -> expr       (* fun (x : T) => body *)
  | EApp    : expr -> expr -> expr.              (* function application *)

(** ** Values

    Values are the results of evaluation. They form a subset of expressions
    that cannot be reduced further. *)

Inductive val : Type :=
  | VInt    : Z -> val                           (* integer value *)
  | VBool   : bool -> val                        (* boolean value *)
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

(** * NanoCore: Abstract Syntax

    This file defines the abstract syntax for NanoCore, a minimal subset
    of NanoLang used for formal verification. NanoCore includes:
    - Integer, boolean, string, and unit literals
    - Binary operators (arithmetic, comparison, logical, string)
    - Unary operators (negation, logical not, string length)
    - If/then/else expressions
    - Let bindings
    - Mutable variable assignment (set)
    - Sequential composition and while loops
    - Lambda abstractions and function application
    - Array literals, indexing, and length
    - Record (struct) literals and field access
*)

From Stdlib Require Import ZArith.
From Stdlib Require Import String.
From Stdlib Require Import List.
Import ListNotations.
Open Scope string_scope.

(** ** Types *)

Inductive ty : Type :=
  | TInt    : ty                    (* int *)
  | TBool   : ty                    (* bool *)
  | TString : ty                    (* string *)
  | TUnit   : ty                    (* unit *)
  | TArrow  : ty -> ty -> ty        (* function type: T1 -> T2 *)
  | TArray  : ty -> ty              (* array type: array<T> *)
  | TRecord : list (string * ty) -> ty.  (* record type: {f1: T1, ..., fn: Tn} *)

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
  | OpOr    : binop    (* or *)
  (* String *)
  | OpStrCat : binop.  (* string concatenation: + on strings *)

(** ** Unary operators *)

Inductive unop : Type :=
  | OpNeg    : unop     (* unary minus *)
  | OpNot    : unop     (* not *)
  | OpStrLen : unop     (* string length: str_length *)
  | OpArrayLen : unop.  (* array length: array_length *)

(** ** Expressions *)

Inductive expr : Type :=
  | EInt    : Z -> expr                         (* integer literal *)
  | EBool   : bool -> expr                      (* boolean literal *)
  | EString : string -> expr                    (* string literal *)
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
  | EApp    : expr -> expr -> expr               (* function application *)
  | EArray  : list expr -> expr                  (* array literal: [e1, ..., en] *)
  | EIndex  : expr -> expr -> expr               (* array indexing: (at arr i) *)
  | ERecord : list (string * expr) -> expr       (* record literal: {f1: e1, ..., fn: en} *)
  | EField  : expr -> string -> expr.            (* field access: e.f *)

(** ** Values

    Values are the results of evaluation. They form a subset of expressions
    that cannot be reduced further. *)

Inductive val : Type :=
  | VInt    : Z -> val                           (* integer value *)
  | VBool   : bool -> val                        (* boolean value *)
  | VString : string -> val                      (* string value *)
  | VUnit   : val                                (* unit value *)
  | VClos   : string -> expr -> env -> val       (* closure: param, body, captured env *)
  | VArray  : list val -> val                    (* array value *)
  | VRecord : list (string * val) -> val         (* record value *)

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

(** ** Association list lookup (polymorphic) *)

Fixpoint assoc_lookup {A : Type} (x : string) (l : list (string * A)) : option A :=
  match l with
  | [] => None
  | (y, v) :: rest => if String.eqb x y then Some v else assoc_lookup x rest
  end.

(** ** Type equality decidability *)

Lemma ty_eq_dec : forall (t1 t2 : ty), {t1 = t2} + {t1 <> t2}.
Proof.
  fix IH 1.
  intros [| | | | ta tb | ta | fs1] [| | | | ta' tb' | ta' | fs2];
    try (left; reflexivity); try (right; discriminate).
  - (* TArrow *)
    destruct (IH ta ta'); [| right; congruence].
    destruct (IH tb tb'); [| right; congruence].
    left; congruence.
  - (* TArray *)
    destruct (IH ta ta'); [| right; congruence].
    left; congruence.
  - (* TRecord *)
    enough ({fs1 = fs2} + {fs1 <> fs2}) as [->|];
      [left; reflexivity | right; congruence |].
    revert fs2.
    induction fs1 as [|[s1 t1] fs1' IHfs]; intros [|[s2 t2] fs2'].
    + left; reflexivity.
    + right; discriminate.
    + right; discriminate.
    + destruct (string_dec s1 s2); [subst | right; congruence].
      destruct (IH t1 t2); [subst | right; congruence].
      destruct (IHfs fs2'); [subst | right; congruence].
      left; reflexivity.
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

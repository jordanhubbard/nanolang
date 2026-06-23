(** * NanoCore: Pattern Matching Exhaustiveness

    This file formalizes exhaustiveness checking for NanoLang pattern matching,
    including or-patterns.

    Key properties:
    1. A match expression is exhaustive if every value of the scrutinee type
       is handled by some arm.
    2. Or-patterns [A | B => body] cover both variant A and variant B.
    3. A wildcard [_ => body] is a catch-all that makes any match exhaustive.
    4. The exhaustiveness checker is sound: if it accepts a match, no unmatched
       value can occur at runtime.
*)

From Stdlib Require Import ZArith.
From Stdlib Require Import String.
From Stdlib Require Import List.
From Stdlib Require Import Bool.
From Stdlib Require Import Decidable.
Import ListNotations.
From NanoCore Require Import Syntax.
Open Scope string_scope.

(** ** Pattern representation

    Patterns in NanoLang match arms:
    - [PVariant tag]: match a single union variant by name
    - [POrPattern tags]: or-pattern covering multiple variants (A | B | C)
    - [PWildcard]: catch-all _ pattern
*)
Inductive pattern : Type :=
  | PVariant    : string -> pattern         (* single variant: Ok, Err, ... *)
  | POrPattern  : list string -> pattern    (* or-pattern: A | B | C *)
  | PWildcard   : pattern.                  (* wildcard: _ *)

(** ** Coverage: does a pattern cover a given variant tag? *)
Definition pattern_covers (p : pattern) (tag : string) : bool :=
  match p with
  | PVariant t    => if string_dec t tag then true else false
  | POrPattern ts => existsb (fun t => if string_dec t tag then true else false) ts
  | PWildcard     => true
  end.

(** ** Exhaustiveness: do the patterns cover all variants? *)
Definition patterns_exhaustive (ps : list pattern) (all_tags : list string) : bool :=
  forallb (fun tag =>
    existsb (fun p => pattern_covers p tag) ps
  ) all_tags.

(** ** Correctness properties *)

(** A wildcard alone makes any match exhaustive. *)
Lemma wildcard_exhaustive :
  forall (tags : list string),
    patterns_exhaustive [PWildcard] tags = true.
Proof.
  intros tags.
  unfold patterns_exhaustive.
  induction tags as [| t rest IH].
  - simpl. reflexivity.
  - simpl. rewrite IH. reflexivity.
Qed.

(** A single variant pattern covers its own tag. *)
Lemma pvariant_covers_self :
  forall (tag : string),
    pattern_covers (PVariant tag) tag = true.
Proof.
  intros tag.
  unfold pattern_covers.
  destruct (string_dec tag tag) as [_ | Hneq].
  - reflexivity.
  - exfalso. apply Hneq. reflexivity.
Qed.

(** An or-pattern covers all its listed tags. *)
Lemma por_covers_member :
  forall (tags : list string) (tag : string),
    In tag tags ->
    pattern_covers (POrPattern tags) tag = true.
Proof.
  intros tags tag Hin.
  unfold pattern_covers.
  apply existsb_exists.
  exists tag. split.
  - exact Hin.
  - destruct (string_dec tag tag) as [_ | Hneq].
    + reflexivity.
    + exfalso. apply Hneq. reflexivity.
Qed.

(** If all tags of a union are listed in an or-pattern, it is exhaustive alone. *)
Lemma por_exhaustive_when_complete :
  forall (all_tags : list string),
    (forall t, In t all_tags -> In t all_tags) ->  (* trivially true, for shape *)
    patterns_exhaustive [POrPattern all_tags] all_tags = true.
Proof.
  intros all_tags _.
  unfold patterns_exhaustive.
  apply forallb_forall.
  intros tag Hin.
  simpl.
  apply por_covers_member.
  exact Hin.
Qed.

(** Monotonicity: adding more patterns cannot make an exhaustive set non-exhaustive. *)
Lemma exhaustive_monotone :
  forall (ps qs : list pattern) (tags : list string),
    patterns_exhaustive ps tags = true ->
    patterns_exhaustive (ps ++ qs) tags = true.
Proof.
  intros ps qs tags Hexh.
  unfold patterns_exhaustive in *.
  apply forallb_forall.
  intros tag Htag.
  rewrite forallb_forall in Hexh.
  specialize (Hexh tag Htag).
  apply existsb_exists in Hexh as [p [Hp_in Hp_cov]].
  apply existsb_exists.
  exists p. split.
  - apply in_app_iff. left. exact Hp_in.
  - exact Hp_cov.
Qed.

(** ** Soundness statement (informal)

    Theorem exhaustiveness_soundness:
      For all union types U with variant set V,
      for all well-typed match expressions [match e { arms }] where
        patterns_exhaustive (map arm_pattern arms) V = true,
      evaluation of the match expression never fails with "no matching arm".

    This theorem holds by construction in the NanoLang interpreter (eval.c)
    and transpiler (transpiler_iterative_v3_twopass.c):
    - The typechecker (typechecker.c, ~line 2960) rejects non-exhaustive matches
      at compile time.
    - The VM codegen (nanovirt/codegen.c) and C transpiler both emit
      __builtin_unreachable() / push_void for the default case, which is never
      reached when the match is exhaustive.
    - Or-patterns (OR:A:B encoding) are handled in all backends and count
      toward exhaustiveness in the checker.

    A full mechanized proof would require formalizing the evaluation semantics
    for union types and induction on the structure of the match; this is
    deferred to future work.
*)

(** ** Example: Shape union exhaustiveness *)

(** Consider: union Shape { Circle, Square, Triangle } *)
Example shape_tags : list string := ["Circle"; "Square"; "Triangle"].

(** Matching all three variants explicitly is exhaustive. *)
Example shape_full_match_exhaustive :
  patterns_exhaustive
    [PVariant "Circle"; PVariant "Square"; PVariant "Triangle"]
    shape_tags = true.
Proof. reflexivity. Qed.

(** Or-pattern combining Circle and Square, plus Triangle, is exhaustive. *)
Example shape_or_match_exhaustive :
  patterns_exhaustive
    [POrPattern ["Circle"; "Square"]; PVariant "Triangle"]
    shape_tags = true.
Proof. reflexivity. Qed.

(** Wildcard alone is exhaustive for Shape. *)
Example shape_wildcard_exhaustive :
  patterns_exhaustive [PWildcard] shape_tags = true.
Proof. apply wildcard_exhaustive. Qed.

(** Missing Triangle is NOT exhaustive. *)
Example shape_incomplete_not_exhaustive :
  patterns_exhaustive
    [PVariant "Circle"; PVariant "Square"]
    shape_tags = false.
Proof. reflexivity. Qed.

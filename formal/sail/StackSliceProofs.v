(* I check reusable laws of the generated stack model, not the production VM. *)
From SailStdpp Require Import Base Real.
Require Import stack_slice_types stack_slice.
Import ListNotations.

Theorem nop_identity (stack : list (mword 64)) :
  execute (Nop tt) stack = Some stack.
Proof. reflexivity. Qed.

Theorem push_then_pop (value : mword 64) (stack : list (mword 64)) :
  match execute (Push value) stack with
  | Some next => execute (Pop tt) next
  | None => None
  end = Some stack.
Proof. reflexivity. Qed.

Theorem dup_then_pop (value : mword 64) (rest : list (mword 64)) :
  match execute (Dup tt) (value :: rest) with
  | Some next => execute (Pop tt) next
  | None => None
  end = Some (value :: rest).
Proof. reflexivity. Qed.

Theorem swap_involution (first second : mword 64) (rest : list (mword 64)) :
  match execute (Swap tt) (first :: second :: rest) with
  | Some next => execute (Swap tt) next
  | None => None
  end = Some (first :: second :: rest).
Proof. reflexivity. Qed.

Theorem dup_underflow : execute (Dup tt) [] = None.
Proof. reflexivity. Qed.

Theorem pop_underflow : execute (Pop tt) [] = None.
Proof. reflexivity. Qed.

Theorem swap_empty_underflow : execute (Swap tt) [] = None.
Proof. reflexivity. Qed.

Theorem swap_singleton_underflow (value : mword 64) :
  execute (Swap tt) [value] = None.
Proof. reflexivity. Qed.

Theorem execute_frame_extension instruction stack next suffix :
  execute instruction stack = Some next ->
  execute instruction (stack ++ suffix) = Some (next ++ suffix).
Proof.
  intro H.
  destruct instruction; simpl in *;
    try (inversion H; subst; reflexivity).
  all: destruct stack as [|first rest]; simpl in *;
    try discriminate; try (inversion H; subst; reflexivity).
  destruct rest as [|second rest]; simpl in *; try discriminate.
  inversion H; subst; reflexivity.
Qed.

Print Assumptions execute_frame_extension.
Print Assumptions nop_identity.
Print Assumptions push_then_pop.
Print Assumptions dup_then_pop.
Print Assumptions swap_involution.
Print Assumptions dup_underflow.
Print Assumptions pop_underflow.
Print Assumptions swap_empty_underflow.
Print Assumptions swap_singleton_underflow.

(** I pin the advertised theorem types independently of their implementations.
    Removing a theorem or weakening its conclusion must break this build. *)
From NanoCore Require Import Syntax Semantics Typing Soundness Progress
  Determinism Equivalence EvalFn.

Definition preservation_contract : forall renv e renv' result,
  eval renv e renv' result -> forall gamma t,
  has_type gamma e t -> env_ctx_agree renv gamma ->
  val_has_type result t /\ env_ctx_agree renv' gamma := preservation.

Definition progress_contract : forall e t,
  has_type CtxNil e t -> is_value e \/ (exists e', step e e') := progress.

Definition determinism_contract : forall renv e renv' v,
  eval renv e renv' v -> forall renv'' v',
  eval renv e renv'' v' -> v = v' /\ renv' = renv'' := eval_deterministic.

Definition equivalence_contract : forall e v,
  pure e -> eclosed e -> eval ENil e ENil v ->
  multi_step_equiv e (val_to_expr v) := eval_to_multistep.

Definition evaluator_contract : forall fuel renv e renv' v,
  eval_fn fuel renv e = Some (renv', v) -> eval renv e renv' v := eval_fn_sound.

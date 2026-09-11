(** I print the dependencies of named theorem terms, not source-token counts.
    These results concern NanoCore. Production compiler refinement and the
    general eval_fn_sound theorem remain separate, unfinished obligations. *)
From NanoCore Require Import Soundness Progress Determinism Equivalence
  EvalFn Exhaustiveness.

Print Assumptions preservation.
Print Assumptions progress.
Print Assumptions eval_deterministic.
Print Assumptions eval_to_multistep.
Print Assumptions eval_to_multistep_gen.
Print Assumptions eval_fn_sound_simple.
Print Assumptions eval_fn_sound_unop.
Print Assumptions eval_fn_sound_if.
Print Assumptions eval_fn_sound_seq.
Print Assumptions eval_fn_sound_lam.
Print Assumptions eval_fn_sound_fix.
Print Assumptions eval_fn_sound_construct.
Print Assumptions eval_fn_sound_strindex.
Print Assumptions eval_fn_sound_set.
Print Assumptions wildcard_exhaustive.
Print Assumptions pvariant_covers_self.
Print Assumptions por_covers_member.
Print Assumptions por_exhaustive_when_complete.
Print Assumptions exhaustive_monotone.

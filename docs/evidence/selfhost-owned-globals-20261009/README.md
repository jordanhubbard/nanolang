# Self-hosted declared global source checkpoint

I lower declared int/bool/float/string and copyable scalar-payload union globals
through my self-hosted ownership producer. I retain exact slots and union
identities, source-ordered once-only entry initialization, shared helper state,
local/parameter precedence and callable shadowing. My resource-bearing union
producer and broader global storage remain open under #981 and the full
[5.1 contract](../../RELEASE_5.1_SCOPE.md).

My final gate passes bootstrap, raw production, and all 38 global methods in
864.245 seconds. `final.json` records commands, exits and matching
before/after source inventories. My final Stage1/Stage2 modules are byte-identical;
`final-bootstrap/manifest.json` records every stage. I record compiler input hashes,
including the direct `nb_value_call` shadow added during final review. This is a
Darwin checkpoint, not final Linux/Darwin release qualification.

- The raw producer suite inherits all twelve original global source methods.
  Each runs normal entry and every selected source shadow through verified VM
  and strict ASan/UBSan/LeakSanitizer native C, or requires the exact refusal and
  preservation of prior module bytes. The raw path alone is not source checking.
- Both fresh installed stages run those same twelve unchanged sources and one
  additional failing-global-shadow publication control: 26 installed methods.
  The latter requires the actual assertion-execution error, not any shadow error.
- The shared harness adapter retains all 48 C selected-owner/global methods
  (7.681 seconds). All four adjacent affine scalar-union methods pass, including
  imports, exact instance metadata and all-frontends refusals (325.086 seconds).
  These adjacency runs precede only the final direct callable shadow addition;
  the final full bootstrap and global suites include it.

I preserve the failed diagnostic probes. `initial.log` and `current-assembler.log`
show the missing four-byte empty path-count subpayload; rebuilding the assembler
retained that failure. The probe logs per-command outcomes rather than failing its
own process. The corrected envelope retains that count even with no paths.
`installed-first.log` retains six diagnostic-wording assertion failures: all
positives ran and negatives were refused, but C-only wording did not match my
independent Nano checker/lowerer. Explicit per-frontend diagnostic expectations
fix the harness without changing sources or acceptance decisions.

`final-raw/` and `final-installed/` retain commands, source inputs, generated C,
assembly and logs. The initial and earlier successful checkpoints remain separate.
Inventory files retain hashes and sizes for modules and native executables;
I archive only text here. Live binary artifacts remain under `/private/tmp`.
My compiler implementation is based on bfa04d81a plus the retained source patch;
final source hashes identify the exact tested implementation.

# My native-product coverage linker checkpoint

Coverage job113797705210 at 51c8cfb48 reaches Stage1 hello compilation, then fails to resolve __gcov_* from the covered runtime object. My artifact SHA-256 matches the job upload receipt; provenance.json pins the job, revision and artifact. I retain the complete job and individual bootstrap logs.

My local probe uses the existing self-hosted compiler component, real native translation and a coverage-instrumented object in an isolated tool root. LDFLAGS-only coverage fails while explicit NANO_LDFLAGS succeeds; no flags fail without replacing prior output. I prepare a driver patch that selects nonempty NANO_LDFLAGS, otherwise LDFLAGS. Its shadow checks fallback, override and empty settings.

I compile a temporary driver copy with all shadows and repeat the real linker probe. Both fallback and override produce executables that exit zero; missing flags still fail and preserve prior output. The initial temporary-copy edit accidentally inserted the helper into a source-string fixture; I retain that failed build and the corrected insertion at the actual main declaration. The tested patch is prepared-driver.patch. It is not integrated into compiler source yet: bootstrap at 915b85284 remains live with unchanged inputs. Permanent regression tests, integrated qualification and fresh hosted coverage remain required under #982.

I subsequently integrate the prepared fixes and permanent tests after the baseline bootstrap passes. Integrated results and source hashes are recorded in docs/evidence/import-global-managed-20261009/integrated. A fresh bootstrap for the changed source remains required.

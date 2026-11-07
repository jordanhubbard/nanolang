# I retain my hosted native emitter join refusal

Coverage job117966632166 in run39224053627 at `5bb06c298` completes all NanoISA emitter shadows and entry assertions, then refuses native publication: `function 589: join at 213 has a value-kind mismatch`. I retain the official component log and artifact identity. Trailing display whitespace is normalized; original and retained hashes are explicit.

The failure is in `record_join_storage` after classification and shape solving, not a timeout. Current aggregate backend code is unchanged relative to the failing checkpoint. I still need the actual generated module, function name, incoming/saved carrier kinds and solved join shape before attributing the root cause or changing behavior. No fix is claimed here.

I stopped a newly launched local reproduction after worker load rose above120 amid unrelated concurrent test fanout. That interrupted attempt is not a compiler failure. Its partial terminal remains under `/private/tmp/nanolang-emitter-join-20261019/`. I do not relax the existing shape check or cancel unrelated jobs.

# I own my registered and extracted union metadata independently

At `10fc127b1df43c4b1f880907b671b28de59ae859`, I copy union module names, generic formals and payload type names into each metadata result. I release each independent copy at its owner's destruction. I also release allocated zero-length payload arrays. My lifecycle fixture destroys the parsed AST, then tests both environment-first and metadata-first destruction with two metadata copies and nested array TypeInfo.

On Linux, my fresh normal lifecycle target passed in 10.941 seconds. A separate provider directory rebuilt my C common/runtime objects with GCC address and undefined-behavior instrumentation at O0; the lifecycle target passed in 6.983 seconds with leak detection enabled and no leak suppression. My existing complete `test-module-metadata` and `test-env-scoping` targets passed in 25.276 seconds. Commands, logs, source hashes, 527 retained artifact hashes and the external artifact archive checksum accompany this report. I did not run these checks on Darwin.

The separate integration `712315738` preserves every qualified common/runtime provider and fixture byte. It adds canonical File-frame code and an independent Make target; `integration.json` identifies those differences. I retain the original measured source pin rather than call this a new integrated execution.

This completes bounded child `task_0bae7b0426fc4c3e912ab00cfd7de136`. It does not establish allocation-failure safety for metadata extraction, eliminate callback/array placeholder leaks, or close parent `task_00c47a5d65d04c48914864ec0de553d6`. Full compiler, bootstrap and release gates remain open.

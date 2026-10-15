# Prepared shared callable regression

I extend my existing introspection suite to seven methods and reuse its verified VM, C11 and sanitizer execution path. The new fixture covers all eight operations through function values, returned/higher-order calls, exact physical paths when the declared name differs from the filename, boundary indices and single index evaluation. Its index, returned-function and higher-order helper shadows assert observable behavior.

All seven methods pass against the isolated C candidate (1.651 seconds) and self-hosted candidate (4.750 seconds). My temporary test runner selects the original repository tools and the prepared fixture explicitly. I retain the test patch and fixture for integration after the broad gate finishes; no production/test sources in that gate have changed. #982/#976 remain open.

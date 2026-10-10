probe-leaf:
	NANO_BUILD_CACHE=/work/coverage-env-probe/cache python3 coverage_env_probe.py
probe-nested:
	$(MAKE) -f $(PROBE_MAKEFILE) -f coverage_env_probe.mk probe-leaf

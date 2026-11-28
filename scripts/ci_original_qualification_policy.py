"""I preserve original bootstrap and ordinary shadow defaults."""

TIMEOUT_OVERRIDES = (
    'NANO_SHADOW_TIMEOUT_SECONDS',
    'BOOTSTRAP_NANOISA_TIMEOUT',
    'BOOTSTRAP_NANOISA_SHADOW_TIMEOUT',
    'CMD_TIMEOUT',
)


def require_original_timeout_defaults(environment):
    """I refuse inherited policy before creating an execution namespace."""
    for name in TIMEOUT_OVERRIDES:
        if name in environment:
            raise RuntimeError('I require original timeout defaults; inherited override: ' + name)

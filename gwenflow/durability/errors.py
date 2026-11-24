class GwenflowDurabilityError(RuntimeError):
    """Misuse of durable execution (missing name, unsupported feature, sync API in a workflow...)."""

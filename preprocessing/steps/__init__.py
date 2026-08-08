"""Importing this package triggers self-registration of every preprocessing
step (geometric + photometric) into `preprocessing.steps.registry.step_registry`.
"""

from preprocessing.steps import geometric, photometric  # noqa: F401

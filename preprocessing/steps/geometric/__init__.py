"""Importing this package registers every geometric step. Order of these
imports doesn't matter -- pipeline order comes from config, not import
order.
"""

from preprocessing.steps.geometric import perspective_correction, deskew  # noqa: F401

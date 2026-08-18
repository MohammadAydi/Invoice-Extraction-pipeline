"""Importing this package self-registers every extraction flow.

Flows need no optional dependencies of their own -- the heavy pieces are the
detector and recognizer they are *given* -- so unlike those, all three import
unconditionally.
"""

from orchestration.flows import (  # noqa: F401
    detector_driven,
    layout_driven,
    single_engine,
)
from orchestration.flows.base import ExtractionFlow, FlowOutcome  # noqa: F401

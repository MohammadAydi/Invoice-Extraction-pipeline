"""Importing this package self-registers every layout classifier."""

from table_extraction.classifiers import (  # noqa: F401
    bill_layout_classifier,
    passthrough_classifier,
)

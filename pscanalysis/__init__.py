"""Python library to infer copy number variation (CNV) from single-cell RNA-seq data."""

from importlib.metadata import version

from . import tl

__all__ = ["tl", "infercnv"]
__version__ = version("pscanalysis")

"""Factor modules for IdeaRank scoring.

Each factor (U, C, L, Q, T) is implemented as an independent module
that can be configured, tested, and swapped during experimentation.
"""

from idearank.factors.base import BaseFactor, FactorResult
from idearank.factors.uniqueness import UniquenessFactor
from idearank.factors.cohesion import CohesionFactor
from idearank.factors.learning import LearningFactor
from idearank.factors.quality import QualityFactor
from idearank.factors.trust import TrustFactor

__all__ = [
    "BaseFactor",
    "FactorResult",
    "UniquenessFactor",
    "CohesionFactor",
    "LearningFactor",
    "QualityFactor",
    "TrustFactor",
]


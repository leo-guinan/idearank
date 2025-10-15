"""IdeaRank: A PageRank replacement for ideas, not links.

This package implements a multi-factor ranking algorithm for content
that rewards uniqueness, cohesion, learning progression, quality, and trust.

Now includes IdeaRank-Thought competition system for real-time reasoning
evaluation and coaching interventions.
"""

from idearank.models import ContentItem, ContentSource, Embedding, TopicMixture
from idearank.config import IdeaRankConfig
from idearank.scorer import IdeaRankScorer, ContentSourceScorer

# Competition system imports
try:
    from idearank.competition_models import (
        Player, Coach, Challenge, Match, ReasoningTrace, ReasoningNode,
        CoachingEvent, MatchStatus, CoachingType, FactorType,
        OutcomeValidity, ConstraintCompliance, IdeaRankThoughtScore
    )
    from idearank.competition_pipeline import CompetitionPipeline
    from idearank.competition_scorer import IdeaRankThoughtScorer
    from idearank.competition_visualizer import CompetitionVisualizer
    from idearank.interactive_game import InteractiveGame
    COMPETITION_AVAILABLE = True
except ImportError:
    COMPETITION_AVAILABLE = False
    # Define placeholders for when competition modules aren't available
    Player = None
    Coach = None
    Challenge = None
    Match = None
    ReasoningTrace = None
    ReasoningNode = None
    CoachingEvent = None
    MatchStatus = None
    CoachingType = None
    FactorType = None
    OutcomeValidity = None
    ConstraintCompliance = None
    IdeaRankThoughtScore = None
    CompetitionPipeline = None
    IdeaRankThoughtScorer = None
    CompetitionVisualizer = None
    InteractiveGame = None

__version__ = "2.1.0"

__all__ = [
    "ContentItem",
    "ContentSource",
    "Embedding",
    "TopicMixture",
    "IdeaRankConfig",
    "IdeaRankScorer",
    "ContentSourceScorer",
    "COMPETITION_AVAILABLE",
]

# Add competition exports if available
if COMPETITION_AVAILABLE:
    __all__.extend([
        "Player",
        "Coach", 
        "Challenge",
        "Match",
        "ReasoningTrace",
        "ReasoningNode",
        "CoachingEvent",
        "MatchStatus",
        "CoachingType",
        "FactorType",
        "OutcomeValidity",
        "ConstraintCompliance",
        "IdeaRankThoughtScore",
        "CompetitionPipeline",
        "IdeaRankThoughtScorer",
        "CompetitionVisualizer",
        "InteractiveGame",
    ])


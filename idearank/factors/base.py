"""Base classes for IdeaRank factors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from idearank.models import ContentItem, ContentSource


@dataclass
class FactorResult:
    """Result of a factor computation.
    
    Contains the score and optional debugging information.
    """
    
    score: float  # The final factor value
    components: dict[str, float]  # Intermediate values for debugging
    metadata: dict[str, Any]  # Additional context
    
    def __post_init__(self):
        """Validate the score is in a reasonable range."""
        if not 0.0 <= self.score <= 1.0:
            # Allow scores slightly outside [0,1] but warn
            if not -0.01 <= self.score <= 1.01:
                raise ValueError(
                    f"Score {self.score} is outside expected range [0, 1]. "
                    "Check your factor implementation."
                )


class BaseFactor(ABC):
    """Abstract base class for all IdeaRank factors.
    
    Each factor implements a compute() method that takes a content item and context,
    returning a FactorResult with score and debugging info.
    """
    
    def __init__(self, config: Any):
        """Initialize with factor-specific configuration."""
        self.config = config
    
    @abstractmethod
    def compute(
        self, 
        content_item: ContentItem, 
        content_source: ContentSource,
        context: Optional[dict[str, Any]] = None
    ) -> FactorResult:
        """Compute the factor score for a content item.
        
        Args:
            content_item: The content item to score
            content_source: The source containing the content item
            context: Optional context (neighborhoods, analytics, etc.)
            
        Returns:
            FactorResult with score and debugging info
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this factor."""
        pass


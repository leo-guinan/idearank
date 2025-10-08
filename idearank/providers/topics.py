"""Topic model provider interface and implementations."""

from abc import ABC, abstractmethod
from typing import List
import numpy as np

from idearank.models import TopicMixture


class TopicModelProvider(ABC):
    """Abstract interface for topic modeling."""
    
    @abstractmethod
    def get_topics(self, text: str) -> TopicMixture:
        """Get topic distribution for a single text."""
        pass
    
    @abstractmethod
    def get_topics_batch(self, texts: List[str]) -> List[TopicMixture]:
        """Get topic distributions for a batch of texts."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name/identifier of the topic model."""
        pass
    
    @property
    @abstractmethod
    def num_topics(self) -> int:
        """Number of topics in the model."""
        pass


class DummyTopicModelProvider(TopicModelProvider):
    """Dummy provider for testing - generates random topic distributions."""
    
    def __init__(self, num_topics: int = 50, seed: int = 42):
        """Initialize with fixed number of topics and random seed."""
        self._num_topics = num_topics
        self._seed = seed
        self.rng = np.random.default_rng(seed)
    
    @property
    def model_name(self) -> str:
        return f"dummy-lda-{self._num_topics}"
    
    @property
    def num_topics(self) -> int:
        return self._num_topics
    
    def get_topics(self, text: str) -> TopicMixture:
        """Generate random topic distribution based on text hash."""
        # Use text hash for reproducibility
        text_seed = hash(text) % (2**32)
        local_rng = np.random.default_rng(text_seed)
        
        # Generate random distribution
        raw = local_rng.exponential(1.0, self._num_topics)
        distribution = (raw / raw.sum()).astype(np.float32)
        
        return TopicMixture(distribution=distribution, topic_model=self.model_name)
    
    def get_topics_batch(self, texts: List[str]) -> List[TopicMixture]:
        """Generate topic distributions for batch."""
        return [self.get_topics(text) for text in texts]


class LDATopicModelProvider(TopicModelProvider):
    """Provider using scikit-learn's LDA.
    
    NOTE: Requires pre-trained model.
    This is a stub - implement when needed.
    """
    
    def __init__(self, model_path: str, num_topics: int = 50):
        """Initialize with path to trained model."""
        self.model_path = model_path
        self._num_topics = num_topics
        # TODO: Load model from disk
    
    @property
    def model_name(self) -> str:
        return f"lda-{self._num_topics}"
    
    @property
    def num_topics(self) -> int:
        return self._num_topics
    
    def get_topics(self, text: str) -> TopicMixture:
        """Get topic distribution using trained LDA."""
        # TODO: Implement vectorization and transformation
        raise NotImplementedError("LDA provider not yet implemented")
    
    def get_topics_batch(self, texts: List[str]) -> List[TopicMixture]:
        """Get topic distributions for batch."""
        # TODO: Implement batch transformation
        raise NotImplementedError("LDA provider not yet implemented")


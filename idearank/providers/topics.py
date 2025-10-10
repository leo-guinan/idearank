"""Topic model provider interface and implementations."""

from abc import ABC, abstractmethod
from typing import List
import numpy as np
import logging

from idearank.models import TopicMixture

logger = logging.getLogger(__name__)


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
    """Topic model using Latent Dirichlet Allocation.
    
    Uses scikit-learn for LDA implementation.
    """
    
    def __init__(
        self,
        num_topics: int = 50,
        random_state: int = 42,
        max_features: int = 1000,
        min_df: int = 2,
        max_df: float = 0.8,
    ):
        """Initialize LDA model.
        
        Args:
            num_topics: Number of topics to extract
            random_state: Random seed for reproducibility
            max_features: Maximum vocabulary size
            min_df: Minimum document frequency
            max_df: Maximum document frequency (as fraction)
        """
        try:
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.decomposition import LatentDirichletAllocation
        except ImportError:
            raise ImportError(
                "scikit-learn required for LDA. "
                "Install with: pip install scikit-learn"
            )
        
        self._num_topics = num_topics
        self.random_state = random_state
        
        # Initialize vectorizer
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words='english',
            lowercase=True,
        )
        
        # Initialize LDA model
        self.lda = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=random_state,
            learning_method='online',  # More efficient for large corpora
            max_iter=10,
            batch_size=128,
        )
        
        self.is_fitted = False
        
        logger.info(f"Initialized LDA topic model: {num_topics} topics")
    
    def fit(self, texts: List[str]) -> None:
        """Fit the LDA model on a corpus of texts.
        
        Args:
            texts: List of text documents to fit on
        """
        if self.is_fitted:
            logger.info("LDA model already fitted, skipping fit step")
            return
        
        logger.info(f"Fitting LDA topic model on {len(texts)} documents...")
        
        # Vectorize the texts
        doc_term_matrix = self.vectorizer.fit_transform(texts)
        
        # Fit LDA model
        self.lda.fit(doc_term_matrix)
        
        self.is_fitted = True
        logger.info(f"✓ LDA model fitted with {self._num_topics} topics")
    
    @property
    def model_name(self) -> str:
        return f"lda-{self._num_topics}"
    
    @property
    def num_topics(self) -> int:
        return self._num_topics
    
    def get_topics(self, text: str) -> TopicMixture:
        """Get topic distribution for a single text.
        
        Args:
            text: Text document
            
        Returns:
            TopicMixture with probability distribution
        """
        if not self.is_fitted:
            # Return uniform distribution if not fitted
            logger.warning("LDA not fitted yet, returning uniform distribution")
            probs = [1.0 / self._num_topics] * self._num_topics
            return TopicMixture(
                distribution=np.array(probs, dtype=np.float32),
                topic_model=self.model_name
            )
        
        # Vectorize text
        doc_vector = self.vectorizer.transform([text])
        
        # Infer topics
        topic_dist = self.lda.transform(doc_vector)[0]
        
        # Normalize to ensure sum = 1.0 (sometimes LDA has numerical errors)
        topic_dist = topic_dist / topic_dist.sum()
        
        # Add smoothing to avoid exact zeros
        epsilon = 1e-10
        topic_dist = topic_dist + epsilon
        topic_dist = topic_dist / topic_dist.sum()
        
        return TopicMixture(
            distribution=topic_dist.astype(np.float32),
            topic_model=self.model_name
        )
    
    def get_topics_batch(self, texts: List[str]) -> List[TopicMixture]:
        """Get topic distributions for a batch of texts.
        
        Args:
            texts: List of text documents
            
        Returns:
            List of TopicMixture objects
        """
        if not self.is_fitted:
            # Return uniform distributions if not fitted
            logger.warning("LDA not fitted yet, returning uniform distributions")
            return [self.get_topics(text) for text in texts]
        
        # Vectorize all texts
        doc_vectors = self.vectorizer.transform(texts)
        
        # Infer topics for all
        topic_dists = self.lda.transform(doc_vectors)
        
        # Normalize each distribution
        topic_dists = topic_dists / topic_dists.sum(axis=1, keepdims=True)
        
        # Add smoothing
        epsilon = 1e-10
        topic_dists = topic_dists + epsilon
        topic_dists = topic_dists / topic_dists.sum(axis=1, keepdims=True)
        
        return [
            TopicMixture(
                distribution=dist.astype(np.float32),
                topic_model=self.model_name
            )
            for dist in topic_dists
        ]
    
    def get_top_words_per_topic(self, n_words: int = 10) -> List[List[str]]:
        """Get top words for each topic.
        
        Args:
            n_words: Number of top words per topic
            
        Returns:
            List of lists of top words
        """
        if not self.is_fitted:
            return []
        
        feature_names = self.vectorizer.get_feature_names_out()
        top_words = []
        
        for topic_idx, topic in enumerate(self.lda.components_):
            # Get top word indices for this topic
            top_indices = topic.argsort()[-n_words:][::-1]
            words = [feature_names[i] for i in top_indices]
            top_words.append(words)
        
        return top_words


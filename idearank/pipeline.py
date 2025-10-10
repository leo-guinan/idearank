"""End-to-end pipeline for IdeaRank computation.

Orchestrates:
1. Embedding generation
2. Topic modeling
3. Neighborhood search
4. Factor computation
5. Content item & source scoring
6. Optional network layer
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from idearank.models import ContentItem, ContentSource
from idearank.config import IdeaRankConfig
from idearank.scorer import IdeaRankScorer, IdeaRankScore, ContentSourceScorer, ContentSourceRankScore
from idearank.network import KnowledgeRankComputer, KnowledgeRankScore
from idearank.providers import (
    EmbeddingProvider,
    TopicModelProvider,
    NeighborhoodProvider,
)

logger = logging.getLogger(__name__)


class IdeaRankPipeline:
    """End-to-end pipeline for computing IdeaRank scores."""
    
    def __init__(
        self,
        config: IdeaRankConfig,
        embedding_provider: EmbeddingProvider,
        topic_provider: TopicModelProvider,
        neighborhood_provider: NeighborhoodProvider,
    ):
        """Initialize pipeline with config and providers."""
        self.config = config
        self.embedding_provider = embedding_provider
        self.topic_provider = topic_provider
        self.neighborhood_provider = neighborhood_provider
        
        # Initialize scorers
        self.content_scorer = IdeaRankScorer(config)
        self.source_scorer = ContentSourceScorer(config)
        self.network_computer = KnowledgeRankComputer(config.network)
    
    def process_content_item(self, content_item: ContentItem) -> ContentItem:
        """Add embeddings and topics to a content item.
        
        Modifies content item in-place and returns it.
        """
        if content_item.embedding is None:
            logger.debug(f"Generating embedding for content item {content_item.id}")
            content_item.embedding = self.embedding_provider.embed(content_item.full_text)
        
        if content_item.topic_mixture is None:
            logger.debug(f"Generating topics for content item {content_item.id}")
            content_item.topic_mixture = self.topic_provider.get_topics(content_item.full_text)
        
        return content_item
    
    def process_content_batch(self, content_items: List[ContentItem]) -> List[ContentItem]:
        """Process multiple content items efficiently."""
        # Generate embeddings for items that don't have them
        items_needing_embeddings = [item for item in content_items if item.embedding is None]
        if items_needing_embeddings:
            logger.info(f"Generating embeddings for {len(items_needing_embeddings)} content items")
            texts = [item.full_text for item in items_needing_embeddings]
            embeddings = self.embedding_provider.embed_batch(texts)
            for item, embedding in zip(items_needing_embeddings, embeddings):
                item.embedding = embedding
        
        # Fit topic model if using LDA (needs corpus)
        from idearank.providers.topics import LDATopicModelProvider
        if isinstance(self.topic_provider, LDATopicModelProvider):
            if not self.topic_provider.is_fitted:
                # Fit on all available texts
                all_texts = [item.full_text for item in content_items]
                logger.info(f"Fitting LDA topic model on {len(all_texts)} documents...")
                self.topic_provider.fit(all_texts)
        
        # Generate topics
        items_needing_topics = [item for item in content_items if item.topic_mixture is None]
        if items_needing_topics:
            logger.info(f"Generating topics for {len(items_needing_topics)} content items")
            texts = [item.full_text for item in items_needing_topics]
            topics = self.topic_provider.get_topics_batch(texts)
            for item, topic in zip(items_needing_topics, topics):
                item.topic_mixture = topic
        
        return content_items
    
    def index_content(self, content_items: List[ContentItem]) -> None:
        """Add content items to the neighborhood index."""
        logger.info(f"Indexing {len(content_items)} content items for ANN search")
        self.neighborhood_provider.index_content_batch(content_items)
    
    def score_content_item(
        self,
        content_item: ContentItem,
        content_source: ContentSource,
        compute_analytics_context: bool = True,
    ) -> IdeaRankScore:
        """Score a single content item.
        
        Args:
            content_item: Content item to score
            content_source: Source containing the content item
            compute_analytics_context: Whether to compute analytics normalization
            
        Returns:
            IdeaRankScore
        """
        # Ensure content item has embeddings and topics
        self.process_content_item(content_item)
        
        # Build context
        context = self._build_content_context(content_item, content_source, compute_analytics_context)
        
        # Score
        return self.content_scorer.score_content(content_item, content_source, context)
    
    def _build_content_context(
        self,
        content_item: ContentItem,
        content_source: ContentSource,
        compute_analytics: bool = True,
    ) -> Dict[str, Any]:
        """Build context dict for content item scoring."""
        context: Dict[str, Any] = {}
        
        # Global neighbors (for Uniqueness)
        if content_item.embedding is not None:
            global_neighbors = self.neighborhood_provider.find_global_neighbors(
                content_item.embedding,
                k=self.config.uniqueness.k_global,
                exclude_ids=[content_item.id],
            )
            context['global_neighbors'] = global_neighbors
        
        # Intra-source neighbors (for Cohesion and Learning)
        if content_item.embedding is not None:
            intra_neighbors = self.neighborhood_provider.find_intra_source_neighbors(
                content_item.embedding,
                content_source.id,
                k=self.config.cohesion.k_intra,
                exclude_ids=[content_item.id],
            )
            # Extract just the content items
            context['intra_neighbors'] = [item for item, _ in intra_neighbors]
        
        # Prior content item (for Learning)
        context['prior_content'] = content_source.get_prior_content(content_item)
        
        # Analytics normalization (for Quality)
        if compute_analytics:
            # TODO: Implement proper normalization
            # For now, use dummy values
            context['wtpi_distribution'] = {'mean': 100.0, 'std': 50.0}
            context['cr_distribution'] = {'mean': 0.5, 'std': 0.2}
        
        return context
    
    def score_source(
        self,
        content_source: ContentSource,
        end_time: Optional[datetime] = None,
    ) -> ContentSourceRankScore:
        """Score a content source.
        
        Args:
            content_source: Content source to score
            end_time: End of evaluation window
            
        Returns:
            ContentSourceRankScore
        """
        # Ensure all content items are processed
        self.process_content_batch(content_source.content_items)
        
        # Score all content items in the window
        items_in_window = content_source.get_content_in_window(
            end_time or datetime.utcnow(),
            window_days=self.config.content_source.window_days,
        )
        
        content_scores = {}
        for item in items_in_window:
            score = self.score_content_item(item, content_source, compute_analytics_context=True)
            content_scores[item.id] = score
        
        # Score source
        return self.source_scorer.score_source(
            content_source,
            end_time=end_time,
            content_scores=content_scores,
        )
    
    def score_sources_with_network(
        self,
        content_sources: List[ContentSource],
        reference_time: Optional[datetime] = None,
    ) -> Dict[str, KnowledgeRankScore]:
        """Score multiple content sources with network effects.
        
        Args:
            content_sources: Content sources to score
            reference_time: Reference time for evaluation
            
        Returns:
            Dict mapping content_source_id to KnowledgeRankScore
        """
        # Process all content items
        all_items = [item for source in content_sources for item in source.content_items]
        self.process_content_batch(all_items)
        self.index_content(all_items)
        
        # Score all sources
        logger.info(f"Scoring {len(content_sources)} content sources")
        source_scores = {}
        for source in content_sources:
            score = self.score_source(source, end_time=reference_time)
            source_scores[source.id] = score
        
        # Compute network layer
        if self.config.network.enabled:
            logger.info("Computing KnowledgeRank network layer")
            kr_scores = self.network_computer.compute_knowledge_rank(
                content_sources,
                source_scores,
                reference_time,
            )
        else:
            # Convert to KnowledgeRank format without network
            kr_scores = {
                source.id: KnowledgeRankScore(
                    content_source_id=source.id,
                    knowledge_rank=source_scores[source.id].score,
                    idea_rank=source_scores[source.id].score,
                    influence_bonus=0.0,
                    outgoing_influence=[],
                    incoming_influence=[],
                )
                for source in content_sources
            }
        
        return kr_scores


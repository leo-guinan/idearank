"""End-to-end pipeline for IdeaRank computation.

Orchestrates:
1. Embedding generation
2. Topic modeling
3. Neighborhood search
4. Factor computation
5. Video & channel scoring
6. Optional network layer
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from idearank.models import Video, Channel
from idearank.config import IdeaRankConfig
from idearank.scorer import IdeaRankScorer, IdeaRankScore, ChannelScorer, ChannelRankScore
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
        self.video_scorer = IdeaRankScorer(config)
        self.channel_scorer = ChannelScorer(config)
        self.network_computer = KnowledgeRankComputer(config.network)
    
    def process_video(self, video: Video) -> Video:
        """Add embeddings and topics to a video.
        
        Modifies video in-place and returns it.
        """
        if video.embedding is None:
            logger.debug(f"Generating embedding for video {video.id}")
            video.embedding = self.embedding_provider.embed(video.full_text)
        
        if video.topic_mixture is None:
            logger.debug(f"Generating topics for video {video.id}")
            video.topic_mixture = self.topic_provider.get_topics(video.full_text)
        
        return video
    
    def process_videos_batch(self, videos: List[Video]) -> List[Video]:
        """Process multiple videos efficiently."""
        # Generate embeddings for videos that don't have them
        videos_needing_embeddings = [v for v in videos if v.embedding is None]
        if videos_needing_embeddings:
            logger.info(f"Generating embeddings for {len(videos_needing_embeddings)} videos")
            texts = [v.full_text for v in videos_needing_embeddings]
            embeddings = self.embedding_provider.embed_batch(texts)
            for video, embedding in zip(videos_needing_embeddings, embeddings):
                video.embedding = embedding
        
        # Generate topics
        videos_needing_topics = [v for v in videos if v.topic_mixture is None]
        if videos_needing_topics:
            logger.info(f"Generating topics for {len(videos_needing_topics)} videos")
            texts = [v.full_text for v in videos_needing_topics]
            topics = self.topic_provider.get_topics_batch(texts)
            for video, topic in zip(videos_needing_topics, topics):
                video.topic_mixture = topic
        
        return videos
    
    def index_videos(self, videos: List[Video]) -> None:
        """Add videos to the neighborhood index."""
        logger.info(f"Indexing {len(videos)} videos for ANN search")
        self.neighborhood_provider.index_videos_batch(videos)
    
    def score_video(
        self,
        video: Video,
        channel: Channel,
        compute_analytics_context: bool = True,
    ) -> IdeaRankScore:
        """Score a single video.
        
        Args:
            video: Video to score
            channel: Channel containing the video
            compute_analytics_context: Whether to compute analytics normalization
            
        Returns:
            IdeaRankScore
        """
        # Ensure video has embeddings and topics
        self.process_video(video)
        
        # Build context
        context = self._build_video_context(video, channel, compute_analytics_context)
        
        # Score
        return self.video_scorer.score_video(video, channel, context)
    
    def _build_video_context(
        self,
        video: Video,
        channel: Channel,
        compute_analytics: bool = True,
    ) -> Dict[str, Any]:
        """Build context dict for video scoring."""
        context: Dict[str, Any] = {}
        
        # Global neighbors (for Uniqueness)
        if video.embedding is not None:
            global_neighbors = self.neighborhood_provider.find_global_neighbors(
                video.embedding,
                k=self.config.uniqueness.k_global,
                exclude_ids=[video.id],
            )
            context['global_neighbors'] = global_neighbors
        
        # Intra-channel neighbors (for Cohesion and Learning)
        if video.embedding is not None:
            intra_neighbors = self.neighborhood_provider.find_intra_channel_neighbors(
                video.embedding,
                channel.id,
                k=self.config.cohesion.k_intra,
                exclude_ids=[video.id],
            )
            # Extract just the videos
            context['intra_neighbors'] = [v for v, _ in intra_neighbors]
        
        # Prior video (for Learning)
        context['prior_video'] = channel.get_prior_video(video)
        
        # Analytics normalization (for Quality)
        if compute_analytics:
            # TODO: Implement proper normalization
            # For now, use dummy values
            context['wtpi_distribution'] = {'mean': 100.0, 'std': 50.0}
            context['cr_distribution'] = {'mean': 0.5, 'std': 0.2}
        
        return context
    
    def score_channel(
        self,
        channel: Channel,
        end_time: Optional[datetime] = None,
    ) -> ChannelRankScore:
        """Score a channel.
        
        Args:
            channel: Channel to score
            end_time: End of evaluation window
            
        Returns:
            ChannelRankScore
        """
        # Ensure all videos are processed
        self.process_videos_batch(channel.videos)
        
        # Score all videos in the window
        videos_in_window = channel.get_videos_in_window(
            end_time or datetime.utcnow(),
            window_days=self.config.channel.window_days,
        )
        
        video_scores = {}
        for video in videos_in_window:
            score = self.score_video(video, channel, compute_analytics_context=True)
            video_scores[video.id] = score
        
        # Score channel
        return self.channel_scorer.score_channel(
            channel,
            end_time=end_time,
            video_scores=video_scores,
        )
    
    def score_channels_with_network(
        self,
        channels: List[Channel],
        reference_time: Optional[datetime] = None,
    ) -> Dict[str, KnowledgeRankScore]:
        """Score multiple channels with network effects.
        
        Args:
            channels: Channels to score
            reference_time: Reference time for evaluation
            
        Returns:
            Dict mapping channel_id to KnowledgeRankScore
        """
        # Process all videos
        all_videos = [v for c in channels for v in c.videos]
        self.process_videos_batch(all_videos)
        self.index_videos(all_videos)
        
        # Score all channels
        logger.info(f"Scoring {len(channels)} channels")
        channel_scores = {}
        for channel in channels:
            score = self.score_channel(channel, end_time=reference_time)
            channel_scores[channel.id] = score
        
        # Compute network layer
        if self.config.network.enabled:
            logger.info("Computing KnowledgeRank network layer")
            kr_scores = self.network_computer.compute_knowledge_rank(
                channels,
                channel_scores,
                reference_time,
            )
        else:
            # Convert to KnowledgeRank format without network
            kr_scores = {
                c.id: KnowledgeRankScore(
                    channel_id=c.id,
                    knowledge_rank=channel_scores[c.id].score,
                    idea_rank=channel_scores[c.id].score,
                    influence_bonus=0.0,
                    outgoing_influence=[],
                    incoming_influence=[],
                )
                for c in channels
            }
        
        return kr_scores


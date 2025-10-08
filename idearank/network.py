"""KnowledgeRank: Optional network layer for cross-channel idea influence.

Similar to PageRank, but measures "who learns from whom" rather than "who links to whom".

KR_j(t) = (1-d)·IR_S_j(t) + d·Σ_i KR_i(t)·IFR(i→j,t)

where IFR (Idea Flow Rate) measures semantic influence between channels.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np

from idearank.models import Video, Channel
from idearank.config import NetworkConfig
from idearank.scorer import ChannelRankScore


@dataclass
class InfluenceEdge:
    """Represents idea flow from one channel to another."""
    
    source_channel_id: str
    target_channel_id: str
    influence_score: float  # IFR(i→j,t)
    evidence_pairs: List[Tuple[str, str]]  # (source_video_id, target_video_id)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'source': self.source_channel_id,
            'target': self.target_channel_id,
            'influence': self.influence_score,
            'evidence_count': len(self.evidence_pairs),
        }


@dataclass
class KnowledgeRankScore:
    """Result of KnowledgeRank computation."""
    
    channel_id: str
    knowledge_rank: float  # Final KR score
    idea_rank: float  # Base IR_S score
    influence_bonus: float  # Contribution from incoming influence
    
    outgoing_influence: List[InfluenceEdge]
    incoming_influence: List[InfluenceEdge]
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'channel_id': self.channel_id,
            'knowledge_rank': self.knowledge_rank,
            'idea_rank': self.idea_rank,
            'influence_bonus': self.influence_bonus,
            'outgoing_edges': len(self.outgoing_influence),
            'incoming_edges': len(self.incoming_influence),
        }


class KnowledgeRankComputer:
    """Computes KnowledgeRank scores across a network of channels."""
    
    def __init__(self, config: NetworkConfig):
        """Initialize with network configuration."""
        self.config = config
    
    def compute_influence_graph(
        self,
        channels: List[Channel],
        reference_time: Optional[datetime] = None,
    ) -> Dict[str, List[InfluenceEdge]]:
        """Compute the influence graph between channels.
        
        IFR(i→j,t) = mean[max(0, cos(e_v_j(t), e_v_i(t-Δ)) - θ)]
        
        Returns adjacency list of outgoing influences.
        """
        if reference_time is None:
            # Use latest video time across all channels
            all_times = [
                v.published_at
                for c in channels
                for v in c.videos
            ]
            if not all_times:
                return {}
            reference_time = max(all_times)
        
        # Build influence graph
        influence_graph: Dict[str, List[InfluenceEdge]] = {c.id: [] for c in channels}
        
        # For each pair of channels
        for i, source_channel in enumerate(channels):
            for j, target_channel in enumerate(channels):
                if i == j:
                    continue  # Skip self-influence
                
                # Compute IFR(source → target)
                influence = self._compute_influence(
                    source_channel,
                    target_channel,
                    reference_time,
                )
                
                if influence is not None and influence.influence_score > 0:
                    influence_graph[source_channel.id].append(influence)
        
        return influence_graph
    
    def _compute_influence(
        self,
        source: Channel,
        target: Channel,
        reference_time: datetime,
    ) -> Optional[InfluenceEdge]:
        """Compute influence from source channel to target channel."""
        # Get source videos (from past, up to max_lag_days before reference)
        min_source_time = reference_time - timedelta(days=self.config.max_lag_days)
        source_videos = [
            v for v in source.videos
            if min_source_time <= v.published_at <= reference_time
            and v.embedding is not None
        ]
        
        # Get target videos (recent, around reference time)
        window = timedelta(days=30)  # Look at target videos near reference
        target_videos = [
            v for v in target.videos
            if reference_time - window <= v.published_at <= reference_time + window
            and v.embedding is not None
        ]
        
        if not source_videos or not target_videos:
            return None
        
        # Compute pairwise similarities with lag
        influences = []
        evidence_pairs = []
        
        for target_video in target_videos:
            for source_video in source_videos:
                # Check that source came before target (with lag)
                lag = (target_video.published_at - source_video.published_at).days
                if 0 < lag <= self.config.max_lag_days:
                    # Compute similarity
                    sim = target_video.embedding.cosine_similarity(source_video.embedding)
                    
                    # Apply threshold
                    influence_score = max(0.0, sim - self.config.influence_threshold)
                    
                    if influence_score > 0:
                        influences.append(influence_score)
                        evidence_pairs.append((source_video.id, target_video.id))
        
        if not influences:
            return None
        
        # Average influence
        mean_influence = float(np.mean(influences))
        
        return InfluenceEdge(
            source_channel_id=source.id,
            target_channel_id=target.id,
            influence_score=mean_influence,
            evidence_pairs=evidence_pairs,
        )
    
    def compute_knowledge_rank(
        self,
        channels: List[Channel],
        channel_scores: Dict[str, ChannelRankScore],
        reference_time: Optional[datetime] = None,
    ) -> Dict[str, KnowledgeRankScore]:
        """Compute KnowledgeRank for all channels using power iteration.
        
        KR_j = (1-d)·IR_S_j + d·Σ_i KR_i·IFR(i→j)
        
        Args:
            channels: List of channels to rank
            channel_scores: Pre-computed IdeaRank scores
            reference_time: Time point for evaluation
            
        Returns:
            Dict mapping channel_id to KnowledgeRankScore
        """
        if not self.config.enabled:
            # Network layer disabled - just return IR scores as KR
            return {
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
        
        # Compute influence graph
        influence_graph = self.compute_influence_graph(channels, reference_time)
        
        # Build reverse graph (incoming influences)
        incoming_graph: Dict[str, List[InfluenceEdge]] = {c.id: [] for c in channels}
        for source_id, edges in influence_graph.items():
            for edge in edges:
                incoming_graph[edge.target_channel_id].append(edge)
        
        # Initialize KR values with IR scores
        kr_values = {c.id: channel_scores[c.id].score for c in channels}
        
        # Power iteration
        for iteration in range(self.config.max_iterations):
            kr_values_new = {}
            
            for channel in channels:
                # Teleportation (base IR score)
                base_score = (1 - self.config.damping_factor) * channel_scores[channel.id].score
                
                # Influence from incoming edges
                influence_score = 0.0
                for edge in incoming_graph[channel.id]:
                    source_kr = kr_values[edge.source_channel_id]
                    influence_score += source_kr * edge.influence_score
                
                influence_score *= self.config.damping_factor
                
                kr_values_new[channel.id] = base_score + influence_score
            
            # Check convergence
            max_change = max(
                abs(kr_values_new[c.id] - kr_values[c.id])
                for c in channels
            )
            
            kr_values = kr_values_new
            
            if max_change < self.config.convergence_tolerance:
                break
        
        # Build results
        results = {}
        for channel in channels:
            kr = kr_values[channel.id]
            ir = channel_scores[channel.id].score
            
            results[channel.id] = KnowledgeRankScore(
                channel_id=channel.id,
                knowledge_rank=kr,
                idea_rank=ir,
                influence_bonus=kr - ir,
                outgoing_influence=influence_graph[channel.id],
                incoming_influence=incoming_graph[channel.id],
            )
        
        return results


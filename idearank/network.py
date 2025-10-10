"""KnowledgeRank: Optional network layer for cross-source idea influence.

Similar to PageRank, but measures "who learns from whom" rather than "who links to whom".

KR_j(t) = (1-d)·IR_S_j(t) + d·Σ_i KR_i(t)·IFR(i→j,t)

where IFR (Idea Flow Rate) measures semantic influence between content sources.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np

from idearank.models import ContentItem, ContentSource
from idearank.config import NetworkConfig
from idearank.scorer import ContentSourceRankScore


@dataclass
class InfluenceEdge:
    """Represents idea flow from one content source to another."""
    
    source_id: str
    target_id: str
    influence_score: float  # IFR(i→j,t)
    evidence_pairs: List[Tuple[str, str]]  # (source_item_id, target_item_id)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'source': self.source_id,
            'target': self.target_id,
            'influence': self.influence_score,
            'evidence_count': len(self.evidence_pairs),
        }


@dataclass
class KnowledgeRankScore:
    """Result of KnowledgeRank computation."""
    
    content_source_id: str
    knowledge_rank: float  # Final KR score
    idea_rank: float  # Base IR_S score
    influence_bonus: float  # Contribution from incoming influence
    
    outgoing_influence: List[InfluenceEdge]
    incoming_influence: List[InfluenceEdge]
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'content_source_id': self.content_source_id,
            'knowledge_rank': self.knowledge_rank,
            'idea_rank': self.idea_rank,
            'influence_bonus': self.influence_bonus,
            'outgoing_edges': len(self.outgoing_influence),
            'incoming_edges': len(self.incoming_influence),
        }


class KnowledgeRankComputer:
    """Computes KnowledgeRank scores across a network of content sources."""
    
    def __init__(self, config: NetworkConfig):
        """Initialize with network configuration."""
        self.config = config
    
    def compute_influence_graph(
        self,
        content_sources: List[ContentSource],
        reference_time: Optional[datetime] = None,
    ) -> Dict[str, List[InfluenceEdge]]:
        """Compute the influence graph between content sources.
        
        IFR(i→j,t) = mean[max(0, cos(e_item_j(t), e_item_i(t-Δ)) - θ)]
        
        Returns adjacency list of outgoing influences.
        """
        if reference_time is None:
            # Use latest content time across all sources
            all_times = [
                item.published_at
                for source in content_sources
                for item in source.content_items
            ]
            if not all_times:
                return {}
            reference_time = max(all_times)
        
        # Build influence graph
        influence_graph: Dict[str, List[InfluenceEdge]] = {source.id: [] for source in content_sources}
        
        # For each pair of sources
        for i, source in enumerate(content_sources):
            for j, target in enumerate(content_sources):
                if i == j:
                    continue  # Skip self-influence
                
                # Compute IFR(source → target)
                influence = self._compute_influence(
                    source,
                    target,
                    reference_time,
                )
                
                if influence is not None and influence.influence_score > 0:
                    influence_graph[source.id].append(influence)
        
        return influence_graph
    
    def _compute_influence(
        self,
        source: ContentSource,
        target: ContentSource,
        reference_time: datetime,
    ) -> Optional[InfluenceEdge]:
        """Compute influence from source to target content source."""
        # Get source items (from past, up to max_lag_days before reference)
        min_source_time = reference_time - timedelta(days=self.config.max_lag_days)
        source_items = [
            item for item in source.content_items
            if min_source_time <= item.published_at <= reference_time
            and item.embedding is not None
        ]
        
        # Get target items (recent, around reference time)
        window = timedelta(days=30)  # Look at target items near reference
        target_items = [
            item for item in target.content_items
            if reference_time - window <= item.published_at <= reference_time + window
            and item.embedding is not None
        ]
        
        if not source_items or not target_items:
            return None
        
        # Compute pairwise similarities with lag
        influences = []
        evidence_pairs = []
        
        for target_item in target_items:
            for source_item in source_items:
                # Check that source came before target (with lag)
                lag = (target_item.published_at - source_item.published_at).days
                if 0 < lag <= self.config.max_lag_days:
                    # Compute similarity
                    sim = target_item.embedding.cosine_similarity(source_item.embedding)
                    
                    # Apply threshold
                    influence_score = max(0.0, sim - self.config.influence_threshold)
                    
                    if influence_score > 0:
                        influences.append(influence_score)
                        evidence_pairs.append((source_item.id, target_item.id))
        
        if not influences:
            return None
        
        # Average influence
        mean_influence = float(np.mean(influences))
        
        return InfluenceEdge(
            source_id=source.id,
            target_id=target.id,
            influence_score=mean_influence,
            evidence_pairs=evidence_pairs,
        )
    
    def compute_knowledge_rank(
        self,
        content_sources: List[ContentSource],
        source_scores: Dict[str, ContentSourceRankScore],
        reference_time: Optional[datetime] = None,
    ) -> Dict[str, KnowledgeRankScore]:
        """Compute KnowledgeRank for all content sources using power iteration.
        
        KR_j = (1-d)·IR_S_j + d·Σ_i KR_i·IFR(i→j)
        
        Args:
            content_sources: List of content sources to rank
            source_scores: Pre-computed IdeaRank scores
            reference_time: Time point for evaluation
            
        Returns:
            Dict mapping content_source_id to KnowledgeRankScore
        """
        if not self.config.enabled:
            # Network layer disabled - just return IR scores as KR
            return {
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
        
        # Compute influence graph
        influence_graph = self.compute_influence_graph(content_sources, reference_time)
        
        # Build reverse graph (incoming influences)
        incoming_graph: Dict[str, List[InfluenceEdge]] = {source.id: [] for source in content_sources}
        for source_id, edges in influence_graph.items():
            for edge in edges:
                incoming_graph[edge.target_id].append(edge)
        
        # Initialize KR values with IR scores
        kr_values = {source.id: source_scores[source.id].score for source in content_sources}
        
        # Power iteration
        for iteration in range(self.config.max_iterations):
            kr_values_new = {}
            
            for source in content_sources:
                # Teleportation (base IR score)
                base_score = (1 - self.config.damping_factor) * source_scores[source.id].score
                
                # Influence from incoming edges
                influence_score = 0.0
                for edge in incoming_graph[source.id]:
                    source_kr = kr_values[edge.source_id]
                    influence_score += source_kr * edge.influence_score
                
                influence_score *= self.config.damping_factor
                
                kr_values_new[source.id] = base_score + influence_score
            
            # Check convergence
            max_change = max(
                abs(kr_values_new[source.id] - kr_values[source.id])
                for source in content_sources
            )
            
            kr_values = kr_values_new
            
            if max_change < self.config.convergence_tolerance:
                break
        
        # Build results
        results = {}
        for source in content_sources:
            kr = kr_values[source.id]
            ir = source_scores[source.id].score
            
            results[source.id] = KnowledgeRankScore(
                content_source_id=source.id,
                knowledge_rank=kr,
                idea_rank=ir,
                influence_bonus=kr - ir,
                outgoing_influence=influence_graph[source.id],
                incoming_influence=incoming_graph[source.id],
            )
        
        return results


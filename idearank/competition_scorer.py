"""IdeaRank-Thought scoring engine for competition system.

Implements the computational methods from the technical appendix:
- Gate formulas (Outcome Validity, Constraint Compliance)
- Factor computation (U, C, L, Q, T, D)
- Coaching signal processing
- Anti-gaming measures
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass

from idearank.competition_models import (
    Match, ReasoningTrace, ReasoningNode, CoachingEvent, 
    FactorType, OutcomeValidity, ConstraintCompliance, 
    IdeaRankThoughtScore, Player, Challenge
)
from idearank.providers.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)


class IdeaRankThoughtScorer:
    """Main scorer for IdeaRank-Thought competition system."""
    
    def __init__(self, embedding_provider: EmbeddingProvider):
        """Initialize the scorer.
        
        Args:
            embedding_provider: For semantic similarity calculations
        """
        self.embedding_provider = embedding_provider
    
    def score_match(self, match: Match, challenge: Challenge) -> Dict[str, IdeaRankThoughtScore]:
        """Score a complete match for all players.
        
        Args:
            match: The match to score
            challenge: The challenge that was solved
            
        Returns:
            Dict mapping player_id to IdeaRankThoughtScore
        """
        scores = {}
        
        # Score each player
        if match.player1_trace:
            scores[match.player1_id] = self._score_player(
                match, match.player1_trace, challenge, match.player1_id
            )
        
        if match.player2_trace and match.player2_id:
            scores[match.player2_id] = self._score_player(
                match, match.player2_trace, challenge, match.player2_id
            )
        
        return scores
    
    def _score_player(
        self, 
        match: Match, 
        trace: ReasoningTrace, 
        challenge: Challenge,
        player_id: str
    ) -> IdeaRankThoughtScore:
        """Score a single player's performance."""
        
        # Calculate gate scores
        outcome_validity = self._calculate_outcome_validity(trace, challenge)
        constraint_compliance = self._calculate_constraint_compliance(trace, challenge)
        
        # Calculate factor scores
        factor_scores = self._calculate_factor_scores(trace, match, player_id)
        
        # Calculate raw IR-T score with anti-gaming (multiplicative)
        raw_score = self._calculate_raw_score(factor_scores, outcome_validity, constraint_compliance)
        
        # Calculate coaching impact
        coach_impact = self._calculate_coach_impact(match, trace, player_id)
        
        # Calculate plan adherence
        plan_adherence = self._calculate_plan_adherence(match, trace, player_id)
        
        # Calculate timeout deltas
        timeout_deltas = self._calculate_timeout_deltas(match, trace, player_id)
        
        # Calculate entropy index
        entropy_index = trace.calculate_entropy_index()
        
        # Apply class adjustments (placeholder - would depend on player class/level)
        class_adjusted_score = raw_score  # TODO: Implement class-specific adjustments
        
        return IdeaRankThoughtScore(
            match_id=match.id,
            player_id=player_id,
            outcome_validity=outcome_validity,
            constraint_compliance=constraint_compliance,
            factor_scores=factor_scores,
            raw_score=raw_score,
            class_adjusted_score=class_adjusted_score,
            coach_impact_index=coach_impact,
            meta_rating=0.0,  # Would be calculated from historical performance
            entropy_index=entropy_index,
            plan_adherence=plan_adherence,
            timeout_deltas=timeout_deltas,
        )
    
    def _calculate_outcome_validity(
        self, 
        trace: ReasoningTrace, 
        challenge: Challenge
    ) -> OutcomeValidity:
        """Calculate Outcome Validity (O) gate.
        
        O = 0.7 × correctness + 0.3 × robustness
        """
        
        # Calculate correctness based on final solution quality
        correctness = self._evaluate_correctness(trace, challenge)
        
        # Calculate robustness based on stability under variation
        robustness = self._evaluate_robustness(trace, challenge)
        
        return OutcomeValidity(
            correctness=correctness,
            robustness=robustness
        )
    
    def _evaluate_correctness(self, trace: ReasoningTrace, challenge: Challenge) -> float:
        """Evaluate solution correctness against ground truth."""
        if not trace.nodes:
            return 0.0
        
        # Get the final reasoning node (solution)
        final_node = trace.nodes[-1]
        
        # Compare against ground truth if available
        if challenge.ground_truth:
            # Use embedding similarity for semantic correctness
            solution_embedding = self.embedding_provider.embed(final_node.content)
            ground_truth_text = str(challenge.ground_truth)
            gt_embedding = self.embedding_provider.embed(ground_truth_text)
            
            # Cosine similarity
            similarity = np.dot(solution_embedding.vector, gt_embedding.vector) / (
                np.linalg.norm(solution_embedding.vector) * np.linalg.norm(gt_embedding.vector)
            )
            
            # Convert to 0-1 scale
            correctness = max(0.0, min(1.0, (similarity + 1) / 2))
        else:
            # Fallback: use confidence and completeness heuristics
            confidence_score = final_node.confidence
            completeness_score = min(1.0, len(final_node.content) / 100)  # Rough heuristic
            correctness = (confidence_score + completeness_score) / 2
        
        return correctness
    
    def _evaluate_robustness(self, trace: ReasoningTrace, challenge: Challenge) -> float:
        """Evaluate robustness under variation and edge cases."""
        if not trace.nodes:
            return 0.0
        
        # Measure consistency across reasoning path
        confidence_scores = [node.confidence for node in trace.nodes]
        confidence_variance = np.var(confidence_scores)
        
        # Lower variance = higher robustness
        robustness = max(0.0, 1.0 - confidence_variance)
        
        # Check for edge case handling
        edge_case_handling = self._check_edge_case_handling(trace, challenge)
        
        # Combine consistency and edge case handling
        return 0.6 * robustness + 0.4 * edge_case_handling
    
    def _check_edge_case_handling(self, trace: ReasoningTrace, challenge: Challenge) -> float:
        """Check how well the reasoning handles edge cases."""
        # Look for conditional reasoning and error handling
        edge_case_indicators = 0
        total_nodes = len(trace.nodes)
        
        if total_nodes == 0:
            return 0.0
        
        for node in trace.nodes:
            content_lower = node.content.lower()
            # Look for conditional reasoning
            if any(phrase in content_lower for phrase in ['if', 'else', 'unless', 'however', 'but']):
                edge_case_indicators += 1
            # Look for error handling
            if any(phrase in content_lower for phrase in ['error', 'exception', 'fail', 'invalid']):
                edge_case_indicators += 1
        
        return min(1.0, edge_case_indicators / total_nodes * 2)
    
    def _calculate_constraint_compliance(
        self, 
        trace: ReasoningTrace, 
        challenge: Challenge
    ) -> ConstraintCompliance:
        """Calculate Constraint Compliance (X) gate.
        
        X = 1 - violation_penalty
        """
        violations = []
        
        # Check against challenge constraints
        for constraint in challenge.constraints:
            violation = self._check_constraint_violation(trace, constraint)
            if violation:
                violations.append(violation)
        
        # Check for resource constraint violations
        resource_violations = self._check_resource_violations(trace, challenge)
        violations.extend(resource_violations)
        
        # Check for ethical constraint violations
        ethical_violations = self._check_ethical_violations(trace)
        violations.extend(ethical_violations)
        
        return ConstraintCompliance(violations=violations)
    
    def _check_constraint_violation(self, trace: ReasoningTrace, constraint: str) -> Optional[Dict[str, Any]]:
        """Check if a specific constraint was violated."""
        # Simple keyword-based violation detection
        # In practice, this would use more sophisticated NLP
        
        constraint_lower = constraint.lower()
        
        for node in trace.nodes:
            content_lower = node.content.lower()
            
            # Check for violation patterns
            if 'not allowed' in constraint_lower:
                forbidden_terms = constraint_lower.replace('not allowed', '').strip()
                if forbidden_terms in content_lower:
                    return {
                        'type': 'constraint_violation',
                        'constraint': constraint,
                        'node_id': node.id,
                        'severity': 0.5,
                        'frequency': 1,
                    }
        
        return None
    
    def _check_resource_violations(self, trace: ReasoningTrace, challenge: Challenge) -> List[Dict[str, Any]]:
        """Check for resource constraint violations."""
        violations = []
        
        # Check for excessive reasoning steps (resource constraint)
        max_reasoning_steps = challenge.metadata.get('max_reasoning_steps', 50)
        if len(trace.nodes) > max_reasoning_steps:
            violations.append({
                'type': 'resource_violation',
                'constraint': 'max_reasoning_steps',
                'actual': len(trace.nodes),
                'limit': max_reasoning_steps,
                'severity': min(1.0, (len(trace.nodes) - max_reasoning_steps) / 20),
                'frequency': 1,
            })
        
        return violations
    
    def _check_ethical_violations(self, trace: ReasoningTrace) -> List[Dict[str, Any]]:
        """Check for ethical constraint violations."""
        violations = []
        
        # Simple ethical violation detection
        unethical_keywords = ['harm', 'deceive', 'cheat', 'exploit', 'manipulate']
        
        for node in trace.nodes:
            content_lower = node.content.lower()
            for keyword in unethical_keywords:
                if keyword in content_lower:
                    violations.append({
                        'type': 'ethical_violation',
                        'keyword': keyword,
                        'node_id': node.id,
                        'severity': 0.8,
                        'frequency': 1,
                    })
        
        return violations
    
    def _calculate_factor_scores(
        self, 
        trace: ReasoningTrace, 
        match: Match, 
        player_id: str
    ) -> Dict[FactorType, float]:
        """Calculate all factor scores (U, C, L, Q, T, D)."""
        
        factor_scores = {}
        
        # U - Uniqueness: Novelty vs league corpus
        factor_scores[FactorType.UNIQUENESS] = self._calculate_uniqueness(trace)
        
        # C - Cohesion: Reasoning map structural consistency
        factor_scores[FactorType.COHESION] = self._calculate_cohesion(trace)
        
        # L - Learning: Skill gain over time
        factor_scores[FactorType.LEARNING] = self._calculate_learning(trace, match, player_id)
        
        # Q - Quality: Readability, completeness, clarity
        factor_scores[FactorType.QUALITY] = self._calculate_quality(trace)
        
        # T - Trust: Pass rate × citation quality
        factor_scores[FactorType.TRUST] = self._calculate_trust(trace)
        
        # D - Density: Information-per-step compression ratio
        factor_scores[FactorType.DENSITY] = self._calculate_density(trace)
        
        return factor_scores
    
    def _calculate_uniqueness(self, trace: ReasoningTrace) -> float:
        """Calculate Uniqueness (U) factor.
        
        Novelty z-score → sigmoid normalization
        """
        if not trace.nodes:
            return 0.0
        
        # For now, use reasoning path uniqueness heuristics
        # In practice, would compare against league corpus
        
        # Measure reasoning path diversity
        unique_concepts = set()
        for node in trace.nodes:
            # Extract concepts (simplified)
            words = node.content.lower().split()
            unique_concepts.update(words)
        
        # Calculate uniqueness score based on concept diversity
        total_concepts = len(unique_concepts)
        total_content = sum(len(node.content.split()) for node in trace.nodes)
        
        if total_content == 0:
            return 0.0
        
        uniqueness_ratio = total_concepts / total_content
        
        # Apply sigmoid normalization
        import math
        uniqueness_score = 1 / (1 + math.exp(-5 * (uniqueness_ratio - 0.5)))
        
        return min(1.0, max(0.0, uniqueness_score))
    
    def _calculate_cohesion(self, trace: ReasoningTrace) -> float:
        """Calculate Cohesion (C) factor.
        
        Structural consistency, linguistic coherence
        """
        if len(trace.nodes) < 2:
            return 0.0
        
        # Measure structural consistency
        structural_score = self._measure_structural_consistency(trace)
        
        # Measure linguistic coherence
        coherence_score = self._measure_linguistic_coherence(trace)
        
        return 0.5 * structural_score + 0.5 * coherence_score
    
    def _measure_structural_consistency(self, trace: ReasoningTrace) -> float:
        """Measure structural consistency of reasoning graph."""
        if len(trace.nodes) < 2:
            return 0.0
        
        # Calculate graph connectivity
        total_edges = sum(len(node.child_ids) for node in trace.nodes)
        total_possible_edges = len(trace.nodes) * (len(trace.nodes) - 1)
        
        if total_possible_edges == 0:
            return 0.0
        
        connectivity = total_edges / total_possible_edges
        
        # Check for logical flow (no orphaned nodes)
        orphaned_nodes = 0
        for node in trace.nodes:
            if not node.parent_ids and not node.child_ids:
                orphaned_nodes += 1
        
        orphan_penalty = orphaned_nodes / len(trace.nodes)
        
        return max(0.0, connectivity - orphan_penalty)
    
    def _measure_linguistic_coherence(self, trace: ReasoningTrace) -> float:
        """Measure linguistic coherence across reasoning steps."""
        if len(trace.nodes) < 2:
            return 0.0
        
        # Calculate semantic similarity between consecutive nodes
        similarities = []
        
        for i in range(len(trace.nodes) - 1):
            node1 = trace.nodes[i]
            node2 = trace.nodes[i + 1]
            
            # Use embedding similarity
            emb1 = self.embedding_provider.embed(node1.content)
            emb2 = self.embedding_provider.embed(node2.content)
            
            similarity = np.dot(emb1.vector, emb2.vector) / (
                np.linalg.norm(emb1.vector) * np.linalg.norm(emb2.vector)
            )
            
            similarities.append(similarity)
        
        if not similarities:
            return 0.0
        
        # Average similarity represents coherence
        coherence = np.mean(similarities)
        
        # Convert to 0-1 scale
        return max(0.0, min(1.0, (coherence + 1) / 2))
    
    def _calculate_learning(self, trace: ReasoningTrace, match: Match, player_id: str) -> float:
        """Calculate Learning (L) factor.
        
        Skill gain function over time
        """
        if len(trace.nodes) < 3:
            return 0.0
        
        # Measure improvement in reasoning quality over time
        # Use confidence and factor contributions as proxy for skill
        
        early_nodes = trace.nodes[:len(trace.nodes)//3]
        late_nodes = trace.nodes[2*len(trace.nodes)//3:]
        
        if not early_nodes or not late_nodes:
            return 0.0
        
        # Calculate average confidence
        early_confidence = np.mean([node.confidence for node in early_nodes])
        late_confidence = np.mean([node.confidence for node in late_nodes])
        
        confidence_improvement = late_confidence - early_confidence
        
        # Calculate factor contribution improvement
        early_factor_quality = np.mean([
            np.mean(list(node.factor_contributions.values())) 
            for node in early_nodes if node.factor_contributions
        ])
        late_factor_quality = np.mean([
            np.mean(list(node.factor_contributions.values())) 
            for node in late_nodes if node.factor_contributions
        ])
        
        factor_improvement = late_factor_quality - early_factor_quality
        
        # Combine improvements
        learning_score = 0.6 * confidence_improvement + 0.4 * factor_improvement
        
        return max(0.0, min(1.0, (learning_score + 1) / 2))
    
    def _calculate_quality(self, trace: ReasoningTrace) -> float:
        """Calculate Quality (Q) factor.
        
        Readability, completeness, clarity metrics
        """
        if not trace.nodes:
            return 0.0
        
        # Measure readability
        readability_score = self._measure_readability(trace)
        
        # Measure completeness
        completeness_score = self._measure_completeness(trace)
        
        # Measure clarity
        clarity_score = self._measure_clarity(trace)
        
        return (readability_score + completeness_score + clarity_score) / 3
    
    def _measure_readability(self, trace: ReasoningTrace) -> float:
        """Measure readability of reasoning."""
        if not trace.nodes:
            return 0.0
        
        # Simple readability heuristics
        total_sentences = 0
        total_words = 0
        complex_words = 0
        
        for node in trace.nodes:
            sentences = node.content.split('.')
            total_sentences += len(sentences)
            
            words = node.content.split()
            total_words += len(words)
            
            # Count complex words (length > 6)
            complex_words += sum(1 for word in words if len(word) > 6)
        
        if total_words == 0:
            return 0.0
        
        # Calculate readability metrics
        avg_words_per_sentence = total_words / max(1, total_sentences)
        complex_word_ratio = complex_words / total_words
        
        # Simple readability score (lower is better for these metrics)
        readability = max(0.0, 1.0 - (avg_words_per_sentence / 20) - (complex_word_ratio / 2))
        
        return min(1.0, readability)
    
    def _measure_completeness(self, trace: ReasoningTrace) -> float:
        """Measure completeness of reasoning."""
        if not trace.nodes:
            return 0.0
        
        # Check for reasoning gaps
        gaps = 0
        for node in trace.nodes:
            if len(node.parent_ids) == 0 and len(node.child_ids) == 0:
                gaps += 1
        
        completeness = 1.0 - (gaps / len(trace.nodes))
        
        # Check for sufficient detail
        avg_content_length = np.mean([len(node.content) for node in trace.nodes])
        detail_score = min(1.0, avg_content_length / 50)  # 50 chars as baseline
        
        return 0.7 * completeness + 0.3 * detail_score
    
    def _measure_clarity(self, trace: ReasoningTrace) -> float:
        """Measure clarity of reasoning."""
        if not trace.nodes:
            return 0.0
        
        # Look for clear logical connectors
        clarity_indicators = 0
        total_nodes = len(trace.nodes)
        
        for node in trace.nodes:
            content_lower = node.content.lower()
            # Look for clear logical connectors
            if any(phrase in content_lower for phrase in ['therefore', 'thus', 'hence', 'because', 'since']):
                clarity_indicators += 1
        
        return min(1.0, clarity_indicators / max(1, total_nodes) * 2)
    
    def _calculate_trust(self, trace: ReasoningTrace) -> float:
        """Calculate Trust (T) factor.
        
        Pass rate × citation quality
        """
        if not trace.nodes:
            return 0.0
        
        # Calculate pass rate (based on confidence)
        pass_rate = np.mean([node.confidence for node in trace.nodes])
        
        # Calculate citation quality (based on metadata and references)
        citation_quality = self._calculate_citation_quality(trace)
        
        return pass_rate * citation_quality
    
    def _calculate_citation_quality(self, trace: ReasoningTrace) -> float:
        """Calculate citation quality across reasoning."""
        if not trace.nodes:
            return 0.0
        
        citation_scores = []
        
        for node in trace.nodes:
            # Look for citations in metadata
            citations = node.metadata.get('citations', [])
            
            if citations:
                # Simple quality heuristic based on citation count and diversity
                citation_count = len(citations)
                citation_diversity = len(set(citations))
                
                quality = min(1.0, (citation_count + citation_diversity) / 10)
                citation_scores.append(quality)
            else:
                citation_scores.append(0.0)
        
        return np.mean(citation_scores) if citation_scores else 0.0
    
    def _calculate_density(self, trace: ReasoningTrace) -> float:
        """Calculate Density (D) factor.
        
        Information-per-step compression ratio
        """
        if not trace.nodes:
            return 0.0
        
        # Calculate information content per step
        total_information = 0
        for node in trace.nodes:
            # Use content length and confidence as proxy for information
            information = len(node.content) * node.confidence
            total_information += information
        
        # Calculate compression ratio
        total_steps = len(trace.nodes)
        avg_information_per_step = total_information / total_steps
        
        # Normalize to 0-1 scale
        density = min(1.0, avg_information_per_step / 100)  # 100 as baseline
        
        return density
    
    def _calculate_raw_score(
        self, 
        factor_scores: Dict[FactorType, float],
        outcome_validity: OutcomeValidity,
        constraint_compliance: ConstraintCompliance
    ) -> float:
        """Calculate raw IR-T score with anti-gaming measures.
        
        Uses multiplicative scoring to prevent single-factor dominance.
        """
        # Check if gates pass
        if not (outcome_validity.score >= 0.5 and constraint_compliance.score >= 0.5):
            return 0.0
        
        # Multiplicative scoring for anti-gaming
        raw_score = 1.0
        
        # Apply each factor multiplicatively
        for factor, score in factor_scores.items():
            raw_score *= (0.1 + 0.9 * score)  # Scale to avoid zero multiplication
        
        # Apply gates multiplicatively
        raw_score *= outcome_validity.score
        raw_score *= constraint_compliance.score
        
        return min(1.0, max(0.0, raw_score))
    
    def _calculate_coach_impact(self, match: Match, trace: ReasoningTrace, player_id: str) -> float:
        """Calculate Coach Impact Index (CI).
        
        Measures coaching effectiveness: CI ∈ [-1.0, +1.0]
        """
        coaching_events = match.get_coaching_events_for_player(player_id)
        
        if not coaching_events:
            return 0.0
        
        # Calculate impact for each coaching event
        impacts = []
        
        for event in coaching_events:
            impact = self._calculate_single_coach_impact(event, trace)
            impacts.append(impact)
        
        # Average impact across all coaching events
        return np.mean(impacts) if impacts else 0.0
    
    def _calculate_single_coach_impact(self, event: CoachingEvent, trace: ReasoningTrace) -> float:
        """Calculate impact of a single coaching event."""
        # Find reasoning nodes before and after coaching
        pre_nodes = [node for node in trace.nodes if node.timestamp < event.timestamp]
        post_nodes = [node for node in trace.nodes if node.timestamp > event.timestamp]
        
        if not pre_nodes or not post_nodes:
            return 0.0
        
        # Calculate quality improvement
        pre_quality = np.mean([node.confidence for node in pre_nodes])
        post_quality = np.mean([node.confidence for node in post_nodes])
        
        improvement = post_quality - pre_quality
        
        # Scale to [-1.0, +1.0]
        return max(-1.0, min(1.0, improvement * 2))
    
    def _calculate_plan_adherence(self, match: Match, trace: ReasoningTrace, player_id: str) -> float:
        """Calculate plan adherence.
        
        Semantic similarity between prebrief plan and executed reasoning.
        """
        coaching_events = match.get_coaching_events_for_player(player_id)
        
        if not coaching_events:
            return 0.0
        
        adherence_scores = []
        
        for event in coaching_events:
            if event.prebrief_plan:
                # Get reasoning after this coaching event
                post_reasoning = [
                    node for node in trace.nodes 
                    if node.timestamp > event.timestamp
                ]
                
                if post_reasoning:
                    # Combine post-reasoning content
                    executed_content = ' '.join([node.content for node in post_reasoning])
                    
                    # Calculate semantic similarity
                    plan_emb = self.embedding_provider.embed(event.prebrief_plan)
                    executed_emb = self.embedding_provider.embed(executed_content)
                    
                    similarity = np.dot(plan_emb.vector, executed_emb.vector) / (
                        np.linalg.norm(plan_emb.vector) * np.linalg.norm(executed_emb.vector)
                    )
                    
                    adherence = max(0.0, min(1.0, (similarity + 1) / 2))
                    adherence_scores.append(adherence)
        
        return np.mean(adherence_scores) if adherence_scores else 0.0
    
    def _calculate_timeout_deltas(self, match: Match, trace: ReasoningTrace, player_id: str) -> List[float]:
        """Calculate timeout delta measurements.
        
        ΔIR-T = IR-T_post - IR-T_pre
        """
        coaching_events = match.get_coaching_events_for_player(player_id)
        deltas = []
        
        for event in coaching_events:
            if event.coaching_type.value == 'timeout':
                # Calculate pre and post timeout performance
                pre_nodes = [node for node in trace.nodes if node.timestamp < event.timestamp]
                post_nodes = [node for node in trace.nodes if node.timestamp > event.timestamp]
                
                if pre_nodes and post_nodes:
                    # Calculate IR-T scores for pre and post
                    pre_score = np.mean([node.confidence for node in pre_nodes])
                    post_score = np.mean([node.confidence for node in post_nodes])
                    
                    delta = post_score - pre_score
                    deltas.append(delta)
        
        return deltas

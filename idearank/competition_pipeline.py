"""Competition pipeline for IdeaRank-Thought matches.

Manages the complete lifecycle of competition matches including:
- Match creation and management
- Real-time reasoning trace collection
- Coaching event processing
- Score calculation and validation
- Integrity protocols
"""

import logging
import uuid
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import asdict

from idearank.competition_models import (
    Match, Player, Coach, Challenge, ReasoningTrace, ReasoningNode,
    CoachingEvent, MatchStatus, CoachingType, FactorType
)
from idearank.competition_scorer import IdeaRankThoughtScorer
from idearank.providers.embeddings import EmbeddingProvider
from idearank.providers.chroma import ChromaProvider

logger = logging.getLogger(__name__)


class CompetitionPipeline:
    """Main pipeline for managing IdeaRank-Thought competitions."""
    
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        chroma_provider: ChromaProvider,
        storage_backend: Optional[Any] = None
    ):
        """Initialize the competition pipeline.
        
        Args:
            embedding_provider: For semantic similarity calculations
            chroma_provider: For reasoning trace storage and retrieval
            storage_backend: Optional storage backend for persistence
        """
        self.embedding_provider = embedding_provider
        self.chroma_provider = chroma_provider
        self.storage_backend = storage_backend
        
        # Initialize scorer
        self.scorer = IdeaRankThoughtScorer(embedding_provider)
        
        # Active matches
        self.active_matches: Dict[str, Match] = {}
        
        # Players and coaches registry
        self.players: Dict[str, Player] = {}
        self.coaches: Dict[str, Coach] = {}
        self.challenges: Dict[str, Challenge] = {}
    
    def create_match(
        self,
        challenge_id: str,
        player1_id: str,
        player2_id: Optional[str] = None,
        coach1_id: Optional[str] = None,
        coach2_id: Optional[str] = None,
        time_limit_minutes: Optional[int] = None
    ) -> Match:
        """Create a new competition match.
        
        Args:
            challenge_id: ID of the challenge to solve
            player1_id: Primary player ID
            player2_id: Secondary player ID (optional for solo matches)
            coach1_id: Coach for player 1 (optional)
            coach2_id: Coach for player 2 (optional)
            time_limit_minutes: Match time limit (optional)
            
        Returns:
            Created Match object
        """
        if challenge_id not in self.challenges:
            raise ValueError(f"Challenge {challenge_id} not found")
        
        challenge = self.challenges[challenge_id]
        
        # Create match
        match_id = str(uuid.uuid4())
        match = Match(
            id=match_id,
            challenge_id=challenge_id,
            player1_id=player1_id,
            player2_id=player2_id,
            coach1_id=coach1_id,
            coach2_id=coach2_id,
            status=MatchStatus.PENDING
        )
        
        # Initialize reasoning traces
        match.player1_trace = ReasoningTrace(
            match_id=match_id,
            player_id=player1_id
        )
        
        if player2_id:
            match.player2_trace = ReasoningTrace(
                match_id=match_id,
                player_id=player2_id
            )
        
        # Store in active matches
        self.active_matches[match_id] = match
        
        logger.info(f"Created match {match_id} for challenge {challenge_id}")
        return match
    
    def start_match(self, match_id: str) -> bool:
        """Start a match.
        
        Args:
            match_id: ID of the match to start
            
        Returns:
            True if match started successfully
        """
        if match_id not in self.active_matches:
            logger.error(f"Match {match_id} not found")
            return False
        
        match = self.active_matches[match_id]
        
        if match.status != MatchStatus.PENDING:
            logger.error(f"Match {match_id} is not in PENDING status")
            return False
        
        # Start the match
        match.status = MatchStatus.ACTIVE
        match.started_at = datetime.now()
        
        # Initialize reasoning traces
        if match.player1_trace:
            match.player1_trace.start_time = datetime.now()
        if match.player2_trace:
            match.player2_trace.start_time = datetime.now()
        
        logger.info(f"Started match {match_id}")
        return True
    
    def add_reasoning_node(
        self,
        match_id: str,
        player_id: str,
        content: str,
        confidence: float = 1.0,
        parent_ids: Optional[List[str]] = None,
        factor_contributions: Optional[Dict[FactorType, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[ReasoningNode]:
        """Add a reasoning node to a player's trace.
        
        Args:
            match_id: ID of the match
            player_id: ID of the player
            content: Reasoning content
            confidence: Confidence level (0-1)
            parent_ids: IDs of parent reasoning nodes
            factor_contributions: Factor contribution weights
            metadata: Additional metadata
            
        Returns:
            Created ReasoningNode or None if failed
        """
        if match_id not in self.active_matches:
            logger.error(f"Match {match_id} not found")
            return None
        
        match = self.active_matches[match_id]
        
        if match.status not in [MatchStatus.ACTIVE, MatchStatus.PAUSED]:
            logger.error(f"Cannot add reasoning to match {match_id} in status {match.status}")
            return None
        
        # Find the appropriate trace
        trace = None
        if match.player1_id == player_id:
            trace = match.player1_trace
        elif match.player2_id == player_id:
            trace = match.player2_trace
        
        if not trace:
            logger.error(f"Player {player_id} not found in match {match_id}")
            return None
        
        # Create reasoning node
        node_id = str(uuid.uuid4())
        node = ReasoningNode(
            id=node_id,
            content=content,
            timestamp=datetime.now(),
            confidence=confidence,
            parent_ids=parent_ids or [],
            factor_contributions=factor_contributions or {},
            metadata=metadata or {}
        )
        
        # Add to trace
        trace.add_node(node)
        
        logger.debug(f"Added reasoning node {node_id} for player {player_id} in match {match_id}")
        return node
    
    def add_coaching_event(
        self,
        match_id: str,
        player_id: str,
        coach_id: str,
        coaching_type: CoachingType,
        content: str,
        prebrief_plan: Optional[str] = None,
        duration_seconds: int = 0
    ) -> Optional[CoachingEvent]:
        """Add a coaching event to a match.
        
        Args:
            match_id: ID of the match
            player_id: ID of the player being coached
            coach_id: ID of the coach
            coaching_type: Type of coaching intervention
            content: Coaching content
            prebrief_plan: Optional prebrief plan
            duration_seconds: Coaching duration in seconds
            
        Returns:
            Created CoachingEvent or None if failed
        """
        if match_id not in self.active_matches:
            logger.error(f"Match {match_id} not found")
            return None
        
        match = self.active_matches[match_id]
        
        # Validate coach is assigned to this match
        if (match.coach1_id != coach_id and match.coach2_id != coach_id):
            logger.error(f"Coach {coach_id} not assigned to match {match_id}")
            return None
        
        # Validate player is in this match
        if (match.player1_id != player_id and match.player2_id != player_id):
            logger.error(f"Player {player_id} not in match {match_id}")
            return None
        
        # Create coaching event
        event_id = str(uuid.uuid4())
        event = CoachingEvent(
            id=event_id,
            match_id=match_id,
            player_id=player_id,
            coach_id=coach_id,
            timestamp=datetime.now(),
            coaching_type=coaching_type,
            content=content,
            prebrief_plan=prebrief_plan,
            duration_seconds=duration_seconds
        )
        
        # Add to match
        match.add_coaching_event(event)
        
        # Pause match during coaching timeout
        if coaching_type == CoachingType.TIMEOUT:
            match.status = MatchStatus.PAUSED
        
        logger.info(f"Added coaching event {event_id} for player {player_id} in match {match_id}")
        return event
    
    def resume_match_after_coaching(self, match_id: str) -> bool:
        """Resume match after coaching timeout.
        
        Args:
            match_id: ID of the match to resume
            
        Returns:
            True if match resumed successfully
        """
        if match_id not in self.active_matches:
            logger.error(f"Match {match_id} not found")
            return False
        
        match = self.active_matches[match_id]
        
        if match.status != MatchStatus.PAUSED:
            logger.error(f"Match {match_id} is not in PAUSED status")
            return False
        
        match.status = MatchStatus.ACTIVE
        logger.info(f"Resumed match {match_id} after coaching")
        return True
    
    def complete_match(self, match_id: str) -> Optional[Dict[str, Any]]:
        """Complete a match and calculate scores.
        
        Args:
            match_id: ID of the match to complete
            
        Returns:
            Dict with match results and scores
        """
        if match_id not in self.active_matches:
            logger.error(f"Match {match_id} not found")
            return None
        
        match = self.active_matches[match_id]
        
        if match.status not in [MatchStatus.ACTIVE, MatchStatus.PAUSED]:
            logger.error(f"Cannot complete match {match_id} in status {match.status}")
            return None
        
        # Get challenge
        challenge = self.challenges.get(match.challenge_id)
        if not challenge:
            logger.error(f"Challenge {match.challenge_id} not found")
            return None
        
        # Finalize reasoning traces
        if match.player1_trace:
            match.player1_trace.end_time = datetime.now()
        if match.player2_trace:
            match.player2_trace.end_time = datetime.now()
        
        # Calculate scores
        scores = self.scorer.score_match(match, challenge)
        
        # Store scores in match
        match.player1_scores = scores.get(match.player1_id, {}).to_dict() if match.player1_id in scores else {}
        if match.player2_id and match.player2_id in scores:
            match.player2_scores = scores[match.player2_id].to_dict()
        
        # Finalize match
        match.finalize()
        
        # Update player statistics
        self._update_player_stats(match, scores)
        
        # Store match results
        if self.storage_backend:
            self._store_match_results(match, scores)
        
        # Remove from active matches
        del self.active_matches[match_id]
        
        logger.info(f"Completed match {match_id}")
        
        return {
            'match': match,
            'scores': scores,
            'challenge': challenge
        }
    
    def _update_player_stats(self, match: Match, scores: Dict[str, Any]) -> None:
        """Update player statistics after match completion."""
        for player_id, score in scores.items():
            if player_id in self.players:
                player = self.players[player_id]
                
                # Determine if player won (simplified logic)
                won = False
                if match.player2_id:
                    # Two-player match - compare scores
                    player1_score = score.raw_score
                    player2_score = scores.get(match.player2_id, score).raw_score
                    won = player1_score > player2_score
                else:
                    # Solo match - check if score passes gates
                    won = score.passes_gates
                
                # Update player stats
                player.update_stats(
                    ir_t_score=score.raw_score,
                    won=won,
                    coach_impact=score.coach_impact_index
                )
                
                logger.debug(f"Updated stats for player {player_id}")
    
    def _store_match_results(self, match: Match, scores: Dict[str, Any]) -> None:
        """Store match results in persistent storage."""
        if not self.storage_backend:
            return
        
        try:
            # Store match data
            match_data = {
                'match': asdict(match),
                'scores': {k: v.to_dict() for k, v in scores.items()},
                'stored_at': datetime.now().isoformat()
            }
            
            # Store in backend (implementation depends on backend type)
            # self.storage_backend.store_match_results(match_data)
            
            logger.debug(f"Stored results for match {match.id}")
        except Exception as e:
            logger.error(f"Failed to store match results: {e}")
    
    def get_match_status(self, match_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a match.
        
        Args:
            match_id: ID of the match
            
        Returns:
            Dict with match status information
        """
        if match_id not in self.active_matches:
            return None
        
        match = self.active_matches[match_id]
        
        return {
            'match_id': match.id,
            'status': match.status.value,
            'challenge_id': match.challenge_id,
            'player1_id': match.player1_id,
            'player2_id': match.player2_id,
            'started_at': match.started_at.isoformat() if match.started_at else None,
            'duration': (
                (datetime.now() - match.started_at).total_seconds() 
                if match.started_at else 0
            ),
            'reasoning_nodes': {
                'player1': len(match.player1_trace.nodes) if match.player1_trace else 0,
                'player2': len(match.player2_trace.nodes) if match.player2_trace else 0,
            },
            'coaching_events': len(match.coaching_events)
        }
    
    def register_player(self, player: Player) -> None:
        """Register a new player."""
        self.players[player.id] = player
        logger.info(f"Registered player {player.id}")
    
    def register_coach(self, coach: Coach) -> None:
        """Register a new coach."""
        self.coaches[coach.id] = coach
        logger.info(f"Registered coach {coach.id}")
    
    def register_challenge(self, challenge: Challenge) -> None:
        """Register a new challenge."""
        self.challenges[challenge.id] = challenge
        logger.info(f"Registered challenge {challenge.id}")
    
    def get_leaderboard(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get current leaderboard.
        
        Args:
            limit: Number of players to return
            
        Returns:
            List of player stats sorted by meta rating
        """
        sorted_players = sorted(
            self.players.values(),
            key=lambda p: p.meta_rating,
            reverse=True
        )
        
        leaderboard = []
        for player in sorted_players[:limit]:
            leaderboard.append({
                'player_id': player.id,
                'name': player.name,
                'meta_rating': player.meta_rating,
                'total_matches': player.total_matches,
                'win_rate': player.win_rate,
                'average_ir_t_score': player.average_ir_t_score,
                'coaching_impact_index': player.coaching_impact_index
            })
        
        return leaderboard
    
    def validate_match_integrity(self, match_id: str) -> bool:
        """Validate match integrity using ledger hash.
        
        Args:
            match_id: ID of the match to validate
            
        Returns:
            True if match integrity is valid
        """
        if match_id not in self.active_matches:
            logger.error(f"Match {match_id} not found")
            return False
        
        match = self.active_matches[match_id]
        return match.validate_integrity()
    
    def get_reasoning_trace(self, match_id: str, player_id: str) -> Optional[ReasoningTrace]:
        """Get reasoning trace for a player in a match.
        
        Args:
            match_id: ID of the match
            player_id: ID of the player
            
        Returns:
            ReasoningTrace or None if not found
        """
        if match_id not in self.active_matches:
            return None
        
        match = self.active_matches[match_id]
        
        if match.player1_id == player_id:
            return match.player1_trace
        elif match.player2_id == player_id:
            return match.player2_trace
        
        return None
    
    def get_coaching_events(self, match_id: str, player_id: Optional[str] = None) -> List[CoachingEvent]:
        """Get coaching events for a match.
        
        Args:
            match_id: ID of the match
            player_id: Optional player ID to filter events
            
        Returns:
            List of CoachingEvent objects
        """
        if match_id not in self.active_matches:
            return []
        
        match = self.active_matches[match_id]
        
        if player_id:
            return match.get_coaching_events_for_player(player_id)
        else:
            return match.coaching_events
    
    def export_match_data(self, match_id: str) -> Optional[Dict[str, Any]]:
        """Export complete match data for analysis.
        
        Args:
            match_id: ID of the match to export
            
        Returns:
            Dict with complete match data
        """
        if match_id not in self.active_matches:
            return None
        
        match = self.active_matches[match_id]
        
        return {
            'match': asdict(match),
            'player1_trace': match.player1_trace.to_dict() if match.player1_trace else None,
            'player2_trace': match.player2_trace.to_dict() if match.player2_trace else None,
            'coaching_events': [event.to_dict() for event in match.coaching_events],
            'challenge': asdict(self.challenges.get(match.challenge_id)),
            'exported_at': datetime.now().isoformat()
        }

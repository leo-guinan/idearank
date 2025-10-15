"""Models for IdeaRank-Thought competition system.

This module defines the core data structures for the MetaSPN competition,
including matches, players, coaching, reasoning traces, and scoring.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import json
import hashlib

from idearank.models import ContentItem, Embedding


class MatchStatus(Enum):
    """Status of a competition match."""
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"  # During coaching timeout
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class CoachingType(Enum):
    """Type of coaching intervention."""
    TIMEOUT = "timeout"
    STRATEGY = "strategy"
    CORRECTION = "correction"
    ENCOURAGEMENT = "encouragement"
    TACTICAL = "tactical"


class FactorType(Enum):
    """IdeaRank-Thought factors."""
    UNIQUENESS = "uniqueness"  # U
    COHESION = "cohesion"      # C
    LEARNING = "learning"      # L
    QUALITY = "quality"        # Q
    TRUST = "trust"            # T
    DENSITY = "density"        # D


@dataclass
class ReasoningNode:
    """A single node in the reasoning trace."""
    
    id: str
    content: str
    timestamp: datetime
    factor_contributions: Dict[FactorType, float] = field(default_factory=dict)
    confidence: float = 1.0
    parent_ids: List[str] = field(default_factory=list)
    child_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'factor_contributions': {k.value: v for k, v in self.factor_contributions.items()},
            'confidence': self.confidence,
            'parent_ids': self.parent_ids,
            'child_ids': self.child_ids,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReasoningNode':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            factor_contributions={
                FactorType(k): v for k, v in data.get('factor_contributions', {}).items()
            },
            confidence=data.get('confidence', 1.0),
            parent_ids=data.get('parent_ids', []),
            child_ids=data.get('child_ids', []),
            metadata=data.get('metadata', {}),
        )


@dataclass
class ReasoningTrace:
    """Complete reasoning trace for a match."""
    
    match_id: str
    player_id: str
    nodes: List[ReasoningNode] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def add_node(self, node: ReasoningNode) -> None:
        """Add a reasoning node to the trace."""
        self.nodes.append(node)
        
        # Update parent-child relationships
        for parent_id in node.parent_ids:
            for existing_node in self.nodes:
                if existing_node.id == parent_id:
                    if node.id not in existing_node.child_ids:
                        existing_node.child_ids.append(node.id)
    
    def get_node(self, node_id: str) -> Optional[ReasoningNode]:
        """Get a node by ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def calculate_entropy_index(self) -> float:
        """Calculate entropy index for tiebreaks.
        
        EI = -∑(pi * log2(pi))
        where pi represents the probability distribution over reasoning branches.
        """
        if not self.nodes:
            return 0.0
        
        # Calculate branching probabilities
        branch_counts = {}
        total_branches = 0
        
        for node in self.nodes:
            branch_count = len(node.child_ids)
            branch_counts[node.id] = branch_count
            total_branches += branch_count
        
        if total_branches == 0:
            return 0.0
        
        # Calculate entropy
        entropy = 0.0
        for count in branch_counts.values():
            if count > 0:
                p = count / total_branches
                entropy -= p * (p.bit_length() - 1) if p > 0 else 0  # log2 approximation
        
        return entropy
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'match_id': self.match_id,
            'player_id': self.player_id,
            'nodes': [node.to_dict() for node in self.nodes],
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReasoningTrace':
        """Create from dictionary."""
        trace = cls(
            match_id=data['match_id'],
            player_id=data['player_id'],
            start_time=datetime.fromisoformat(data['start_time']) if data.get('start_time') else None,
            end_time=datetime.fromisoformat(data['end_time']) if data.get('end_time') else None,
        )
        
        for node_data in data.get('nodes', []):
            trace.add_node(ReasoningNode.from_dict(node_data))
        
        return trace


@dataclass
class CoachingEvent:
    """A coaching intervention during a match."""
    
    id: str
    match_id: str
    player_id: str
    coach_id: str
    timestamp: datetime
    coaching_type: CoachingType
    content: str
    prebrief_plan: Optional[str] = None
    duration_seconds: int = 0
    effectiveness_score: Optional[float] = None  # Calculated post-match
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'match_id': self.match_id,
            'player_id': self.player_id,
            'coach_id': self.coach_id,
            'timestamp': self.timestamp.isoformat(),
            'coaching_type': self.coaching_type.value,
            'content': self.content,
            'prebrief_plan': self.prebrief_plan,
            'duration_seconds': self.duration_seconds,
            'effectiveness_score': self.effectiveness_score,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CoachingEvent':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            match_id=data['match_id'],
            player_id=data['player_id'],
            coach_id=data['coach_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            coaching_type=CoachingType(data['coaching_type']),
            content=data['content'],
            prebrief_plan=data.get('prebrief_plan'),
            duration_seconds=data.get('duration_seconds', 0),
            effectiveness_score=data.get('effectiveness_score'),
        )


@dataclass
class Player:
    """A competition player."""
    
    id: str
    name: str
    team_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    meta_rating: float = 0.0
    total_matches: int = 0
    wins: int = 0
    losses: int = 0
    average_ir_t_score: float = 0.0
    coaching_impact_index: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.total_matches == 0:
            return 0.0
        return self.wins / self.total_matches
    
    def update_stats(self, ir_t_score: float, won: bool, coach_impact: float) -> None:
        """Update player statistics after a match."""
        self.total_matches += 1
        if won:
            self.wins += 1
        else:
            self.losses += 1
        
        # Update running averages
        self.average_ir_t_score = (
            (self.average_ir_t_score * (self.total_matches - 1) + ir_t_score) / self.total_matches
        )
        self.coaching_impact_index = (
            (self.coaching_impact_index * (self.total_matches - 1) + coach_impact) / self.total_matches
        )
        
        # Update meta rating (simplified formula)
        self.meta_rating = (
            0.4 * self.win_rate + 
            0.3 * self.average_ir_t_score + 
            0.3 * max(0, self.coaching_impact_index)
        )


@dataclass
class Coach:
    """A competition coach."""
    
    id: str
    name: str
    team_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    total_coaching_events: int = 0
    average_effectiveness: float = 0.0
    specialization: Optional[CoachingType] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Match:
    """A competition match between players."""
    
    id: str
    challenge_id: str
    player1_id: str
    player2_id: Optional[str] = None  # None for solo challenges
    coach1_id: Optional[str] = None
    coach2_id: Optional[str] = None
    status: MatchStatus = MatchStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Reasoning traces
    player1_trace: Optional[ReasoningTrace] = None
    player2_trace: Optional[ReasoningTrace] = None
    
    # Coaching events
    coaching_events: List[CoachingEvent] = field(default_factory=list)
    
    # Scoring results
    player1_scores: Dict[str, float] = field(default_factory=dict)
    player2_scores: Dict[str, float] = field(default_factory=dict)
    
    # Integrity
    ledger_hash: Optional[str] = None
    validation_hash: Optional[str] = None
    
    def add_coaching_event(self, event: CoachingEvent) -> None:
        """Add a coaching event to the match."""
        self.coaching_events.append(event)
    
    def get_coaching_events_for_player(self, player_id: str) -> List[CoachingEvent]:
        """Get all coaching events for a specific player."""
        return [event for event in self.coaching_events if event.player_id == player_id]
    
    def calculate_ledger_hash(self) -> str:
        """Calculate SHA-256 hash of match data for integrity."""
        # Create a deterministic representation of the match
        match_data = {
            'id': self.id,
            'challenge_id': self.challenge_id,
            'player1_id': self.player1_id,
            'player2_id': self.player2_id,
            'coaching_events': [event.to_dict() for event in self.coaching_events],
            'player1_trace': self.player1_trace.to_dict() if self.player1_trace else None,
            'player2_trace': self.player2_trace.to_dict() if self.player2_trace else None,
        }
        
        # Sort keys for deterministic hashing
        match_json = json.dumps(match_data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(match_json.encode('utf-8')).hexdigest()
    
    def validate_integrity(self) -> bool:
        """Validate match integrity using ledger hash."""
        if self.ledger_hash is None:
            return False
        
        calculated_hash = self.calculate_ledger_hash()
        return calculated_hash == self.ledger_hash
    
    def finalize(self) -> None:
        """Finalize the match and calculate ledger hash."""
        self.status = MatchStatus.COMPLETED
        self.completed_at = datetime.now()
        self.ledger_hash = self.calculate_ledger_hash()


@dataclass
class Challenge:
    """A competition challenge/problem."""
    
    id: str
    title: str
    description: str
    difficulty_level: int  # 1-10 scale
    time_limit_minutes: int
    constraints: List[str] = field(default_factory=list)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Ground truth for evaluation
    ground_truth: Optional[Dict[str, Any]] = None
    evaluation_rubric: Dict[str, float] = field(default_factory=dict)


@dataclass
class OutcomeValidity:
    """Outcome Validity (O) gate calculation."""
    
    correctness: float  # 0-1 scale
    robustness: float   # 0-1 scale
    
    @property
    def score(self) -> float:
        """Calculate Outcome Validity score.
        
        O = 0.7 × correctness + 0.3 × robustness
        """
        return 0.7 * self.correctness + 0.3 * self.robustness


@dataclass
class ConstraintCompliance:
    """Constraint Compliance (X) gate calculation."""
    
    violations: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def score(self) -> float:
        """Calculate Constraint Compliance score.
        
        X = 1 - violation_penalty
        """
        if not self.violations:
            return 1.0
        
        # Calculate penalty based on severity and frequency
        total_penalty = 0.0
        for violation in self.violations:
            severity = violation.get('severity', 0.1)  # 0.1 to 1.0
            frequency = violation.get('frequency', 1)
            penalty = severity * frequency * 0.1  # Scale factor
            total_penalty += penalty
        
        return max(0.0, 1.0 - total_penalty)


@dataclass
class IdeaRankThoughtScore:
    """Complete IdeaRank-Thought score for a match."""
    
    match_id: str
    player_id: str
    
    # Gate scores
    outcome_validity: OutcomeValidity
    constraint_compliance: ConstraintCompliance
    
    # Factor scores (U, C, L, Q, T, D)
    factor_scores: Dict[FactorType, float]
    
    # Raw IR-T score
    raw_score: float
    
    # Adjusted scores
    class_adjusted_score: float
    coach_impact_index: float  # -1.0 to +1.0
    
    # Meta metrics
    meta_rating: float
    entropy_index: float
    
    # Coaching analysis
    plan_adherence: float  # 0-1 scale
    timeout_deltas: List[float] = field(default_factory=list)
    
    # Timestamps
    calculated_at: datetime = field(default_factory=datetime.now)
    
    @property
    def passes_gates(self) -> bool:
        """Check if the score passes both gates."""
        return (
            self.outcome_validity.score >= 0.5 and 
            self.constraint_compliance.score >= 0.5
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'match_id': self.match_id,
            'player_id': self.player_id,
            'outcome_validity': {
                'correctness': self.outcome_validity.correctness,
                'robustness': self.outcome_validity.robustness,
                'score': self.outcome_validity.score,
            },
            'constraint_compliance': {
                'violations': self.constraint_compliance.violations,
                'score': self.constraint_compliance.score,
            },
            'factor_scores': {k.value: v for k, v in self.factor_scores.items()},
            'raw_score': self.raw_score,
            'class_adjusted_score': self.class_adjusted_score,
            'coach_impact_index': self.coach_impact_index,
            'meta_rating': self.meta_rating,
            'entropy_index': self.entropy_index,
            'plan_adherence': self.plan_adherence,
            'timeout_deltas': self.timeout_deltas,
            'passes_gates': self.passes_gates,
            'calculated_at': self.calculated_at.isoformat(),
        }

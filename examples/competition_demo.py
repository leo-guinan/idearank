"""Demo script for IdeaRank-Thought competition system.

This script demonstrates the complete competition workflow:
1. Setting up players, coaches, and challenges
2. Creating and running a match
3. Real-time reasoning trace collection
4. Coaching interventions
5. Score calculation and analysis
6. Visualization of results
"""

import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from idearank.competition_models import (
    Player, Coach, Challenge, Match, MatchStatus, CoachingType, FactorType,
    ReasoningNode, CoachingEvent
)
from idearank.competition_pipeline import CompetitionPipeline
from idearank.competition_scorer import IdeaRankThoughtScorer
from idearank.competition_visualizer import CompetitionVisualizer
from idearank.providers.embeddings import SentenceTransformerEmbeddingProvider
from idearank.providers.chroma import ChromaProvider

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_demo_players() -> Dict[str, Player]:
    """Create demo players for the competition."""
    
    players = {}
    
    # Player 1: Analytical Thinker
    players['alice'] = Player(
        id='alice',
        name='Alice Chen',
        team_id='analytics_team',
        meta_rating=0.75,
        total_matches=5,
        wins=3,
        losses=2,
        average_ir_t_score=0.68,
        coaching_impact_index=0.3
    )
    
    # Player 2: Creative Problem Solver
    players['bob'] = Player(
        id='bob',
        name='Bob Rodriguez',
        team_id='creative_team',
        meta_rating=0.82,
        total_matches=7,
        wins=5,
        losses=2,
        average_ir_t_score=0.74,
        coaching_impact_index=0.45
    )
    
    return players


def create_demo_coaches() -> Dict[str, Coach]:
    """Create demo coaches for the competition."""
    
    coaches = {}
    
    # Coach 1: Strategy Specialist
    coaches['coach_strategy'] = Coach(
        id='coach_strategy',
        name='Dr. Sarah Johnson',
        team_id='analytics_team',
        total_coaching_events=25,
        average_effectiveness=0.72,
        specialization=CoachingType.STRATEGY
    )
    
    # Coach 2: Tactical Expert
    coaches['coach_tactical'] = Coach(
        id='coach_tactical',
        name='Mike Thompson',
        team_id='creative_team',
        total_coaching_events=18,
        average_effectiveness=0.68,
        specialization=CoachingType.TACTICAL
    )
    
    return coaches


def create_demo_challenges() -> Dict[str, Challenge]:
    """Create demo challenges for the competition."""
    
    challenges = {}
    
    # Challenge 1: Logic Puzzle
    challenges['logic_puzzle'] = Challenge(
        id='logic_puzzle',
        title='The Missing Number Sequence',
        description="""
        Given the sequence: 2, 6, 12, 20, 30, ?
        
        Find the next number in the sequence and explain your reasoning.
        Consider multiple approaches and justify your final answer.
        """,
        difficulty_level=6,
        time_limit_minutes=15,
        constraints=[
            'Must show step-by-step reasoning',
            'Cannot use external tools or calculators',
            'Must consider alternative approaches'
        ],
        success_criteria={
            'correctness': 0.8,
            'explanation_quality': 0.7,
            'alternative_approaches': 0.6
        },
        ground_truth={
            'answer': 42,
            'pattern': 'n(n+1)',
            'reasoning': 'Each number is the product of two consecutive integers'
        },
        evaluation_rubric={
            'correctness': 0.4,
            'reasoning_quality': 0.3,
            'explanation_clarity': 0.3
        }
    )
    
    # Challenge 2: Problem-Solving Scenario
    challenges['resource_optimization'] = Challenge(
        id='resource_optimization',
        title='Emergency Resource Allocation',
        description="""
        A natural disaster has struck a city with 100,000 residents.
        You have limited resources:
        - 50 medical teams
        - 200 food distribution points
        - 1000 emergency shelters
        
        Design an allocation strategy that maximizes lives saved.
        Consider constraints like geography, demographics, and resource logistics.
        """,
        difficulty_level=8,
        time_limit_minutes=25,
        constraints=[
            'Must consider ethical implications',
            'Must address resource scarcity',
            'Must provide implementation timeline',
            'Cannot exceed available resources'
        ],
        success_criteria={
            'lives_saved_estimate': 0.8,
            'feasibility': 0.7,
            'ethical_considerations': 0.6
        },
        evaluation_rubric={
            'strategic_thinking': 0.3,
            'practical_feasibility': 0.3,
            'ethical_reasoning': 0.2,
            'communication': 0.2
        }
    )
    
    return challenges


def simulate_reasoning_trace(pipeline: CompetitionPipeline, match_id: str, player_id: str, challenge: Challenge) -> None:
    """Simulate a reasoning trace for a player."""
    
    logger.info(f"Simulating reasoning trace for player {player_id}")
    
    # Simulate reasoning steps based on challenge type
    if challenge.id == 'logic_puzzle':
        reasoning_steps = [
            ("Looking at the sequence: 2, 6, 12, 20, 30", 0.7, {FactorType.UNIQUENESS: 0.3, FactorType.QUALITY: 0.4}),
            ("Let me check the differences between consecutive numbers: 4, 6, 8, 10", 0.8, {FactorType.COHESION: 0.5, FactorType.LEARNING: 0.3}),
            ("The differences are increasing by 2 each time: 4, 6, 8, 10, so next difference should be 12", 0.9, {FactorType.COHESION: 0.7, FactorType.QUALITY: 0.6}),
            ("Therefore, the next number should be 30 + 12 = 42", 0.95, {FactorType.UNIQUENESS: 0.6, FactorType.TRUST: 0.7}),
            ("Let me verify this pattern: 2=1×2, 6=2×3, 12=3×4, 20=4×5, 30=5×6, so 42=6×7", 0.98, {FactorType.DENSITY: 0.8, FactorType.TRUST: 0.9})
        ]
    else:  # resource_optimization
        reasoning_steps = [
            ("First, I need to assess the scale of the disaster and population distribution", 0.6, {FactorType.UNIQUENESS: 0.4, FactorType.QUALITY: 0.3}),
            ("With 100,000 residents, I need to prioritize based on vulnerability and location", 0.7, {FactorType.COHESION: 0.5, FactorType.LEARNING: 0.4}),
            ("Medical teams should focus on high-density areas and vulnerable populations first", 0.8, {FactorType.QUALITY: 0.6, FactorType.TRUST: 0.5}),
            ("Food distribution needs to be accessible and evenly distributed across the city", 0.75, {FactorType.DENSITY: 0.7, FactorType.COHESION: 0.6}),
            ("Emergency shelters should be strategically placed to maximize coverage", 0.85, {FactorType.UNIQUENESS: 0.7, FactorType.TRUST: 0.8})
        ]
    
    # Add reasoning nodes
    for i, (content, confidence, factor_contributions) in enumerate(reasoning_steps):
        parent_ids = [f"node_{i-1}"] if i > 0 else []
        
        node = pipeline.add_reasoning_node(
            match_id=match_id,
            player_id=player_id,
            content=content,
            confidence=confidence,
            parent_ids=parent_ids,
            factor_contributions=factor_contributions,
            metadata={'step': i + 1, 'challenge_type': challenge.id}
        )
        
        # Add small delay to simulate real-time reasoning
        import time
        time.sleep(0.1)


def simulate_coaching_interventions(pipeline: CompetitionPipeline, match_id: str, player_id: str, coach_id: str) -> None:
    """Simulate coaching interventions during the match."""
    
    logger.info(f"Simulating coaching interventions for player {player_id}")
    
    # Add strategic coaching
    pipeline.add_coaching_event(
        match_id=match_id,
        player_id=player_id,
        coach_id=coach_id,
        coaching_type=CoachingType.STRATEGY,
        content="Remember to consider multiple approaches and validate your reasoning",
        prebrief_plan="Use systematic analysis: identify patterns, test hypotheses, verify results",
        duration_seconds=120
    )
    
    # Add tactical coaching
    pipeline.add_coaching_event(
        match_id=match_id,
        player_id=player_id,
        coach_id=coach_id,
        coaching_type=CoachingType.TACTICAL,
        content="Good progress! Now focus on the implementation details and edge cases",
        duration_seconds=90
    )


def run_competition_demo():
    """Run the complete competition demo."""
    
    logger.info("Starting IdeaRank-Thought Competition Demo")
    
    # Initialize providers
    embedding_provider = SentenceTransformerEmbeddingProvider(model_name="all-MiniLM-L6-v2")
    chroma_provider = ChromaProvider(
        collection_name="competition_reasoning",
        persist_directory="./chroma_competition"
    )
    
    # Initialize pipeline
    pipeline = CompetitionPipeline(
        embedding_provider=embedding_provider,
        chroma_provider=chroma_provider
    )
    
    # Create demo entities
    players = create_demo_players()
    coaches = create_demo_coaches()
    challenges = create_demo_challenges()
    
    # Register entities
    for player in players.values():
        pipeline.register_player(player)
    
    for coach in coaches.values():
        pipeline.register_coach(coach)
    
    for challenge in challenges.values():
        pipeline.register_challenge(challenge)
    
    # Create and run a match
    challenge = challenges['logic_puzzle']
    match = pipeline.create_match(
        challenge_id=challenge.id,
        player1_id='alice',
        player2_id='bob',
        coach1_id='coach_strategy',
        coach2_id='coach_tactical'
    )
    
    logger.info(f"Created match {match.id}")
    
    # Start the match
    pipeline.start_match(match.id)
    logger.info("Match started")
    
    # Simulate reasoning for both players
    simulate_reasoning_trace(pipeline, match.id, 'alice', challenge)
    simulate_reasoning_trace(pipeline, match.id, 'bob', challenge)
    
    # Add coaching interventions
    simulate_coaching_interventions(pipeline, match.id, 'alice', 'coach_strategy')
    simulate_coaching_interventions(pipeline, match.id, 'bob', 'coach_tactical')
    
    # Complete the match
    results = pipeline.complete_match(match.id)
    
    if results:
        logger.info("Match completed successfully!")
        
        # Display results
        print("\n" + "="*60)
        print("COMPETITION RESULTS")
        print("="*60)
        
        for player_id, score in results['scores'].items():
            player_name = players[player_id].name
            print(f"\n{player_name} ({player_id}):")
            print(f"  Raw IR-T Score: {score.raw_score:.3f}")
            print(f"  Class Adjusted: {score.class_adjusted_score:.3f}")
            print(f"  Coach Impact: {score.coach_impact_index:.3f}")
            print(f"  Passes Gates: {'Yes' if score.passes_gates else 'No'}")
            print(f"  Entropy Index: {score.entropy_index:.3f}")
            
            print(f"  Factor Scores:")
            for factor, factor_score in score.factor_scores.items():
                print(f"    {factor.value.title()}: {factor_score:.3f}")
            
            print(f"  Gate Scores:")
            print(f"    Outcome Validity: {score.outcome_validity.score:.3f}")
            print(f"    Constraint Compliance: {score.constraint_compliance.score:.3f}")
        
        # Create visualizations
        visualizer = CompetitionVisualizer()
        
        # Create reason maps for both players
        for player_id in ['alice', 'bob']:
            trace = results['match'].player1_trace if player_id == 'alice' else results['match'].player2_trace
            if trace:
                fig = visualizer.create_reason_map(
                    trace=trace,
                    match=results['match'],
                    player_id=player_id,
                    title=f"Reason Map - {players[player_id].name}"
                )
                
                if fig:
                    # Save visualization
                    output_file = f"reason_map_{player_id}_{match.id[:8]}.html"
                    fig.write_html(output_file)
                    logger.info(f"Saved reason map to {output_file}")
        
        # Create coaching impact chart
        for player_id in ['alice', 'bob']:
            fig = visualizer.create_coaching_impact_chart(results['match'], player_id)
            if fig:
                output_file = f"coaching_impact_{player_id}_{match.id[:8]}.html"
                fig.write_html(output_file)
                logger.info(f"Saved coaching impact chart to {output_file}")
        
        # Export visualization data
        for player_id in ['alice', 'bob']:
            viz_data = visualizer.export_visualization_data(results['match'], player_id)
            output_file = f"viz_data_{player_id}_{match.id[:8]}.json"
            import json
            with open(output_file, 'w') as f:
                json.dump(viz_data, f, indent=2)
            logger.info(f"Exported visualization data to {output_file}")
        
        # Display leaderboard
        print(f"\n" + "="*60)
        print("CURRENT LEADERBOARD")
        print("="*60)
        
        leaderboard = pipeline.get_leaderboard()
        for i, player_data in enumerate(leaderboard, 1):
            print(f"{i:2d}. {player_data['name']:20s} "
                  f"MR: {player_data['meta_rating']:.3f} "
                  f"WR: {player_data['win_rate']:.3f} "
                  f"Avg: {player_data['average_ir_t_score']:.3f}")
        
        print(f"\nMatch {match.id} completed successfully!")
        print("Check the generated HTML files for visualizations.")
        
    else:
        logger.error("Failed to complete match")


if __name__ == "__main__":
    run_competition_demo()

"""
Interactive IdeaRank-Thought Competition Game

This module provides an interactive CLI game where users can play as either
a player solving reasoning challenges or a coach providing interventions.
"""

import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt, Confirm
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich.align import Align
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    from idearank.competition_models import (
        Player, Coach, Challenge, Match, ReasoningTrace, ReasoningNode,
        CoachingEvent, CoachingType, FactorType, IdeaRankThoughtScore
    )
    from idearank.competition_scorer import IdeaRankThoughtScorer
    from idearank.competition_pipeline import CompetitionPipeline
    from idearank.competition_visualizer import CompetitionVisualizer
    COMPETITION_AVAILABLE = True
except ImportError:
    COMPETITION_AVAILABLE = False


class GameMode(Enum):
    PLAYER = "player"
    COACH = "coach"


@dataclass
class GameState:
    """Current state of the interactive game."""
    mode: GameMode
    current_match: Optional[Match] = None
    player_trace: Optional[ReasoningTrace] = None
    ai_trace: Optional[ReasoningTrace] = None
    score: float = 0.0
    round_number: int = 1
    max_rounds: int = 3


class InteractiveGame:
    """Interactive CLI game for IdeaRank-Thought competition."""
    
    def __init__(self):
        if not RICH_AVAILABLE:
            raise ImportError("Rich library is required for the interactive game")
        if not COMPETITION_AVAILABLE:
            raise ImportError("Competition system is required for the interactive game")
        
        self.console = Console()
        
        # Initialize providers
        try:
            from idearank.providers.embeddings import SentenceTransformerEmbeddingProvider
            from idearank.providers.chroma import ChromaProvider
            
            self.embedding_provider = SentenceTransformerEmbeddingProvider()
            # Use local mode for interactive game
            self.chroma_provider = ChromaProvider(
                collection_name="interactive_game",
                persist_directory="./chroma_interactive_game"  # Use local persistence
            )
            
            self.scorer = IdeaRankThoughtScorer(self.embedding_provider)
            self.pipeline = CompetitionPipeline(self.embedding_provider, self.chroma_provider)
        except ImportError:
            # Fallback if providers not available
            self.scorer = None
            self.pipeline = None
            self.embedding_provider = None
            self.chroma_provider = None
        
        self.visualizer = CompetitionVisualizer()
        self.state = GameState(mode=GameMode.PLAYER)
        
        # Initialize game components
        self._setup_game()
    
    def _setup_game(self):
        """Initialize game components and data."""
        # Create human player
        self.human_player = Player(
            id="human_player",
            name="You"
        )
        
        # Create AI opponent
        self.ai_player = Player(
            id="ai_player", 
            name="AI Opponent"
        )
        
        # Create AI coach
        self.ai_coach = Coach(
            id="ai_coach",
            name="AI Coach"
        )
        
        # Create challenges
        self.challenges = [
            Challenge(
                id="logic_puzzle",
                title="Number Sequence Challenge",
                description="Find the next number in the sequence: 2, 6, 12, 20, 30, ?",
                difficulty_level=6,
                time_limit_minutes=5
            ),
            Challenge(
                id="resource_optimization",
                title="Resource Allocation Problem",
                description="Optimize resource allocation across 3 projects with constraints: Project A needs 40% resources, Project B needs 35%, Project C needs 25%. How do you balance efficiency and fairness?",
                difficulty_level=7,
                time_limit_minutes=7
            ),
            Challenge(
                id="pattern_recognition",
                title="Pattern Recognition Challenge", 
                description="Identify the pattern in: A1, B4, C9, D16, E25, ? What comes next and why?",
                difficulty_level=5,
                time_limit_minutes=4
            )
        ]
        
        # Register players, coaches, and challenges with the pipeline
        if self.pipeline:
            self.pipeline.register_player(self.human_player)
            self.pipeline.register_player(self.ai_player)
            self.pipeline.register_coach(self.ai_coach)
            for challenge in self.challenges:
                self.pipeline.register_challenge(challenge)
    
    def run(self):
        """Main game loop."""
        self._show_welcome()
        
        # Select game mode
        mode_choice = Prompt.ask(
            "Choose your role",
            choices=["player", "coach"],
            default="player"
        )
        
        self.state.mode = GameMode.PLAYER if mode_choice == "player" else GameMode.COACH
        
        if self.state.mode == GameMode.PLAYER:
            self._play_as_player()
        else:
            self._play_as_coach()
        
        self._show_final_results()
    
    def _show_welcome(self):
        """Display welcome screen."""
        welcome_text = """
🎯 Welcome to IdeaRank-Thought Interactive Competition! 🎯

Test your reasoning skills in real-time against AI opponents.
Experience coaching interventions and see their impact on performance.

Choose your role:
• Player: Solve reasoning challenges with AI coaching support
• Coach: Provide interventions to help an AI player improve

Let's begin your journey into competitive reasoning!
        """
        
        panel = Panel(
            Align.center(welcome_text),
            title="[bold blue]IdeaRank-Thought Competition[/bold blue]",
            border_style="blue"
        )
        
        self.console.print(panel)
        self.console.print()
    
    def _play_as_player(self):
        """Play as a reasoning player."""
        self.console.print("[bold green]🎮 Playing as Player[/bold green]")
        self.console.print("You'll solve reasoning challenges with AI coaching support.\n")
        
        all_scores = []
        
        for round_num in range(1, self.state.max_rounds + 1):
            self.state.round_number = round_num
            challenge = random.choice(self.challenges)
            
            self.console.print(f"[bold yellow]Round {round_num}/{self.state.max_rounds}[/bold yellow]")
            self._play_challenge_as_player(challenge)
            
            # Calculate and store round score
            if self.state.player_trace and self.state.player_trace.nodes:
                round_score = self._calculate_player_score()
                all_scores.append(round_score)
                
                # Offer visualization after each round
                if Confirm.ask("\n💾 Save visualization of this round?", default=True):
                    self._generate_visualizations()
            
            if round_num < self.state.max_rounds:
                if not Confirm.ask("\nContinue to next round?", default=True):
                    break
        
        # Calculate final score as average of all rounds
        if all_scores:
            self.state.score = sum(all_scores) / len(all_scores)
    
    def _play_challenge_as_player(self, challenge: Challenge):
        """Play a single challenge as a player."""
        # Display challenge
        challenge_panel = Panel(
            f"[bold]{challenge.title}[/bold]\n\n{challenge.description}\n\n"
            f"[dim]Difficulty: {challenge.difficulty_level}/10 | Time Limit: {challenge.time_limit_minutes}min[/dim]",
            title="Challenge",
            border_style="yellow"
        )
        self.console.print(challenge_panel)
        
        # Create match
        if self.pipeline:
            match = self.pipeline.create_match(
                challenge_id=challenge.id,
                player1_id=self.human_player.id,
                player2_id=self.ai_player.id,
                coach1_id=self.ai_coach.id
            )
        else:
            # Fallback: create a simple match object
            match = Match(
                id=f"match_{random.randint(1000, 9999)}",
                challenge_id=challenge.id,
                player1_id=self.human_player.id,
                player2_id=self.ai_player.id,
                coach1_id=self.ai_coach.id,
                coach2_id=None,
                status=MatchStatus.ACTIVE,
                start_time=datetime.now(),
                end_time=None,
                player1_trace=None,
                player2_trace=None,
                coaching_events=[],
                results=None
            )
        
        self.state.current_match = match
        self.state.player_trace = ReasoningTrace(
            match_id=match.id,
            player_id=self.human_player.id,
            nodes=[],
            start_time=datetime.now()
        )
        
        # Start reasoning session
        self.console.print("\n[bold]Start reasoning! Type your thoughts step by step.[/bold]")
        self.console.print("[dim]Commands: 'submit' to finish, 'help' for coaching, 'quit' to exit[/dim]\n")
        
        start_time = datetime.now()
        reasoning_nodes = []
        
        while True:
            # Check time limit
            elapsed = (datetime.now() - start_time).total_seconds()
            time_limit_seconds = challenge.time_limit_minutes * 60
            remaining = time_limit_seconds - elapsed
            
            if remaining <= 0:
                self.console.print(f"[red]Time's up! ({elapsed:.0f}s elapsed)[/red]")
                break
            
            # Get user input
            try:
                mins_remaining = int(remaining // 60)
                secs_remaining = int(remaining % 60)
                time_str = f"{mins_remaining}:{secs_remaining:02d}" if mins_remaining > 0 else f"{secs_remaining}s"
                user_input = Prompt.ask(f"[cyan]Reasoning Step ({time_str} left)[/cyan]")
                
                if user_input.lower() == 'quit':
                    self.console.print("[yellow]Challenge abandoned.[/yellow]")
                    return
                
                elif user_input.lower() == 'submit':
                    break
                
                elif user_input.lower() == 'help':
                    self._provide_coaching_intervention()
                    continue
                
                # Create reasoning node
                node = ReasoningNode(
                    id=f"node_{len(reasoning_nodes)}",
                    content=user_input,
                    confidence=self._calculate_confidence(user_input, challenge),
                    timestamp=datetime.now(),
                    factor_contributions=self._analyze_factor_contributions(user_input),
                    parent_ids=[reasoning_nodes[-1].id] if reasoning_nodes else [],
                    child_ids=[]
                )
                
                # Update parent-child relationships
                if reasoning_nodes:
                    reasoning_nodes[-1].child_ids.append(node.id)
                
                reasoning_nodes.append(node)
                
                # Show confidence feedback
                self.console.print(f"[green]✓ Confidence: {node.confidence:.2f}[/green]")
                
                # Simulate AI coaching intervention
                if len(reasoning_nodes) % 3 == 0:  # Every 3 steps
                    self._provide_coaching_intervention()
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Challenge interrupted.[/yellow]")
                return
        
        # Finalize trace
        self.state.player_trace.nodes = reasoning_nodes
        self.state.player_trace.end_time = datetime.now()
        
        # Show results
        self._show_challenge_results(challenge, reasoning_nodes)
    
    def _play_as_coach(self):
        """Play as a coach providing interventions."""
        self.console.print("[bold green]🎯 Playing as Coach[/bold green]")
        self.console.print("You'll provide coaching interventions to help an AI player improve.\n")
        
        all_scores = []
        
        for round_num in range(1, self.state.max_rounds + 1):
            self.state.round_number = round_num
            challenge = random.choice(self.challenges)
            
            self.console.print(f"[bold yellow]Round {round_num}/{self.state.max_rounds}[/bold yellow]")
            self._play_challenge_as_coach(challenge)
            
            # Calculate and store round score
            if self.state.ai_trace and self.state.ai_trace.nodes:
                round_score = self._calculate_coaching_score()
                all_scores.append(round_score)
                
                # Offer visualization after each round
                if Confirm.ask("\n💾 Save visualization of this round?", default=True):
                    self._generate_visualizations()
            
            if round_num < self.state.max_rounds:
                if not Confirm.ask("\nContinue to next round?", default=True):
                    break
        
        # Calculate final score as average of all rounds
        if all_scores:
            self.state.score = sum(all_scores) / len(all_scores)
    
    def _play_challenge_as_coach(self, challenge: Challenge):
        """Play a single challenge as a coach."""
        # Display challenge
        challenge_panel = Panel(
            f"[bold]{challenge.title}[/bold]\n\n{challenge.description}\n\n"
            f"[dim]Difficulty: {challenge.difficulty_level}/10 | Time Limit: {challenge.time_limit_minutes}min[/dim]",
            title="Challenge",
            border_style="yellow"
        )
        self.console.print(challenge_panel)
        
        # Create match
        if self.pipeline:
            match = self.pipeline.create_match(
                challenge_id=challenge.id,
                player1_id=self.ai_player.id,
                player2_id=self.human_player.id,
                coach1_id="human_coach"
            )
        else:
            # Fallback: create a simple match object
            match = Match(
                id=f"match_{random.randint(1000, 9999)}",
                challenge_id=challenge.id,
                player1_id=self.ai_player.id,
                player2_id=self.human_player.id,
                coach1_id="human_coach",
                coach2_id=None,
                status=MatchStatus.ACTIVE,
                start_time=datetime.now(),
                end_time=None,
                player1_trace=None,
                player2_trace=None,
                coaching_events=[],
                results=None
            )
        
        self.state.current_match = match
        self.state.ai_trace = ReasoningTrace(
            match_id=match.id,
            player_id=self.ai_player.id,
            nodes=[],
            start_time=datetime.now()
        )
        
        self.console.print("\n[bold]Watch the AI player reason and provide coaching interventions.[/bold]")
        self.console.print("[dim]Commands: 'strategy', 'tactical', 'timeout', 'submit' to finish[/dim]\n")
        
        # Simulate AI reasoning with coaching opportunities
        self._simulate_ai_reasoning_with_coaching(challenge)
    
    def _simulate_ai_reasoning_with_coaching(self, challenge: Challenge):
        """Simulate AI player reasoning with coaching intervention points."""
        reasoning_steps = [
            "Looking at the sequence: 2, 6, 12, 20, 30",
            "Let me check the differences between consecutive numbers",
            "The differences are: 4, 6, 8, 10...",
            "The differences are increasing by 2 each time",
            "So the next difference should be 12",
            "Therefore, the next number should be 30 + 12 = 42"
        ]
        
        ai_nodes = []
        coaching_events = []
        
        for i, step in enumerate(reasoning_steps):
            # Show AI reasoning step
            self.console.print(f"[blue]AI: {step}[/blue]")
            
            # Create AI node
            node = ReasoningNode(
                id=f"ai_node_{i}",
                content=step,
                confidence=self._calculate_ai_confidence(step, challenge),
                timestamp=datetime.now(),
                factor_contributions=self._analyze_factor_contributions(step),
                parent_ids=[ai_nodes[-1].id] if ai_nodes else [],
                child_ids=[]
            )
            
            if ai_nodes:
                ai_nodes[-1].child_ids.append(node.id)
            
            ai_nodes.append(node)
            
            # Show confidence
            self.console.print(f"[dim]AI Confidence: {node.confidence:.2f}[/dim]")
            
            # Provide coaching opportunity every 2 steps
            if (i + 1) % 2 == 0 and i < len(reasoning_steps) - 1:
                self._get_coaching_input(coaching_events, node.timestamp)
            
            time.sleep(1)  # Pause for readability
        
        self.state.ai_trace.nodes = ai_nodes
        self.state.ai_trace.end_time = datetime.now()
        
        # Show coaching results
        self._show_coaching_results(challenge, ai_nodes, coaching_events)
    
    def _get_coaching_input(self, coaching_events: List, timestamp: datetime):
        """Get coaching input from human coach."""
        coaching_choice = Prompt.ask(
            "Provide coaching intervention",
            choices=["strategy", "tactical", "timeout", "none"],
            default="none"
        )
        
        if coaching_choice != "none":
            content = Prompt.ask("Enter coaching message")
            
            event = CoachingEvent(
                id=f"coaching_{len(coaching_events)}",
                coach_id="human_coach",
                player_id=self.ai_player.id,
                match_id=self.state.current_match.id,
                coaching_type=CoachingType(coaching_choice),
                content=content,
                timestamp=timestamp
            )
            
            coaching_events.append(event)
            display_name = "Strategic" if coaching_choice == "strategy" else coaching_choice.title()
            self.console.print(f"[green]✓ {display_name} coaching provided: {content}[/green]")
    
    def _provide_coaching_intervention(self):
        """Provide AI coaching intervention to human player."""
        coaching_messages = {
            "strategy": [
                "Consider the bigger picture and overall approach",
                "Think about the fundamental principles involved",
                "What's the core concept you're working with?"
            ],
            "tactical": [
                "Focus on the specific step you're working on",
                "Double-check your calculations",
                "Try breaking this into smaller parts"
            ]
        }
        
        coaching_type = random.choice(["strategy", "tactical"])
        message = random.choice(coaching_messages[coaching_type])
        
        display_name = "Strategic" if coaching_type == "strategy" else "Tactical"
        self.console.print(f"[yellow]🤖 AI Coach ({display_name}): {message}[/yellow]")
        
        # Create coaching event
        if self.state.current_match:
            event = CoachingEvent(
                id=f"ai_coaching_{len(self.state.current_match.coaching_events)}",
                coach_id=self.ai_coach.id,
                player_id=self.human_player.id,
                match_id=self.state.current_match.id,
                coaching_type=CoachingType(coaching_type),
                content=message,
                timestamp=datetime.now()
            )
            self.state.current_match.coaching_events.append(event)
    
    def _calculate_confidence(self, reasoning: str, challenge: Challenge) -> float:
        """Calculate confidence score for human reasoning."""
        base_confidence = 0.5
        
        # Factor in reasoning length
        if len(reasoning.split()) > 5:
            base_confidence += 0.1
        
        # Factor in mathematical terms
        math_terms = ["difference", "sequence", "pattern", "calculate", "formula"]
        if any(term in reasoning.lower() for term in math_terms):
            base_confidence += 0.15
        
        # Factor in logical connectors
        logical_terms = ["therefore", "because", "so", "thus", "since"]
        if any(term in reasoning.lower() for term in logical_terms):
            base_confidence += 0.1
        
        return min(0.95, max(0.1, base_confidence + random.uniform(-0.1, 0.1)))
    
    def _calculate_ai_confidence(self, reasoning: str, challenge: Challenge) -> float:
        """Calculate confidence score for AI reasoning."""
        base_confidence = 0.6 + (challenge.difficulty_level / 10.0) * 0.2
        return min(0.95, max(0.3, base_confidence + random.uniform(-0.05, 0.05)))
    
    def _analyze_factor_contributions(self, reasoning: str) -> Dict[FactorType, float]:
        """Analyze factor contributions in reasoning."""
        contributions = {}
        
        # Simple keyword-based analysis
        if any(term in reasoning.lower() for term in ["unique", "different", "novel", "creative"]):
            contributions[FactorType.UNIQUENESS] = 0.7
        
        if any(term in reasoning.lower() for term in ["connect", "relate", "together", "coherent"]):
            contributions[FactorType.COHESION] = 0.6
        
        if any(term in reasoning.lower() for term in ["learn", "understand", "insight", "realize"]):
            contributions[FactorType.LEARNING] = 0.8
        
        if any(term in reasoning.lower() for term in ["quality", "correct", "accurate", "precise"]):
            contributions[FactorType.QUALITY] = 0.7
        
        if any(term in reasoning.lower() for term in ["trust", "reliable", "consistent", "valid"]):
            contributions[FactorType.TRUST] = 0.5
        
        if any(term in reasoning.lower() for term in ["dense", "compact", "efficient", "concise"]):
            contributions[FactorType.DENSITY] = 0.6
        
        # Default contributions if none found
        if not contributions:
            contributions[FactorType.LEARNING] = 0.5
            contributions[FactorType.COHESION] = 0.4
        
        return contributions
    
    def _show_challenge_results(self, challenge: Challenge, nodes: List[ReasoningNode]):
        """Show results of a challenge."""
        if not nodes:
            self.console.print("[red]No reasoning steps recorded.[/red]")
            return
        
        # Calculate score
        total_confidence = sum(node.confidence for node in nodes)
        avg_confidence = total_confidence / len(nodes)
        
        # Show results table
        table = Table(title="Challenge Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Reasoning Steps", str(len(nodes)))
        table.add_row("Average Confidence", f"{avg_confidence:.2f}")
        table.add_row("Total Confidence", f"{total_confidence:.2f}")
        table.add_row("Challenge Difficulty", f"{challenge.difficulty_level}/10")
        
        self.console.print(table)
        
        # Show reasoning trace
        self.console.print("\n[bold]Your Reasoning Trace:[/bold]")
        for i, node in enumerate(nodes, 1):
            self.console.print(f"{i}. [dim]{node.timestamp.strftime('%H:%M:%S')}[/dim] {node.content}")
            self.console.print(f"   Confidence: {node.confidence:.2f}")
    
    def _show_coaching_results(self, challenge: Challenge, ai_nodes: List[ReasoningNode], coaching_events: List):
        """Show results of coaching session."""
        if not ai_nodes:
            self.console.print("[red]No AI reasoning steps recorded.[/red]")
            return
        
        # Calculate coaching effectiveness
        total_confidence = sum(node.confidence for node in ai_nodes)
        avg_confidence = total_confidence / len(ai_nodes)
        
        # Show results table
        table = Table(title="Coaching Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("AI Reasoning Steps", str(len(ai_nodes)))
        table.add_row("Average AI Confidence", f"{avg_confidence:.2f}")
        table.add_row("Coaching Interventions", str(len(coaching_events)))
        table.add_row("Challenge Difficulty", f"{challenge.difficulty_level}/10")
        
        self.console.print(table)
        
        # Show coaching events
        if coaching_events:
            self.console.print("\n[bold]Your Coaching Interventions:[/bold]")
            for i, event in enumerate(coaching_events, 1):
                self.console.print(f"{i}. [yellow]{event.coaching_type.value.title()}[/yellow]: {event.content}")
    
    def _calculate_player_score(self) -> float:
        """Calculate final player score."""
        if not self.state.player_trace or not self.state.player_trace.nodes:
            return 0.0
        
        # Use the IdeaRank-Thought scorer if available
        if self.scorer:
            try:
                score = self.scorer.calculate_raw_score(self.state.player_trace)
                return score
            except Exception:
                pass
        
        # Fallback: calculate simple score based on confidence
        total_confidence = sum(node.confidence for node in self.state.player_trace.nodes)
        avg_confidence = total_confidence / len(self.state.player_trace.nodes)
        return avg_confidence * 0.8  # Scale down for fallback scoring
    
    def _calculate_coaching_score(self) -> float:
        """Calculate coaching effectiveness score."""
        if not self.state.ai_trace or not self.state.ai_trace.nodes:
            return 0.0
        
        # Base score on AI performance improvement
        avg_confidence = sum(node.confidence for node in self.state.ai_trace.nodes) / len(self.state.ai_trace.nodes)
        return avg_confidence * 0.8  # Coaching gets 80% of AI's performance
    
    def _show_final_results(self):
        """Show final game results."""
        results_text = f"""
🎉 Game Complete! 🎉

Mode: {self.state.mode.value.title()}
Rounds Completed: {self.state.round_number}
Final Score: {self.state.score:.3f}

"""
        
        if self.state.mode == GameMode.PLAYER:
            if self.state.score >= 0.7:
                results_text += "🏆 Excellent reasoning performance!"
            elif self.state.score >= 0.5:
                results_text += "👍 Good reasoning skills!"
            else:
                results_text += "💪 Keep practicing your reasoning!"
        else:
            if self.state.score >= 0.6:
                results_text += "🎯 Excellent coaching effectiveness!"
            elif self.state.score >= 0.4:
                results_text += "👏 Good coaching skills!"
            else:
                results_text += "📚 Keep developing your coaching abilities!"
        
        panel = Panel(
            Align.center(results_text),
            title="[bold green]Final Results[/bold green]",
            border_style="green"
        )
        
        self.console.print(panel)
    
    def _generate_visualizations(self):
        """Generate visualizations of the game performance."""
        try:
            if self.state.mode == GameMode.PLAYER and self.state.player_trace:
                # Check if we have reasoning nodes
                if not self.state.player_trace.nodes:
                    self.console.print("[yellow]No reasoning steps to visualize. Try solving a challenge first![/yellow]")
                    return
                
                # Assign trace to match for visualization
                if self.state.current_match:
                    self.state.current_match.player1_trace = self.state.player_trace
                
                # Generate reason map
                fig = self.visualizer.create_reason_map(
                    self.state.player_trace,
                    self.state.current_match,
                    self.human_player.id,
                    f"Your Reasoning Performance - Round {self.state.round_number}"
                )
                
                if fig:
                    filename = f"interactive_game_player_round_{self.state.round_number}.html"
                    fig.write_html(filename)
                    self.console.print(f"[green]✓ Reason map saved to {filename}[/green]")
                    self.console.print(f"[dim]Open {filename} in your browser to view your reasoning trace![/dim]")
            
            elif self.state.mode == GameMode.COACH and self.state.ai_trace:
                # Check if we have AI reasoning nodes
                if not self.state.ai_trace.nodes:
                    self.console.print("[yellow]No AI reasoning to visualize.[/yellow]")
                    return
                
                # Assign trace to match for visualization
                if self.state.current_match:
                    self.state.current_match.player1_trace = self.state.ai_trace
                
                # Generate coaching impact chart
                fig = self.visualizer.create_coaching_impact_chart(
                    self.state.current_match,
                    self.ai_player.id
                )
                
                if fig:
                    filename = f"interactive_game_coach_round_{self.state.round_number}.html"
                    fig.write_html(filename)
                    self.console.print(f"[green]✓ Coaching impact chart saved to {filename}[/green]")
                    self.console.print(f"[dim]Open {filename} in your browser to view coaching impact![/dim]")
        
        except Exception as e:
            self.console.print(f"[red]Error generating visualizations: {e}[/red]")
            import traceback
            self.console.print(f"[dim]{traceback.format_exc()}[/dim]")

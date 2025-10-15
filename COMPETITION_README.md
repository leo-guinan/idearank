# IdeaRank-Thought Competition System

## Overview

The IdeaRank-Thought Competition System is a real-time reasoning evaluation platform that implements the MetaSPN competition framework described in the technical appendix. It enables live matches between players solving complex challenges, with coaching interventions and comprehensive scoring.

## Key Features

### 🏆 **Competition Framework**
- **Real-time matches** between players solving challenges
- **Coaching interventions** during matches with timeout management
- **Comprehensive scoring** using the IdeaRank-Thought algorithm
- **Integrity protocols** with cryptographic verification

### 🧠 **Reasoning Evaluation**
- **Factor-based scoring** (U, C, L, Q, T, D) as described in the technical appendix
- **Gate validation** with Outcome Validity (O) and Constraint Compliance (X)
- **Anti-gaming measures** using multiplicative scoring
- **Coaching impact analysis** with timeout delta measurements

### 📊 **Visualization System**
- **Reason Map Overlays** with multi-dimensional visual encoding
- **Interactive features** for spectators and analysis
- **Real-time coaching impact** visualization
- **Factor contribution tracking** over time

### 🔒 **Integrity & Validation**
- **Ledger hashing** with SHA-256 for match verification
- **Replay validation** for reproducibility
- **Anti-gaming measures** preventing single-factor optimization
- **Audit trails** for complete match transparency

## Quick Start

### Installation

The competition system is included with IdeaRank. Ensure you have the required dependencies:

```bash
pip install idearank[competition]
```

### Basic Usage

```python
from idearank import (
    CompetitionPipeline, Player, Coach, Challenge, 
    MatchStatus, CoachingType, FactorType
)
from idearank.providers.embeddings import SentenceTransformerEmbeddingProvider
from idearank.providers.chroma import ChromaProvider

# Initialize providers
embedding_provider = SentenceTransformerEmbeddingProvider()
chroma_provider = ChromaProvider(collection_name="competition")

# Create pipeline
pipeline = CompetitionPipeline(
    embedding_provider=embedding_provider,
    chroma_provider=chroma_provider
)

# Create a challenge
challenge = Challenge(
    id="logic_puzzle",
    title="The Missing Number Sequence",
    description="Given the sequence: 2, 6, 12, 20, 30, ? Find the next number.",
    difficulty_level=6,
    time_limit_minutes=15,
    constraints=["Must show step-by-step reasoning"],
    ground_truth={"answer": 42, "pattern": "n(n+1)"}
)

# Create players
player1 = Player(id="alice", name="Alice Chen")
player2 = Player(id="bob", name="Bob Rodriguez")

# Create coaches
coach1 = Coach(id="coach1", name="Dr. Sarah Johnson")

# Register entities
pipeline.register_challenge(challenge)
pipeline.register_player(player1)
pipeline.register_player(player2)
pipeline.register_coach(coach1)

# Create and start a match
match = pipeline.create_match(
    challenge_id=challenge.id,
    player1_id=player1.id,
    player2_id=player2.id,
    coach1_id=coach1.id,
    coach2_id=coach1.id
)

pipeline.start_match(match.id)

# Add reasoning nodes (simulated)
pipeline.add_reasoning_node(
    match_id=match.id,
    player_id=player1.id,
    content="Looking at the sequence: 2, 6, 12, 20, 30",
    confidence=0.7,
    factor_contributions={FactorType.UNIQUENESS: 0.3, FactorType.QUALITY: 0.4}
)

# Add coaching intervention
pipeline.add_coaching_event(
    match_id=match.id,
    player_id=player1.id,
    coach_id=coach1.id,
    coaching_type=CoachingType.STRATEGY,
    content="Remember to consider multiple approaches"
)

# Complete the match
results = pipeline.complete_match(match.id)
print(f"Player 1 Score: {results['scores'][player1.id].raw_score:.3f}")
```

## Architecture

### Core Components

#### 1. **Competition Models** (`competition_models.py`)
- `Player`: Competition participants with stats and ratings
- `Coach`: Coaching entities with specialization and effectiveness
- `Challenge`: Problems to be solved with constraints and evaluation criteria
- `Match`: Competition instances with reasoning traces and coaching events
- `ReasoningTrace`: Complete reasoning paths with nodes and connections
- `ReasoningNode`: Individual reasoning steps with factor contributions
- `CoachingEvent`: Coaching interventions with timing and content

#### 2. **Scoring Engine** (`competition_scorer.py`)
- `IdeaRankThoughtScorer`: Main scoring engine implementing the technical appendix
- **Gate Calculations**: Outcome Validity (O) and Constraint Compliance (X)
- **Factor Scoring**: U, C, L, Q, T, D factors with anti-gaming measures
- **Coaching Analysis**: Impact measurement and plan adherence

#### 3. **Pipeline Management** (`competition_pipeline.py`)
- `CompetitionPipeline`: Main orchestrator for match lifecycle
- **Match Management**: Creation, execution, and completion
- **Real-time Processing**: Reasoning trace collection and coaching events
- **Integrity Protocols**: Ledger hashing and validation

#### 4. **Visualization System** (`competition_visualizer.py`)
- `CompetitionVisualizer`: Reason Map Overlay implementation
- **Visual Encoding**: Node colors, edge thickness, halo glow effects
- **Interactive Features**: Factor isolation, replay, detailed breakdowns
- **Export Capabilities**: Data export for external visualization

## Scoring System

### Factor Computation

Each factor is calculated using specific methods from the technical appendix:

| Factor | Method | Description |
|--------|--------|-------------|
| **U** (Uniqueness) | Novelty z-score → sigmoid | Novelty vs league corpus |
| **C** (Cohesion) | Reasoning map analysis | Structural consistency, linguistic coherence |
| **L** (Learning) | Early/late checkpoint delta | Skill gain function over time |
| **Q** (Quality) | Human & AI rubric | Readability, completeness, clarity |
| **T** (Trust) | Test logs, citations | Pass rate × citation quality |
| **D** (Density) | Tokens vs valid nodes | Information-per-step compression |

### Gate Formulas

#### Outcome Validity (O)
```
O = 0.7 × correctness + 0.3 × robustness
```

#### Constraint Compliance (X)
```
X = 1 - violation_penalty
```

### Anti-Gaming Measures

- **Multiplicative scoring** prevents single-factor dominance
- **Balanced performance** across factors yields higher scores
- **Extreme specialization** results in score penalties

## Visualization Schema

### Reason Map Overlays

The visualization system implements the multi-dimensional encoding:

#### Visual Encoding
- **Node color** → Factor weight contribution
  - Hue indicates dominant factor (blue=Cohesion, green=Learning, etc.)
  - Saturation indicates contribution magnitude
- **Edge thickness** → Cohesion strength
  - Thicker edges represent stronger logical connections
  - Dashed edges indicate tentative reasoning
- **Halo glow** → Coaching intervention zones
  - Glowing nodes indicate post-timeout reasoning
  - Glow intensity correlates with coaching impact

#### Interactive Features
- **Factor isolation**: View individual factor contributions
- **Replay sequences**: Variable speed reasoning playback
- **Alternative paths**: Compare reasoning branches not taken
- **Detailed breakdowns**: Per-node factor analysis

## Integrity Protocols

### Ledger Hashing
- **SHA-256** cryptographic hashing of match data
- **Merkle tree structure** for efficient partial verification
- **RFC 3161** compliant trusted timestamps

### Replay Validation
- **Deterministic execution** with fixed random seeds
- **Bit-level comparison** for AI system verification
- **Ghost replay protocols** for human reasoning validation

### Anti-Gaming Measures
- **Multiplicative scoring** prevents optimization of single factors
- **Balance requirements** across all scoring dimensions
- **Penalty systems** for extreme specialization

## Output Metrics

Each match produces comprehensive performance metrics:

- **IR-T Raw Score (0–1)**: Unmodified IdeaRank-Thought score
- **IR-T Class Adjusted Score**: Raw score with class-specific adjustments
- **Coach Impact Index (CI)**: Coaching effectiveness [-1.0, +1.0]
- **Meta Rating (MR)**: Long-term performance composite
- **Entropy Index (EI)**: Reasoning tree efficiency for tiebreaks

## Example Usage

### Running the Demo

```bash
python examples/competition_demo.py
```

This will:
1. Create demo players, coaches, and challenges
2. Run a complete match with reasoning traces
3. Apply coaching interventions
4. Calculate comprehensive scores
5. Generate visualizations
6. Display results and leaderboard

### Custom Challenges

```python
# Create a custom challenge
challenge = Challenge(
    id="my_challenge",
    title="Custom Problem",
    description="Your problem description here...",
    difficulty_level=7,
    time_limit_minutes=20,
    constraints=["Constraint 1", "Constraint 2"],
    success_criteria={
        "correctness": 0.8,
        "creativity": 0.7
    },
    ground_truth={"expected_answer": "solution"},
    evaluation_rubric={
        "correctness": 0.4,
        "reasoning_quality": 0.3,
        "creativity": 0.3
    }
)

pipeline.register_challenge(challenge)
```

### Custom Scoring

```python
# The scorer implements all methods from the technical appendix
scorer = IdeaRankThoughtScorer(embedding_provider)

# Calculate scores for a completed match
scores = scorer.score_match(match, challenge)

# Access detailed factor breakdowns
for player_id, score in scores.items():
    print(f"Player {player_id}:")
    print(f"  Raw Score: {score.raw_score:.3f}")
    print(f"  Factor Scores: {score.factor_scores}")
    print(f"  Gate Scores: O={score.outcome_validity.score:.3f}, X={score.constraint_compliance.score:.3f}")
    print(f"  Coach Impact: {score.coach_impact_index:.3f}")
```

## Advanced Features

### Real-time Visualization

```python
from idearank.competition_visualizer import CompetitionVisualizer

visualizer = CompetitionVisualizer()

# Create reason map for a player
fig = visualizer.create_reason_map(
    trace=match.player1_trace,
    match=match,
    player_id=player1.id,
    title="Alice's Reasoning Process"
)

# Save as interactive HTML
fig.write_html("reason_map.html")
```

### Coaching Analysis

```python
# Analyze coaching effectiveness
coaching_events = pipeline.get_coaching_events(match.id, player1.id)

for event in coaching_events:
    print(f"Coaching Type: {event.coaching_type.value}")
    print(f"Content: {event.content}")
    print(f"Effectiveness: {event.effectiveness_score}")
```

### Data Export

```python
# Export complete match data
match_data = pipeline.export_match_data(match.id)

# Export visualization data
viz_data = visualizer.export_visualization_data(match, player1.id)

# Save for external analysis
import json
with open("match_data.json", "w") as f:
    json.dump(match_data, f, indent=2, default=str)
```

## Integration with Existing IdeaRank

The competition system integrates seamlessly with the existing IdeaRank framework:

- **Shared providers**: Uses the same embedding and Chroma providers
- **Compatible models**: Extends existing data structures
- **Unified configuration**: Works with existing IdeaRank config
- **Pipeline integration**: Can be used alongside content processing

## Open Science Compliance

Following the MetaSPN Charter principles:

- **Open Standards**: All algorithms and rubrics are publicly documented
- **Reference Implementations**: Available in this open-source repository
- **Validation Datasets**: Provided for calibration and testing
- **Auditability**: Complete transparency in scoring calculations

## Technical Specifications

### Dependencies
- `plotly`: Interactive visualizations
- `numpy`: Numerical calculations
- `chromadb`: Vector storage for reasoning traces
- `sentence-transformers`: Embedding generation

### Performance
- **Real-time processing**: Sub-second reasoning node addition
- **Scalable storage**: ChromaDB handles large reasoning traces
- **Efficient scoring**: Optimized factor calculations
- **Memory efficient**: Streaming processing for long matches

## Future Enhancements

Planned improvements include:

- **Multi-language support** for international competitions
- **Advanced coaching AI** for automated coaching suggestions
- **Tournament management** with brackets and elimination
- **Mobile interface** for real-time match participation
- **API endpoints** for external integration
- **Machine learning** for improved factor scoring

## Contributing

The competition system follows the same contribution guidelines as IdeaRank. See the main README for details.

## License

Same license as IdeaRank. See LICENSE file for details.

---

*For the latest technical documentation and updates, visit the [IdeaRank documentation](https://idearank.readthedocs.io/competition).*

# IdeaRank

**A PageRank replacement for ideas, not links.**

IdeaRank is a multi-factor ranking algorithm for video content that rewards:
- **Uniqueness** (U): Semantic originality vs. the global corpus
- **Cohesion** (C): Fit within the channel's conceptual lattice
- **Learning** (L): Forward progression, not repetition
- **Quality** (Q): Genuine engagement, not clickbait
- **Trust** (T): Verifiable claims and sources

## Formula

```
IR(v,t) = U^w_U · C^w_C · L^w_L · Q^w_Q · T^w_T
```

Each factor is independently computed and configurable, making IdeaRank adaptable to different domains and use cases.

## Architecture

IdeaRank is designed for **modularity and experimentation**:

```
idearank/
├── models.py           # Core data models (Video, Channel, Embedding)
├── config.py           # Hyperparameter configuration
├── factors/            # Independent factor modules
│   ├── uniqueness.py   # U factor
│   ├── cohesion.py     # C factor
│   ├── learning.py     # L factor
│   ├── quality.py      # Q factor
│   └── trust.py        # T factor
├── scorer.py           # Video and channel scoring
├── network.py          # Optional KnowledgeRank network layer
├── pipeline.py         # End-to-end orchestration
└── providers/          # Pluggable implementations
    ├── embeddings.py   # Embedding providers
    ├── topics.py       # Topic model providers
    └── neighborhoods.py # ANN search providers
```

## Installation

```bash
# Basic installation
pip install -e .

# With optional dependencies
pip install -e ".[openai]"    # For OpenAI embeddings
pip install -e ".[faiss]"     # For FAISS ANN search
pip install -e ".[ml]"        # For scikit-learn topic models
pip install -e ".[all]"       # Everything
```

## Quick Start

```python
from datetime import datetime
from idearank import IdeaRankConfig, Video, Channel
from idearank.pipeline import IdeaRankPipeline
from idearank.providers import (
    DummyEmbeddingProvider,
    DummyTopicModelProvider,
    DummyNeighborhoodProvider,
)

# 1. Create configuration
config = IdeaRankConfig.default()

# 2. Initialize providers
embedding_provider = DummyEmbeddingProvider()
topic_provider = DummyTopicModelProvider()
neighborhood_provider = DummyNeighborhoodProvider()

# 3. Create pipeline
pipeline = IdeaRankPipeline(
    config=config,
    embedding_provider=embedding_provider,
    topic_provider=topic_provider,
    neighborhood_provider=neighborhood_provider,
)

# 4. Create your data
video = Video(
    id="video_1",
    channel_id="channel_1",
    title="Introduction to IdeaRank",
    description="How to rank ideas instead of links",
    transcript="In this video we explore...",
    published_at=datetime(2024, 1, 1),
    snapshot_time=datetime.utcnow(),
    # ... analytics and trust signals ...
)

channel = Channel(
    id="channel_1",
    name="IdeaRank Channel",
    description="Content about ranking algorithms",
    created_at=datetime(2023, 1, 1),
    videos=[video],
)

# 5. Score the video
pipeline.process_videos_batch(channel.videos)
pipeline.index_videos(channel.videos)

score = pipeline.score_video(video, channel)

print(f"IdeaRank Score: {score.score:.4f}")
print(f"  Uniqueness:  {score.uniqueness.score:.4f}")
print(f"  Cohesion:    {score.cohesion.score:.4f}")
print(f"  Learning:    {score.learning.score:.4f}")
print(f"  Quality:     {score.quality.score:.4f}")
print(f"  Trust:       {score.trust.score:.4f}")
```

## Configuration

All hyperparameters are configurable via `IdeaRankConfig`:

```python
from idearank import IdeaRankConfig
from idearank.config import FactorWeights

config = IdeaRankConfig(
    weights=FactorWeights(
        uniqueness=0.35,  # w_U
        cohesion=0.20,    # w_C
        learning=0.25,    # w_L
        quality=0.15,     # w_Q
        trust=0.05,       # w_T
    ),
    # Per-factor configs
    uniqueness=UniquenessConfig(k_global=50, min_threshold=0.15),
    cohesion=CohesionConfig(k_intra=15, window_days=270),
    learning=LearningConfig(stability_sigma=0.5, min_threshold=0.05),
    # ... etc
)
```

### Default Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k_global` | 50 | Global neighbors for Uniqueness |
| `k_intra` | 15 | Intra-channel neighbors for Cohesion |
| `window_days` | 180 | Channel scoring window |
| `stability_sigma` | 0.5 | Learning stability penalty |
| `damping_factor` | 0.75 | KnowledgeRank damping (if enabled) |

See `idearank/config.py` for all parameters.

## Channel-Level Scoring

IdeaRank aggregates video scores to channel scores with a learning bonus:

```python
channel_score = pipeline.score_channel(channel)

print(f"Channel IdeaRank: {channel_score.score:.4f}")
print(f"  Mean Video Score: {channel_score.mean_video_score:.4f}")
print(f"  AUL Bonus: {channel_score.aul_bonus:.4f}")
print(f"  Crystallization Detected: {channel_score.crystallization_detected}")
```

**Area Under Learning (AUL)**: Rewards consistent forward progression over time.

**Anti-Crystallization**: Detects when a channel stops learning and applies decay.

## Network Layer (KnowledgeRank)

Optional cross-channel influence layer:

```python
config.network.enabled = True

kr_scores = pipeline.score_channels_with_network(channels)

for channel_id, kr in kr_scores.items():
    print(f"{channel_id}: KR={kr.knowledge_rank:.4f}, IR={kr.idea_rank:.4f}")
    print(f"  Influence bonus: {kr.influence_bonus:+.4f}")
    print(f"  Outgoing edges: {len(kr.outgoing_influence)}")
    print(f"  Incoming edges: {len(kr.incoming_influence)}")
```

**KnowledgeRank** measures "who learns from whom" using semantic similarity with temporal lag.

## Providers

IdeaRank uses pluggable providers for external services:

### Embedding Providers

```python
# Dummy (for testing)
from idearank.providers import DummyEmbeddingProvider
provider = DummyEmbeddingProvider(dimension=384)

# OpenAI (requires API key)
from idearank.providers import OpenAIEmbeddingProvider
provider = OpenAIEmbeddingProvider(model="text-embedding-3-small", api_key="...")
```

### Topic Model Providers

```python
# Dummy (for testing)
from idearank.providers import DummyTopicModelProvider
provider = DummyTopicModelProvider(num_topics=50)

# LDA (requires trained model)
from idearank.providers import LDATopicModelProvider
provider = LDATopicModelProvider(model_path="path/to/lda.pkl")
```

### Neighborhood Providers

```python
# Dummy (brute-force, for testing)
from idearank.providers import DummyNeighborhoodProvider
provider = DummyNeighborhoodProvider()

# FAISS (for production)
from idearank.providers import FAISSNeighborhoodProvider
provider = FAISSNeighborhoodProvider(dimension=384)
```

## Examples

See the `examples/` directory:

- `basic_usage.py`: End-to-end example
- `custom_weights.py`: Comparing different weight configurations

```bash
python examples/basic_usage.py
python examples/custom_weights.py
```

## Use Cases

IdeaRank is designed for:

1. **Content Discovery**: Surface genuinely novel ideas, not viral clickbait
2. **Creator Dashboards**: Show learning progression and channel health
3. **Research**: Track idea evolution across a knowledge graph
4. **Recommendation**: Suggest content that advances understanding
5. **Trust Scoring**: Identify well-sourced, verifiable content

## Design Principles

1. **Modularity**: Each factor is independent and swappable
2. **Configurability**: All hyperparameters exposed and tunable
3. **Transparency**: Full score breakdowns for debugging
4. **Extensibility**: Easy to add new factors or providers
5. **Anti-Gaming**: Multiple factors prevent optimization for any single metric

## What It's NOT

- ❌ A replacement for engagement metrics (it uses them, normalized)
- ❌ A content moderation tool (it measures learning, not truthfulness)
- ❌ A production-ready system (it's a research framework)
- ❌ Optimized for speed (it's optimized for clarity)

## Roadmap

- [ ] Real OpenAI and FAISS provider implementations
- [ ] Pre-trained topic models for common domains
- [ ] Analytics normalization across channels
- [ ] Citation parsing and verification
- [ ] Temporal trend analysis
- [ ] Interactive visualization dashboard
- [ ] Benchmark datasets

## License

MIT

## Citation

If you use IdeaRank in research, please cite:

```
@software{idearank2024,
  title={IdeaRank: A Multi-Factor Ranking Algorithm for Ideas},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/idearank}
}
```

## Contributing

This is a research framework. Contributions welcome:

1. Fork the repo
2. Create a feature branch
3. Add tests for new functionality
4. Submit a PR

## Contact

Questions? Open an issue or reach out at [your contact].

---

**Remember**: All innovation is cyclical amnesia. IdeaRank just measures how well you're navigating the cycle.


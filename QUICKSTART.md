# IdeaRank Quickstart

Get started with IdeaRank in 5 minutes.

## Installation

```bash
# Clone the repo
git clone <your-repo-url>
cd idearank

# Install in development mode
pip install -e .

# Or with optional dependencies
pip install -e ".[dev,openai,faiss,ml]"
```

## Your First Score

```python
from datetime import datetime
from idearank import IdeaRankConfig, Video, Channel
from idearank.pipeline import IdeaRankPipeline
from idearank.providers import (
    DummyEmbeddingProvider,
    DummyTopicModelProvider,
    DummyNeighborhoodProvider,
)

# 1. Setup
config = IdeaRankConfig.default()
pipeline = IdeaRankPipeline(
    config=config,
    embedding_provider=DummyEmbeddingProvider(),
    topic_provider=DummyTopicModelProvider(),
    neighborhood_provider=DummyNeighborhoodProvider(),
)

# 2. Create a video
video = Video(
    id="intro_video",
    channel_id="my_channel",
    title="Introduction to Machine Learning",
    description="A beginner's guide to ML concepts",
    transcript="In this video, we cover the basics of machine learning...",
    published_at=datetime(2024, 1, 1),
    snapshot_time=datetime.utcnow(),
    view_count=10000,
    impression_count=50000,
    watch_time_seconds=500000.0,
    avg_view_duration=250.0,
    video_duration=600.0,
    has_citations=True,
    citation_count=5,
    source_diversity_score=0.8,
)

# 3. Create a channel
channel = Channel(
    id="my_channel",
    name="My ML Channel",
    description="Educational content about machine learning",
    created_at=datetime(2023, 1, 1),
    videos=[video],
)

# 4. Process and score
pipeline.process_videos_batch(channel.videos)
pipeline.index_videos(channel.videos)
score = pipeline.score_video(video, channel)

# 5. View results
print(f"IdeaRank Score: {score.score:.4f}")
print(f"  Uniqueness:  {score.uniqueness.score:.4f}")
print(f"  Cohesion:    {score.cohesion.score:.4f}")
print(f"  Learning:    {score.learning.score:.4f}")
print(f"  Quality:     {score.quality.score:.4f}")
print(f"  Trust:       {score.trust.score:.4f}")
```

## Run Examples

```bash
# Basic usage
python examples/basic_usage.py

# Compare configurations
python examples/custom_weights.py
```

## Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=idearank

# Specific test file
pytest tests/test_models.py -v
```

## Customize Weights

```python
from idearank.config import FactorWeights

# Emphasize trust for journalism
journalism_config = IdeaRankConfig(
    weights=FactorWeights(
        uniqueness=0.2,
        cohesion=0.15,
        learning=0.15,
        quality=0.1,
        trust=0.4,  # High!
    )
)

# Emphasize novelty for research
research_config = IdeaRankConfig(
    weights=FactorWeights(
        uniqueness=0.6,  # Very high!
        cohesion=0.1,
        learning=0.2,
        quality=0.05,
        trust=0.05,
    )
)
```

## Use Real Providers

### OpenAI Embeddings (When Implemented)

```python
from idearank.providers import OpenAIEmbeddingProvider

provider = OpenAIEmbeddingProvider(
    model="text-embedding-3-small",
    api_key="sk-..."
)
```

### FAISS ANN Search (When Implemented)

```python
from idearank.providers import FAISSNeighborhoodProvider

provider = FAISSNeighborhoodProvider(dimension=384)
```

## Score Multiple Channels

```python
channels = [channel1, channel2, channel3]

# With network effects
config.network.enabled = True
kr_scores = pipeline.score_channels_with_network(channels)

for channel_id, kr in kr_scores.items():
    print(f"{channel_id}:")
    print(f"  KnowledgeRank: {kr.knowledge_rank:.4f}")
    print(f"  IdeaRank: {kr.idea_rank:.4f}")
    print(f"  Influence Bonus: {kr.influence_bonus:+.4f}")
```

## Export Results

```python
# To dict
video_dict = score.to_dict()

# To JSON
import json
with open("scores.json", "w") as f:
    json.dump(video_dict, f, indent=2)

# To pandas (for analysis)
import pandas as pd
df = pd.DataFrame([
    {
        'video_id': score.video_id,
        'score': score.score,
        'uniqueness': score.uniqueness.score,
        'cohesion': score.cohesion.score,
        'learning': score.learning.score,
        'quality': score.quality.score,
        'trust': score.trust.score,
    }
    for score in all_scores
])
```

## Debugging

Each factor returns detailed components:

```python
# Uniqueness breakdown
print(score.uniqueness.components)
# {
#   'mean_similarity': 0.42,
#   'neighbor_count': 50,
#   'min_similarity': 0.12,
#   'max_similarity': 0.89,
# }

# Learning breakdown
print(score.learning.components)
# {
#   'delta_self': 0.65,
#   'revision_quality': 1.0,
#   'stability': 0.87,
# }
```

## Common Patterns

### Batch Processing

```python
# Process many videos efficiently
all_videos = [v for c in channels for v in c.videos]
pipeline.process_videos_batch(all_videos)
pipeline.index_videos(all_videos)

# Then score them
scores = [
    pipeline.score_video(video, channel)
    for channel in channels
    for video in channel.videos
]
```

### Time Series Analysis

```python
# Score a video at different time snapshots
snapshots = [
    datetime(2024, 1, 1),
    datetime(2024, 3, 1),
    datetime(2024, 6, 1),
]

for snapshot_time in snapshots:
    score = pipeline.score_channel(channel, end_time=snapshot_time)
    print(f"{snapshot_time}: {score.score:.4f}")
```

### A/B Testing Configs

```python
configs = {
    "default": IdeaRankConfig.default(),
    "experimental": IdeaRankConfig.experimental(),
    "custom": my_custom_config,
}

results = {}
for name, config in configs.items():
    pipeline = IdeaRankPipeline(config, ...)
    score = pipeline.score_video(video, channel)
    results[name] = score.score

# Compare
print(results)
# {'default': 0.52, 'experimental': 0.48, 'custom': 0.61}
```

## Next Steps

1. Read [ARCHITECTURE.md](ARCHITECTURE.md) for system design
2. Read [README.md](README.md) for full documentation
3. Explore `idearank/factors/` to understand each factor
4. Check `examples/` for more use cases
5. Implement real providers (OpenAI, FAISS) for production

## Getting Help

- Check the [README](README.md) for detailed docs
- Read [ARCHITECTURE](ARCHITECTURE.md) for system internals
- Look at `examples/` for working code
- Run tests with `-v` for debugging
- Add `logging.basicConfig(level=logging.DEBUG)` for pipeline traces

## Tips

1. **Start with dummy providers** - They're deterministic and fast
2. **Use experimental config** - Faster for iteration
3. **Check `passes_gates`** - Ensures minimum quality thresholds
4. **Inspect `components`** - Debug why a score is what it is
5. **Normalize analytics** - Quality factor needs proper distributions

---

**You're ready to rank ideas instead of links. Go forth and measure learning progression!** 🚀


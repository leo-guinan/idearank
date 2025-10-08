# IdeaRank Architecture

## Overview

IdeaRank is designed with **modularity** and **configurability** as first principles. Every component can be swapped, every hyperparameter tuned, and every computation traced.

## Core Equation

```
IR(v,t) = U^w_U · C^w_C · L^w_L · Q^w_Q · T^w_T
```

Where:
- **U** (Uniqueness): Semantic novelty vs. global corpus
- **C** (Cohesion): Fit within channel's conceptual lattice
- **L** (Learning): Forward progression, not repetition
- **Q** (Quality): De-biased engagement metrics
- **T** (Trust): Citation quality and correction rate

## Module Structure

```
idearank/
├── models.py              # Core data structures
│   ├── Embedding          # Semantic vector representation
│   ├── TopicMixture       # Topic distribution
│   ├── Video              # Single content item
│   └── Channel            # Collection of videos
│
├── config.py              # Configuration system
│   ├── IdeaRankConfig     # Master config
│   ├── FactorWeights      # U, C, L, Q, T weights
│   └── [Factor]Config     # Per-factor hyperparameters
│
├── factors/               # Independent factor modules
│   ├── base.py            # BaseFactor abstract class
│   ├── uniqueness.py      # U factor implementation
│   ├── cohesion.py        # C factor implementation
│   ├── learning.py        # L factor implementation
│   ├── quality.py         # Q factor implementation
│   └── trust.py           # T factor implementation
│
├── scorer.py              # Scoring engines
│   ├── IdeaRankScorer     # Video-level scoring
│   └── ChannelScorer      # Channel-level scoring
│
├── network.py             # Optional KnowledgeRank layer
│   └── KnowledgeRankComputer
│
├── pipeline.py            # End-to-end orchestration
│   └── IdeaRankPipeline
│
└── providers/             # Pluggable implementations
    ├── embeddings.py      # Embedding generation
    ├── topics.py          # Topic modeling
    └── neighborhoods.py   # ANN search
```

## Data Flow

### 1. Ingestion & Representation

```
Video (raw)
    ↓
EmbeddingProvider.embed(video.full_text)
    ↓
Embedding (vector)
    ↓
TopicModelProvider.get_topics(video.full_text)
    ↓
TopicMixture (distribution)
    ↓
NeighborhoodProvider.index_video(video)
```

### 2. Neighborhood Search

```
Video + Embedding
    ↓
NeighborhoodProvider.find_global_neighbors()  → For Uniqueness
    ↓
NeighborhoodProvider.find_intra_channel_neighbors()  → For Cohesion & Learning
```

### 3. Factor Computation

Each factor is computed independently:

```python
context = {
    'global_neighbors': [...],
    'intra_neighbors': [...],
    'prior_video': ...,
    'wtpi_distribution': {...},
    'cr_distribution': {...},
}

uniqueness = UniquenessFactor.compute(video, channel, context)
cohesion = CohesionFactor.compute(video, channel, context)
learning = LearningFactor.compute(video, channel, context)
quality = QualityFactor.compute(video, channel, context)
trust = TrustFactor.compute(video, channel, context)
```

Each returns a `FactorResult`:
```python
@dataclass
class FactorResult:
    score: float                    # Final [0, 1] value
    components: dict[str, float]    # Intermediate calculations
    metadata: dict[str, Any]        # Debug info
```

### 4. Video Scoring

```
IdeaRankScorer.score_video()
    ↓
Combine factors: U^w_U · C^w_C · L^w_L · Q^w_Q · T^w_T
    ↓
Check gates: U >= threshold_U and L >= threshold_L
    ↓
Return IdeaRankScore
```

### 5. Channel Scoring

```
ChannelScorer.score_channel()
    ↓
Get videos in time window (default: 180 days)
    ↓
Score each video → video_scores[video_id]
    ↓
Compute mean score
    ↓
Compute AUL (Area Under Learning)
    ↓
Check crystallization (low learning variance)
    ↓
IR_S = mean + η·AUL · (decay if crystallized)
    ↓
Return ChannelRankScore
```

### 6. Network Layer (Optional)

```
KnowledgeRankComputer.compute_knowledge_rank()
    ↓
For each channel pair (i, j):
    Compute IFR(i→j) based on semantic similarity with lag
    ↓
Build influence graph: adjacency list of IFR edges
    ↓
Power iteration:
    KR_j = (1-d)·IR_S_j + d·Σ_i KR_i·IFR(i→j)
    ↓
Converge to stable KR scores
    ↓
Return KnowledgeRankScore for each channel
```

## Configuration System

All hyperparameters live in `IdeaRankConfig`:

```python
config = IdeaRankConfig(
    weights=FactorWeights(uniqueness=0.35, cohesion=0.2, ...),
    uniqueness=UniquenessConfig(k_global=50, min_threshold=0.15),
    cohesion=CohesionConfig(k_intra=15, window_days=270),
    learning=LearningConfig(stability_sigma=0.5, target_step_size=(0.1, 0.4)),
    quality=QualityConfig(wtpi_weight=0.5, cr_weight=0.5),
    trust=TrustConfig(lambda1=0.4, lambda2=0.3, lambda3=0.3),
    channel=ChannelRankConfig(window_days=180, aul_bonus_weight=0.1),
    network=NetworkConfig(enabled=True, damping_factor=0.75),
)
```

### Presets

- `IdeaRankConfig.default()`: Production defaults
- `IdeaRankConfig.experimental()`: Faster for rapid iteration

## Provider Interface

Providers are **swappable implementations** of external services:

### Embedding Provider

```python
class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, text: str) -> Embedding:
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[Embedding]:
        pass
```

**Implementations:**
- `DummyEmbeddingProvider`: Random embeddings for testing
- `OpenAIEmbeddingProvider`: OpenAI API (stub)
- Custom: Hugging Face, Cohere, etc.

### Topic Model Provider

```python
class TopicModelProvider(ABC):
    @abstractmethod
    def get_topics(self, text: str) -> TopicMixture:
        pass
```

**Implementations:**
- `DummyTopicModelProvider`: Random distributions
- `LDATopicModelProvider`: Scikit-learn LDA (stub)
- Custom: BERTopic, NMF, etc.

### Neighborhood Provider

```python
class NeighborhoodProvider(ABC):
    @abstractmethod
    def find_global_neighbors(self, embedding, k) -> List[Tuple[Video, float]]:
        pass
    
    @abstractmethod
    def find_intra_channel_neighbors(self, embedding, channel_id, k) -> ...
        pass
```

**Implementations:**
- `DummyNeighborhoodProvider`: Brute-force search (O(n))
- `FAISSNeighborhoodProvider`: FAISS ANN (stub)
- Custom: Annoy, HNSW, etc.

## Pipeline Orchestration

`IdeaRankPipeline` ties everything together:

```python
pipeline = IdeaRankPipeline(
    config=config,
    embedding_provider=embedding_provider,
    topic_provider=topic_provider,
    neighborhood_provider=neighborhood_provider,
)

# End-to-end
kr_scores = pipeline.score_channels_with_network(channels)
```

**Steps:**
1. `process_videos_batch()`: Generate embeddings and topics
2. `index_videos()`: Build ANN indices
3. `score_video()`: Compute IR(v,t) for each video
4. `score_channel()`: Aggregate to IR_S(t)
5. `score_channels_with_network()`: Compute KR if enabled

## Extension Points

### Adding a New Factor

1. Create `idearank/factors/your_factor.py`
2. Extend `BaseFactor`
3. Implement `compute(video, channel, context) -> FactorResult`
4. Add to `IdeaRankScorer` combination formula
5. Add config class to `config.py`

### Adding a New Provider

1. Create `idearank/providers/your_service.py`
2. Extend appropriate provider interface
3. Implement required methods
4. Register in `providers/__init__.py`

### Customizing Weights

```python
# Runtime adjustment
config.weights.uniqueness = 0.5  # Emphasize novelty
config.weights.quality = 0.05    # De-emphasize engagement

# Or create preset
def journalism_config():
    return IdeaRankConfig(
        weights=FactorWeights(
            uniqueness=0.2,
            cohesion=0.15,
            learning=0.15,
            quality=0.1,
            trust=0.4,  # High trust for news
        )
    )
```

## Design Principles

1. **Modularity**: Each component is independent and testable
2. **Transparency**: Full score breakdowns with intermediate values
3. **Configurability**: Every parameter exposed and tunable
4. **Extensibility**: Clean interfaces for new factors/providers
5. **Reproducibility**: Deterministic given same config and data

## Performance Considerations

### Current Design (Clarity > Speed)

- Brute-force neighborhood search: O(n)
- Sequential factor computation
- No caching
- Python loops over videos

**Why?** This is a **research framework**, not production system.

### Production Optimizations (Future)

- FAISS/Annoy for ANN: O(log n)
- Parallel factor computation
- Redis/LRU caching for embeddings
- Batch processing with numpy/torch
- JIT compilation (Numba)
- Distributed scoring (Ray/Dask)

## Testing Strategy

- **Unit tests**: Each factor independently (`tests/test_*.py`)
- **Integration tests**: Pipeline end-to-end
- **Property tests**: Score ranges, monotonicity
- **Benchmark tests**: Performance regression detection

## Future Enhancements

- [ ] Temporal trending (how IR changes over time)
- [ ] Multi-modal embeddings (video + audio + visual)
- [ ] Active learning (which videos to label next)
- [ ] Explainability dashboard (why this score?)
- [ ] A/B testing framework (compare configs)
- [ ] Real-time updates (incremental scoring)
- [ ] Graph visualization (influence networks)

## Summary

IdeaRank is built like **LEGO blocks**: each piece works independently, snaps together cleanly, and can be swapped without breaking the whole system.

This makes it perfect for:
- Research: Test hypotheses about idea evolution
- Experimentation: Try different weight configurations
- Extension: Add domain-specific factors
- Debugging: Trace exactly why a video scored X

**It's not the fastest. It's not the simplest. But it's the most understandable and modifiable.**

Which is exactly what you need when you're trying to measure something as fuzzy as "idea quality."


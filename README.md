# IdeaRank

A multi-factor ranking system for content that measures **uniqueness**, **cohesion**, **learning progression**, **quality**, and **trust** — not just links or popularity.

**Think PageRank, but for ideas.**

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd idearank

# Install with all features
pip install -e ".[all]"
```

### Basic Setup

```bash
# Set up your API keys
idearank config set-youtube-key YOUR_YOUTUBE_API_KEY

# Optional: Configure Whisper transcription (local, no API key needed)
idearank config set-whisper-model small  # Options: tiny, base, small, medium, large
idearank config set-whisper-device auto  # Options: cpu, cuda, auto

# Optional: Set up OpenAI for better embeddings
export OPENAI_API_KEY=your_key_here
```

### Process Your Content

**Two ways to process content:**

#### Quick Single-Source Processing
Process one source directly (doesn't save to sources config):

```bash
# YouTube (limited to 50 videos)
idearank process https://youtube.com/@yourchannel --max-videos 50

# YouTube (ALL videos - full pagination)
idearank process https://youtube.com/@yourchannel --all

# Ghost export (limited)
idearank process-ghost-export ~/Downloads/blog.ghost.json --max-posts 50

# Ghost export (ALL posts)
idearank process-ghost-export ~/Downloads/blog.ghost.json --all

# Medium archive (limited)
idearank process-medium ~/Downloads/medium-export.zip --max-posts 50

# Medium archive (ALL posts)
idearank process-medium ~/Downloads/medium-export.zip --all

# Each creates its own database (idearank_results.db by default)
```

#### Multi-Source Workflow
Add multiple sources, then process all together:

```bash
# Add sources with limits (saved to ~/.idearank/sources.json)
idearank source add https://youtube.com/@channel --max-items 50
idearank source add ~/Downloads/blog.ghost.json --max-items 100

# Add sources with unlimited processing (ALL content)
idearank source add https://youtube.com/@bigchannel --all
idearank source add ~/Downloads/medium-export.zip --all
idearank source add ~/Downloads/twitter-archive.json --all

# View your sources
idearank source list

# Process all enabled sources into one database
idearank process-all

# Creates: idearank_all_content.db with all sources
```

**When to use each:**
- **Single-source:** Quick analysis, one-off checks, testing
- **Multi-source:** Cross-source comparison, comprehensive analysis, tracking over time

**Pagination & Limits:**
- By default, commands process 50 items (videos/posts)
- Use `--all` flag to process **ALL** content with automatic pagination
- Or set custom limits: `--max-videos 200`, `--max-posts 100`, etc.
- For multi-source: use `--all` when adding sources for unlimited processing

### Visualize Results

#### Interactive Web Dashboard (Recommended)
```bash
# Install Streamlit dependencies
pip install "idearank[streamlit]"

# Launch interactive dashboard
streamlit run streamlit_app.py

# Opens in your browser with:
# - Overview metrics and charts
# - Top content rankings
# - Factor analysis and correlations
# - Content explorer with search
# - Source comparison
# - Trends over time
```

#### Command-Line Charts
```bash
# Create complete dashboard
idearank viz dashboard

# Individual plots
idearank viz timeline              # Scores over time
idearank viz factors              # Factor breakdown
idearank viz scatter              # Uniqueness vs Cohesion
idearank viz learning --source-id SOURCE_ID  # Learning trajectory
```

## Core Concepts

### The Six Factors

IdeaRank scores content using six factors:

1. **Uniqueness (U)** - How novel is this content vs. everything else?
2. **Cohesion (C)** - Does it fit the source's theme or is it random?
3. **Learning (L)** - Is the source advancing ideas, not repeating?
4. **Quality (Q)** - Genuine engagement (not just clickbait)
5. **Trust (T)** - Are claims grounded with citations?
6. **Density (D)** - Information density and audience fit (human/AI)

**Final Score:** `IR = U^w_U · C^w_C · L^w_L · Q^w_Q · T^w_T · D^w_D`

### Content Sources

IdeaRank supports multiple content types:
- **YouTube** - Videos with transcripts
- **Ghost Blogs** - Blog posts (via export or API)
- **Medium** - Blog archives (ZIP export)
- **Twitter** - Tweet archives (local JSON file)
- **More coming** - Easy to add new sources

## Python API

### Basic Usage

```python
from idearank import IdeaRankConfig, ContentItem, ContentSource
from idearank.pipeline import IdeaRankPipeline
from idearank.providers import (
    SentenceTransformerEmbeddingProvider,
    LDATopicModelProvider,
    DummyNeighborhoodProvider,
)

# Create content items
items = [
    ContentItem(
        id="item_1",
        content_source_id="my_blog",
        title="My First Post",
        description="Introduction to my ideas",
        body="Full content of the post...",
        published_at=datetime(2024, 1, 1),
        captured_at=datetime.now(),
        view_count=1000,
        impression_count=5000,
        watch_time_seconds=50000.0,
        avg_view_duration=200.0,
        content_duration=300.0,
    ),
    # ... more items
]

# Create content source
source = ContentSource(
    id="my_blog",
    name="My Blog",
    description="Personal blog about ideas",
    created_at=datetime(2023, 1, 1),
    content_items=items,
)

# Set up pipeline
config = IdeaRankConfig.default()
pipeline = IdeaRankPipeline(
    config=config,
    embedding_provider=SentenceTransformerEmbeddingProvider(),
    topic_provider=LDATopicModelProvider(),
    neighborhood_provider=DummyNeighborhoodProvider(),
)

# Process and score
pipeline.process_content_batch(items)
pipeline.index_content(items)

for item in items:
    score = pipeline.score_content_item(item, source)
    print(f"{item.title}: {score.score:.4f}")
```

### Using Chroma Cloud

```python
from idearank.providers.chroma import ChromaProvider

# Initialize Chroma (handles embeddings + vector search)
chroma = ChromaProvider(
    chroma_cloud_api_key="your-api-key",
    chroma_cloud_tenant="your-tenant-id",
    chroma_cloud_database="your-database",
    collection_name="my_content",
)

# Use in pipeline
pipeline = IdeaRankPipeline(
    config=config,
    embedding_provider=chroma.get_embedding_provider(),
    topic_provider=LDATopicModelProvider(),
    neighborhood_provider=chroma.get_neighborhood_provider(),
)
```

## CLI Reference

### Configuration
```bash
idearank config show                          # Show current config
idearank config set-youtube-key KEY          # Set YouTube API key
idearank config set-whisper-model MODEL      # Set Whisper model (tiny/base/small/medium/large)
idearank config set-whisper-device DEVICE    # Set Whisper device (cpu/cuda/auto)
idearank config set-chroma-mode local|cloud  # Choose Chroma mode
idearank config clear                         # Reset configuration
```

### Source Management
```bash
idearank source add URL_OR_PATH [TYPE]   # Add a content source
idearank source list                      # List all sources
idearank source enable NAME               # Enable a source
idearank source disable NAME              # Disable a source
idearank source remove NAME               # Remove a source
idearank source clear                     # Remove all sources
```

### Processing
```bash
idearank process CHANNEL_URL              # Process single YouTube channel
idearank process-ghost-export FILE       # Process Ghost export
idearank process-medium ARCHIVE.zip      # Process Medium archive
idearank process-all                     # Process all enabled sources
```

### Visualization
```bash
idearank viz dashboard                    # Create full dashboard
idearank viz timeline                     # Scores over time
idearank viz factors                      # Factor breakdown over time
idearank viz learning --source-id ID      # Learning trajectory for source
idearank viz scatter                      # Uniqueness vs Cohesion scatter
```

## Configuration

### Custom Weights

Adjust the factor weights to emphasize different aspects:

```python
from idearank.config import IdeaRankConfig, FactorWeights

config = IdeaRankConfig(
    weights=FactorWeights(
        uniqueness=0.30,  # Default: emphasize novelty
        cohesion=0.18,    # Topical consistency
        learning=0.22,    # Progression over time
        quality=0.13,     # Engagement quality
        trust=0.05,       # Citation quality
        density=0.12,     # Information density
    )
)
```

### Factor-Specific Settings

```python
# Uniqueness: More global neighbors for comparison
config.uniqueness.k_global = 100
config.uniqueness.min_threshold = 0.15

# Cohesion: Tighter intra-source comparison
config.cohesion.k_intra = 20
config.cohesion.window_days = 180

# Learning: Target step size
config.learning.target_step_size = (0.1, 0.4)
config.learning.min_threshold = 0.05

# Quality: Metric weights
config.quality.wtpi_weight = 0.5
config.quality.cr_weight = 0.5

# Trust: Citation analysis
config.trust.lambda1 = 0.4  # Citations
config.trust.lambda2 = 0.3  # Source diversity
config.trust.lambda3 = 0.3  # Corrections

# Density: Audience optimization
config.density.audience = "human"  # or "ai" or "balanced" (default)
config.density.citation_density_weight = 0.25
config.density.concept_diversity_weight = 0.25
config.density.information_efficiency_weight = 0.25
config.density.explicitness_weight = 0.25
```

## Advanced Features

### Audience Optimization (Density Factor)

The Density factor adapts to your target audience:

**Human-optimized:**
```python
config.density.audience = "human"
# Favors: Dense, punchy content that assumes context
# Best for: Blogs, essays, Twitter threads
```

**AI-optimized:**
```python
config.density.audience = "ai"
# Favors: Explicit, self-contained content with full context
# Best for: RAG systems, AI training data, documentation
```

**Balanced (default):**
```python
config.density.audience = "balanced"
# Optimizes for both human and AI readers
# Best for: General content, mixed audiences
```

**Why it matters:**
- Humans prefer concise, contextual content (high density, low explicitness)
- AIs prefer comprehensive, explicit content (moderate density, high explicitness)
- The Density factor scores content based on your target audience

### Entity-Idea Citation Parsing

IdeaRank can parse references to people and institutions:

```python
from idearank.citation_parser import analyze_citations

analysis = analyze_citations(
    text="John Smith argues that markets are efficient...",
    use_spacy=True,
    validate=True,  # Optional: AI validation
    openai_api_key="your-key",
)

print(f"Found {analysis.total_attributions} entity-idea connections")
print(f"Unique entities: {analysis.unique_entities}")
print(f"Validation accuracy: {analysis.validation_accuracy}")
```

### Network Layer (KnowledgeRank)

Measure cross-source influence:

```python
config.network.enabled = True
config.network.damping_factor = 0.75
config.network.influence_threshold = 0.2

kr_scores = pipeline.score_sources_with_network(sources)

for source_id, kr_score in kr_scores.items():
    print(f"{source_id}: KR={kr_score.knowledge_rank:.4f}")
    print(f"  Influenced by: {len(kr_score.incoming_influence)} sources")
    print(f"  Influences: {len(kr_score.outgoing_influence)} sources")
```

### Diagnostics

Check data quality:

```bash
# Run diagnostics on your database
python -c "from idearank.diagnostics import IdeaRankDiagnostics; \
           diag = IdeaRankDiagnostics('idearank_all_content.db'); \
           diag.print_report()"
```

## Data Model

### ContentItem
Represents any piece of content (video, blog post, tweet):
- `id` - Unique identifier
- `content_source_id` - Parent source ID
- `title`, `description`, `body` - Content text
- `published_at`, `captured_at` - Timestamps
- `embedding`, `topic_mixture` - Computed representations
- Analytics: `view_count`, `impression_count`, `watch_time_seconds`, etc.
- Trust signals: `has_citations`, `citation_count`, `source_diversity_score`

### ContentSource
Represents a content producer (channel, blog, Twitter account):
- `id`, `name`, `description` - Identity
- `content_items` - List of content
- `subscriber_count`, `total_views` - Metrics
- Methods: `get_content_in_window()`, `get_prior_content()`

## Database Schema

IdeaRank uses SQLite with three main tables:
- `content_items` - All content with metadata
- `content_sources` - Source metadata
- `idearank_scores` - Computed scores with factor breakdowns
- `content_source_rank_scores` - Aggregate source scores

Query example:
```sql
SELECT 
    c.title, 
    c.published_at,
    s.score,
    s.uniqueness_score,
    s.cohesion_score,
    s.learning_score
FROM content_items c
JOIN idearank_scores s ON c.id = s.content_item_id
WHERE c.content_source_id = 'your_source'
ORDER BY s.score DESC
LIMIT 10;
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Content Sources                        │
│  (YouTube, Ghost, Twitter, Custom)                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              IdeaRank Pipeline                          │
│  1. Fetch/Parse Content                                 │
│  2. Generate Embeddings (sentence-transformers/OpenAI)  │
│  3. Extract Topics (LDA)                                │
│  4. Index Vectors (Chroma/Local)                        │
│  5. Compute Factors (U,C,L,Q,T)                        │
│  6. Calculate Scores                                    │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  Storage & Analysis                      │
│  • SQLite Database (content + scores)                   │
│  • Visualizations (matplotlib)                          │
│  • Diagnostics (data quality checks)                    │
│  • Network Layer (cross-source influence)               │
└─────────────────────────────────────────────────────────┘
```

## Requirements

**Core:**
- Python 3.9+
- sentence-transformers (default embeddings)
- scikit-learn (LDA topic modeling)
- chromadb (vector database)
- sqlalchemy (data persistence)

**Optional:**
- openai (better embeddings)
- spacy + language model (enhanced citation parsing - see [SPACY_SETUP.md](SPACY_SETUP.md))
- matplotlib, pandas (visualizations)

**External APIs:**
- YouTube Data API v3 (for YouTube content)
- Chroma Cloud (optional, for cloud vector storage)

**Local Services:**
- faster-whisper (GPU-accelerated local transcription, no API key needed)
- yt-dlp (for audio download and subtitle extraction)

## Transcription

IdeaRank uses a **three-tier transcription strategy** for YouTube videos:

1. **YouTube Transcript API** (free, instant) - tries built-in captions first
2. **yt-dlp subtitle extraction** (free, instant) - downloads native subtitles if available
3. **faster-whisper** (local, GPU-accelerated) - transcribes audio as fallback

### Configuration

```bash
# Set Whisper model (trade-off between speed and quality)
idearank config set-whisper-model small  # Default, good balance

# Model options (from fastest to best):
# - tiny:   ~1GB RAM, fastest
# - base:   ~1GB RAM, fast
# - small:  ~2GB RAM, balanced (default)
# - medium: ~5GB RAM, good quality
# - large:  ~10GB RAM, best quality

# Set device (auto uses GPU if available)
idearank config set-whisper-device auto  # Default
idearank config set-whisper-device cuda  # Force GPU
idearank config set-whisper-device cpu   # Force CPU
```

### Requirements

```bash
# Install ffmpeg (required for audio extraction)
# macOS:
brew install ffmpeg

# Ubuntu/Debian:
sudo apt-get install ffmpeg

# Windows:
# Download from https://ffmpeg.org/download.html
```

The transcription pipeline **automatically caches** all transcripts in SQLite, so videos are never transcribed twice.

## Streamlit Dashboard

For interactive data exploration, use the Streamlit web interface:

```bash
# Install Streamlit extras
pip install "idearank[streamlit]"

# Launch dashboard
streamlit run streamlit_app.py
```

**Features:**
- 📈 Overview metrics and distribution charts
- 🎯 Top content rankings with factor breakdowns
- 📊 Factor correlation analysis
- 🔍 Interactive content explorer with search
- 🏆 Multi-source comparison
- 📉 Trends and time-series analysis

The dashboard automatically finds all `.db` files in the current directory and lets you explore them interactively in your browser.

## Development

### Running Examples

```bash
# Basic usage with dummy providers
python examples/basic_usage.py

# Test with mock YouTube data
python examples/youtube_pipeline_simple_test.py

# Chroma Cloud integration
python examples/chroma_usage.py

# Custom weights comparison
python examples/custom_weights.py
```

### Running Tests

```bash
pytest tests/
```

## Troubleshooting

### Flat Cohesion Scores
If all cohesion scores are identical, the topic model may not be fitted. IdeaRank uses LDA which needs to be trained on your corpus. This happens automatically in `process_content_batch()`.

### Flat Trust Scores
If trust scores don't vary, ensure you're using the citation parser. It's enabled by default but requires content with references to people/institutions.

### Missing Embeddings
IdeaRank defaults to `sentence-transformers`. Install with:
```bash
pip install sentence-transformers
```

Or use OpenAI:
```bash
idearank process-all --openai-key YOUR_KEY
```

### Database Migration
After the v2.0 refactor, table names changed:
- `videos` → `content_items`
- `channels` → `content_sources`
- `channel_id` → `content_source_id`

You'll need to re-run processing to create new databases.

## Version

**Current Version:** 2.0.0

**Breaking Changes in v2.0:**
- Renamed `Video` → `ContentItem`
- Renamed `Channel` → `ContentSource`
- Updated all database table names
- Updated all API method names

## License

[Add your license here]

## Contributing

This project is in active development. The architecture is designed to be extensible - adding new content sources is straightforward.

## Citation

If you use IdeaRank in your research or product, please cite:
```
[Add citation here]
```

# Using IdeaRank with Chroma Cloud

This guide shows how to use **Chroma Cloud** as your vector database backend for IdeaRank.

## Why Chroma?

Chroma provides:
- **Embedding generation** via built-in functions (OpenAI, sentence-transformers, etc.)
- **Vector storage** with automatic persistence
- **Fast ANN search** using HNSW index
- **Cloud hosting** so you don't manage infrastructure
- **Multi-session persistence** - embeddings survive across runs

## Installation

```bash
# Install IdeaRank with Chroma support
pip install -e ".[chroma]"

# Or just install chromadb
pip install chromadb
```

## Getting Chroma Cloud Credentials

1. Sign up at [Chroma Cloud](https://www.trychroma.com/)
2. Create a tenant and database
3. Get your API key, tenant ID, and database name

## Basic Usage

### Option 1: Combined Provider (Recommended)

Use `ChromaProvider` for both embeddings and search:

```python
from idearank.providers.chroma import ChromaProvider

# Initialize combined provider
chroma = ChromaProvider(
    api_key="ck-YOUR_API_KEY",
    tenant="YOUR_TENANT_ID",
    database="YOUR_DATABASE_NAME",
    embedding_function="default",  # or "openai", "sentence-transformers"
    collection_name="idearank_videos",
)

# Get individual providers
embedding_provider = chroma.get_embedding_provider()
neighborhood_provider = chroma.get_neighborhood_provider()
```

### Option 2: Separate Providers

Use embedding and neighborhood providers independently:

```python
from idearank.providers.chroma import (
    ChromaEmbeddingProvider,
    ChromaNeighborhoodProvider,
)

# Embedding generation
embedding_provider = ChromaEmbeddingProvider(
    api_key="ck-YOUR_API_KEY",
    tenant="YOUR_TENANT_ID",
    database="YOUR_DATABASE_NAME",
    embedding_function="default",
)

# Vector search
neighborhood_provider = ChromaNeighborhoodProvider(
    api_key="ck-YOUR_API_KEY",
    tenant="YOUR_TENANT_ID",
    database="YOUR_DATABASE_NAME",
    collection_name="idearank_videos",
)
```

## Embedding Functions

Chroma supports multiple embedding backends:

### 1. Default Embedding Function

```python
chroma = ChromaProvider(
    api_key="...",
    tenant="...",
    database="...",
    embedding_function="default",  # Uses Chroma's default (all-MiniLM-L6-v2)
)
```

**Pros:** No API keys needed, runs locally  
**Cons:** Lower quality than specialized models

### 2. OpenAI Embeddings

```python
# Set OpenAI API key in environment
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

chroma = ChromaProvider(
    api_key="...",
    tenant="...",
    database="...",
    embedding_function="openai",
    model_name="text-embedding-3-small",  # or "text-embedding-3-large"
)
```

**Pros:** High quality, good for production  
**Cons:** Costs money per embedding

### 3. Sentence Transformers

```python
chroma = ChromaProvider(
    api_key="...",
    tenant="...",
    database="...",
    embedding_function="sentence-transformers",
    model_name="all-MiniLM-L6-v2",  # or any HF model
)
```

**Pros:** Free, runs locally, many model options  
**Cons:** Slower than API calls

## Full Pipeline Example

```python
from idearank import IdeaRankConfig, Video, Channel
from idearank.pipeline import IdeaRankPipeline
from idearank.providers import DummyTopicModelProvider
from idearank.providers.chroma import ChromaProvider

# 1. Initialize Chroma
chroma = ChromaProvider(
    api_key="ck-BojTG2QscadMvcrtFX9cPrmbUKHwGJ9VKYrvq1Noa5LG",
    tenant="e59b3318-066b-4aa2-886a-c21fd8f81ef0",
    database="Idea Nexus Ventures",
    embedding_function="default",
    collection_name="idearank_videos",
)

# 2. Create pipeline
config = IdeaRankConfig.default()
pipeline = IdeaRankPipeline(
    config=config,
    embedding_provider=chroma.get_embedding_provider(),
    topic_provider=DummyTopicModelProvider(),  # You can add topic support to Chroma too
    neighborhood_provider=chroma.get_neighborhood_provider(),
)

# 3. Process videos
videos = [...]  # Your Video objects
pipeline.process_videos_batch(videos)  # Generates embeddings
pipeline.index_videos(videos)  # Stores in Chroma Cloud

# 4. Score
channel = Channel(...)
score = pipeline.score_video(videos[0], channel)

print(f"IdeaRank: {score.score:.4f}")
```

## Environment Variables

For security, use environment variables:

```bash
export CHROMA_API_KEY="ck-..."
export CHROMA_TENANT="..."
export CHROMA_DATABASE="..."
```

```python
import os

chroma = ChromaProvider(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
)
```

## Persistence

Embeddings in Chroma Cloud are **automatically persisted**:

```python
# Session 1: Index videos
pipeline.index_videos(videos)

# ... later, Session 2: Videos are still there
neighbors = neighborhood_provider.find_global_neighbors(embedding, k=50)
# Works! Retrieves previously indexed videos
```

## Managing Collections

```python
from idearank.providers.chroma import ChromaNeighborhoodProvider

provider = ChromaNeighborhoodProvider(...)

# Clear all videos
provider.clear_collection()

# Collection is automatically created if it doesn't exist
provider.index_video(video)
```

## Distance to Similarity Conversion

Chroma returns **L2 distances**. IdeaRank converts to similarities:

```
similarity = 1 / (1 + distance)
```

Where:
- `distance = 0` → `similarity = 1.0` (identical)
- `distance = 1` → `similarity = 0.5`
- `distance = ∞` → `similarity = 0.0` (completely different)

## Performance Tips

### 1. Batch Processing

```python
# Good: Process all videos at once
pipeline.process_videos_batch(all_videos)
pipeline.index_videos(all_videos)

# Bad: One at a time
for video in all_videos:
    pipeline.process_video(video)
    provider.index_video(video)
```

### 2. Filtering by Channel

Chroma supports metadata filtering:

```python
# Automatically filters by channel_id
intra_neighbors = provider.find_intra_channel_neighbors(
    embedding=video.embedding,
    channel_id="my_channel",
    k=15,
)
```

### 3. Caching

The `ChromaNeighborhoodProvider` caches `Video` objects in memory:

```python
# Videos are cached after indexing
provider.index_video(video)

# Later retrieval returns the same Video object
neighbors = provider.find_global_neighbors(...)
# neighbors[0][0] is the cached Video instance
```

## Common Issues

### Issue: "Collection not found"

**Solution:** Collections are auto-created on first `index_video()` call:

```python
provider.index_video(video)  # Creates collection if needed
```

### Issue: "No videos with embeddings to index"

**Solution:** Generate embeddings first:

```python
pipeline.process_videos_batch(videos)  # FIRST: Generate embeddings
pipeline.index_videos(videos)  # THEN: Index them
```

### Issue: Slow embedding generation

**Solution:** Use batch processing and OpenAI embeddings:

```python
# Fast: Batch + API
chroma = ChromaProvider(
    embedding_function="openai",
    model_name="text-embedding-3-small",
)
pipeline.process_videos_batch(videos)  # Batch call to OpenAI

# Slow: One-by-one + local model
for video in videos:
    pipeline.process_video(video)  # Sequential local processing
```

## Advanced: Custom Embedding Functions

You can provide custom Chroma embedding functions:

```python
import chromadb

# Create custom function
custom_ef = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Use it
provider = ChromaNeighborhoodProvider(
    api_key="...",
    tenant="...",
    database="...",
    embedding_function=custom_ef,  # Custom!
)
```

## Example Script

See `examples/chroma_usage.py` for a complete working example:

```bash
python examples/chroma_usage.py
```

## Cost Estimation

### OpenAI Embeddings

- `text-embedding-3-small`: $0.02 / 1M tokens
- Average video transcript: ~5,000 tokens
- 1,000 videos: ~$0.10

### Chroma Cloud

Check [Chroma pricing](https://www.trychroma.com/pricing) for storage costs.

## Comparison: Chroma vs Dummy Provider

| Feature | DummyNeighborhoodProvider | ChromaNeighborhoodProvider |
|---------|---------------------------|----------------------------|
| Speed | O(n) brute force | O(log n) HNSW |
| Persistence | None | Cloud storage |
| Scalability | <1k videos | Millions of videos |
| Setup | Zero config | API key needed |
| Cost | Free | Paid (after free tier) |

## Next Steps

1. Sign up for Chroma Cloud
2. Run `examples/chroma_usage.py`
3. Replace dummy providers in your code
4. Index your real video corpus
5. Enjoy fast, persistent vector search!

---

**Ready to scale beyond dummy providers? Chroma's got your back.** 🚀


# 🚀 Quick Start: Your IdeaRank + Chroma Setup

Follow these steps to get IdeaRank running with Chroma Cloud in **5 minutes**.

---

## Step 1: Install Dependencies

```bash
cd /Users/leoguinan/idearank
pip install -e ".[chroma]"
```

✅ **This installs IdeaRank + ChromaDB**

---

## Step 2: Set Up Credentials (Optional)

Your Chroma credentials are already in the example code, but for security:

```bash
# Copy the example env file
cp .env.example .env

# Edit .env with your favorite editor
# (Credentials are already filled in from your original message)
```

---

## Step 3: Test Basic Setup

```bash
# Run the basic example (no Chroma needed)
python examples/basic_usage.py
```

**Expected output:**
```
============================================================
IdeaRank Basic Usage Example
============================================================
...
Overall IdeaRank Score: 0.XXXX
✓ Example completed successfully!
```

---

## Step 4: Test Chroma Integration

```bash
# Run the Chroma Cloud example
python examples/chroma_usage.py
```

**This will:**
1. Connect to your Chroma Cloud database: "Idea Nexus Ventures"
2. Generate embeddings for sample videos
3. Store them in Chroma Cloud
4. Run ANN search to find similar videos
5. Compute IdeaRank scores using real vector search

**Expected output:**
```
======================================================================
IdeaRank with Chroma Cloud Example
======================================================================

1. Connecting to Chroma Cloud...
   Tenant: e59b3318-066b-4aa2-886a-c21fd8f81ef0
   Database: Idea Nexus Ventures
   ✓ Connected successfully!

...

Overall IdeaRank Score: 0.XXXX
  - Uniqueness (U):  0.XXXX
  - Cohesion (C):    0.XXXX
  - Learning (L):    0.XXXX
  - Quality (Q):     0.XXXX
  - Trust (T):       0.XXXX

✓ Example completed successfully!
```

---

## Step 5: Run Your Own Data

Create a new file `my_analysis.py`:

```python
from datetime import datetime
from idearank import IdeaRankConfig, Video, Channel
from idearank.pipeline import IdeaRankPipeline
from idearank.providers import DummyTopicModelProvider
from idearank.providers.chroma import ChromaProvider

# 1. Connect to Chroma
chroma = ChromaProvider(
    api_key="ck-BojTG2QscadMvcrtFX9cPrmbUKHwGJ9VKYrvq1Noa5LG",
    tenant="e59b3318-066b-4aa2-886a-c21fd8f81ef0",
    database="Idea Nexus Ventures",
    embedding_function="default",  # Free, no API key needed
    collection_name="my_videos",   # Your collection name
)

# 2. Create pipeline
pipeline = IdeaRankPipeline(
    config=IdeaRankConfig.default(),
    embedding_provider=chroma.get_embedding_provider(),
    topic_provider=DummyTopicModelProvider(),
    neighborhood_provider=chroma.get_neighborhood_provider(),
)

# 3. Create your video
my_video = Video(
    id="my_first_video",
    channel_id="my_channel",
    title="My Amazing Video Title",
    description="A description of what this video is about...",
    transcript="The full transcript of the video content...",
    published_at=datetime(2024, 1, 1),
    snapshot_time=datetime.now(),
    # Analytics (if you have them)
    view_count=1000,
    impression_count=5000,
    watch_time_seconds=50000.0,
    avg_view_duration=200.0,
    video_duration=300.0,
    # Trust signals
    has_citations=True,
    citation_count=5,
    source_diversity_score=0.8,
)

# 4. Create channel
my_channel = Channel(
    id="my_channel",
    name="My Channel",
    description="My channel description",
    created_at=datetime(2023, 1, 1),
    videos=[my_video],
)

# 5. Process and score
print("Processing video...")
pipeline.process_videos_batch([my_video])
pipeline.index_videos([my_video])

print("Computing IdeaRank score...")
score = pipeline.score_video(my_video, my_channel)

# 6. View results
print(f"\n{'='*60}")
print(f"IdeaRank Score: {score.score:.4f}")
print(f"{'='*60}")
print(f"  Uniqueness:  {score.uniqueness.score:.4f}")
print(f"  Cohesion:    {score.cohesion.score:.4f}")
print(f"  Learning:    {score.learning.score:.4f}")
print(f"  Quality:     {score.quality.score:.4f}")
print(f"  Trust:       {score.trust.score:.4f}")
print(f"\nPasses Quality Gates: {score.passes_gates}")
```

Run it:
```bash
python my_analysis.py
```

---

## What Just Happened?

1. ✅ **Embeddings generated**: Chroma converted your text to vectors
2. ✅ **Stored in cloud**: Vectors saved in "Idea Nexus Ventures" database
3. ✅ **ANN search**: Found similar videos using HNSW index
4. ✅ **IdeaRank computed**: All 5 factors scored and combined
5. ✅ **Persistent**: Your embeddings are saved for future sessions

---

## Next Steps

### Read the Docs
- `README.md` - Complete guide
- `CHROMA_SETUP.md` - Chroma Cloud details
- `ARCHITECTURE.md` - System design
- `SUMMARY.md` - What we built

### Try Different Configurations
```bash
python examples/custom_weights.py
```

### Check Your Data in Chroma
Your videos are stored in Chroma Cloud at:
- **Tenant**: e59b3318-066b-4aa2-886a-c21fd8f81ef0
- **Database**: Idea Nexus Ventures
- **Collection**: idearank_demo (or whatever you named it)

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'chromadb'"
```bash
pip install -e ".[chroma]"
```

### "Connection refused" or "Authentication failed"
Check your credentials in `.env` or the code

### "No videos with embeddings to index"
Make sure to call `process_videos_batch()` BEFORE `index_videos()`

### Want to use OpenAI embeddings instead?
```python
chroma = ChromaProvider(
    api_key="...",
    tenant="...",
    database="...",
    embedding_function="openai",
    model_name="text-embedding-3-small",
)
```

Then set: `export OPENAI_API_KEY="sk-..."`

---

## Need Help?

1. Check `CHROMA_SETUP.md` for detailed Chroma guide
2. Run examples to see working code
3. Read error messages - they're descriptive
4. Check test files for usage patterns

---

**You're ready to rank ideas! 🚀**

```bash
python examples/chroma_usage.py
```


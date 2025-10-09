"""Demo: YouTube channel → IdeaRank scores pipeline.

This example shows the complete pipeline:
1. Fetch YouTube channel data
2. Get transcripts (or transcribe with Gladia)
3. Generate embeddings with Chroma
4. Index in Chroma Cloud
5. Compute IdeaRank scores
6. Save everything to SQLite

Test channel: youtube.com/@ideasupplychain
"""

import os
import logging
from dotenv import load_dotenv

from idearank import IdeaRankConfig
from idearank.pipeline import IdeaRankPipeline
from idearank.providers import DummyTopicModelProvider
from idearank.providers.chroma import ChromaProvider
from idearank.integrations.youtube import YouTubeClient
from idearank.integrations.storage import SQLiteStorage
from idearank.pipelines import YouTubePipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def main():
    """Run the YouTube → IdeaRank pipeline."""
    
    print("=" * 80)
    print("YouTube → IdeaRank Pipeline Demo")
    print("=" * 80)
    
    # Configuration
    CHANNEL_URL = "youtube.com/@ideasupplychain"
    MAX_VIDEOS = 3  # Start with just 3 videos
    
    print(f"\nTarget channel: {CHANNEL_URL}")
    print(f"Max videos: {MAX_VIDEOS}")
    
    # ===================================================================
    # 1. Load credentials
    # ===================================================================
    print("\n[Setup] Loading credentials...")
    
    # Chroma Cloud
    chroma_api_key = os.getenv("CHROMA_API_KEY", "ck-BojTG2QscadMvcrtFX9cPrmbUKHwGJ9VKYrvq1Noa5LG")
    chroma_tenant = os.getenv("CHROMA_TENANT", "e59b3318-066b-4aa2-886a-c21fd8f81ef0")
    chroma_database = os.getenv("CHROMA_DATABASE", "Idea Nexus Ventures")
    
    # YouTube Data API (optional but recommended)
    youtube_api_key = os.getenv("YOUTUBE_API_KEY")
    if not youtube_api_key:
        print("⚠️  No YOUTUBE_API_KEY found. Will use limited data.")
        print("   Get API key from: https://console.cloud.google.com/apis/credentials")
    
    # Gladia API (for transcription fallback)
    gladia_api_key = os.getenv("GLADIA_API_KEY")
    if not gladia_api_key:
        print("⚠️  No GLADIA_API_KEY found. Will only use YouTube transcripts.")
    
    # ===================================================================
    # 2. Initialize components
    # ===================================================================
    print("\n[Setup] Initializing components...")
    
    # Chroma provider (embeddings + vector search)
    print("  - Connecting to Chroma Cloud...")
    chroma = ChromaProvider(
        api_key=chroma_api_key,
        tenant=chroma_tenant,
        database=chroma_database,
        embedding_function="default",  # Free, no API key needed
        collection_name="youtube_idearank_demo",
    )
    print("  ✓ Chroma connected")
    
    # YouTube client
    print("  - Initializing YouTube client...")
    youtube = YouTubeClient(
        youtube_api_key=youtube_api_key,
        gladia_api_key=gladia_api_key,
    )
    print("  ✓ YouTube client ready")
    
    # SQLite storage
    print("  - Setting up SQLite database...")
    storage = SQLiteStorage(db_path="idearank_youtube_demo.db")
    print("  ✓ Database ready")
    
    # IdeaRank pipeline
    print("  - Creating IdeaRank pipeline...")
    config = IdeaRankConfig.default()
    config.uniqueness.k_global = 10  # Fewer neighbors for small dataset
    config.cohesion.k_intra = 3
    
    idearank_pipeline = IdeaRankPipeline(
        config=config,
        embedding_provider=chroma.get_embedding_provider(),
        topic_provider=DummyTopicModelProvider(),
        neighborhood_provider=chroma.get_neighborhood_provider(),
    )
    print("  ✓ IdeaRank pipeline ready")
    
    # YouTube → IdeaRank pipeline
    print("  - Assembling full pipeline...")
    pipeline = YouTubePipeline(
        idearank_pipeline=idearank_pipeline,
        youtube_client=youtube,
        storage=storage,
        config=config,
    )
    print("  ✓ Pipeline assembled")
    
    # ===================================================================
    # 3. Run the pipeline
    # ===================================================================
    print("\n" + "=" * 80)
    print("Running pipeline on:", CHANNEL_URL)
    print("=" * 80)
    
    try:
        channel, scores = pipeline.process_channel(
            channel_url=CHANNEL_URL,
            max_videos=MAX_VIDEOS,
        )
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ===================================================================
    # 4. Display results
    # ===================================================================
    pipeline.print_summary(channel, scores)
    
    # ===================================================================
    # 5. Show what was saved
    # ===================================================================
    print("\n" + "=" * 80)
    print("Saved Data")
    print("=" * 80)
    
    print(f"\n📊 SQLite Database: idearank_youtube_demo.db")
    print(f"   - Videos: {len(channel.videos)}")
    print(f"   - Channels: 1")
    print(f"   - IdeaRank scores: {len(scores)}")
    
    print(f"\n🔍 Chroma Cloud Collection: youtube_idearank_demo")
    print(f"   - Tenant: {chroma_tenant}")
    print(f"   - Database: {chroma_database}")
    print(f"   - Embeddings: {len(channel.videos)}")
    
    print("\n" + "=" * 80)
    print("Pipeline complete! 🎉")
    print("=" * 80)
    
    print("\nNext steps:")
    print("1. Check SQLite database: sqlite3 idearank_youtube_demo.db")
    print("2. Query scores: SELECT * FROM idearank_scores;")
    print("3. Increase MAX_VIDEOS to process more content")
    print("4. Try different channels")
    
    # Clean up
    storage.close()


if __name__ == "__main__":
    main()


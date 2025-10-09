"""Chroma Cloud provider for embeddings and neighborhood search.

ChromaDB provides both:
1. Embedding generation (via built-in embedding functions)
2. Vector storage and ANN search (via collections)

This module implements both EmbeddingProvider and NeighborhoodProvider
using Chroma Cloud.
"""

from typing import List, Tuple, Optional, Dict, Any
import logging

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    chromadb = None

import numpy as np

from idearank.models import Video, Embedding, TopicMixture
from idearank.providers.embeddings import EmbeddingProvider
from idearank.providers.neighborhoods import NeighborhoodProvider

logger = logging.getLogger(__name__)


class ChromaEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using Chroma's built-in embedding functions.
    
    Supports various embedding models through Chroma's function system.
    """
    
    def __init__(
        self,
        api_key: str,
        tenant: str,
        database: str,
        embedding_function: str = "default",
        model_name: Optional[str] = None,
    ):
        """Initialize Chroma embedding provider.
        
        Args:
            api_key: Chroma Cloud API key
            tenant: Chroma Cloud tenant ID
            database: Database name
            embedding_function: Chroma embedding function to use
                - "default": DefaultEmbeddingFunction
                - "openai": OpenAIEmbeddingFunction
                - "sentence-transformers": SentenceTransformerEmbeddingFunction
            model_name: Specific model name (e.g., "text-embedding-3-small")
        """
        if not CHROMA_AVAILABLE:
            raise ImportError(
                "chromadb not installed. Install with: pip install chromadb"
            )
        
        self.api_key = api_key
        self.tenant = tenant
        self.database = database
        self._embedding_function_name = embedding_function
        self._model_name = model_name or "default"
        
        # Initialize client
        self.client = chromadb.CloudClient(
            api_key=api_key,
            tenant=tenant,
            database=database,
        )
        
        # Set up embedding function
        self._embedding_function = self._get_embedding_function(
            embedding_function,
            model_name,
        )
        
        logger.info(
            f"Initialized Chroma embedding provider: "
            f"function={embedding_function}, model={model_name}"
        )
    
    def _get_embedding_function(
        self,
        function_name: str,
        model_name: Optional[str] = None,
    ):
        """Get Chroma embedding function."""
        if function_name == "default":
            return chromadb.utils.embedding_functions.DefaultEmbeddingFunction()
        
        elif function_name == "openai":
            if model_name is None:
                model_name = "text-embedding-3-small"
            return chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
                model_name=model_name,
            )
        
        elif function_name == "sentence-transformers":
            if model_name is None:
                model_name = "all-MiniLM-L6-v2"
            return chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name,
            )
        
        else:
            raise ValueError(
                f"Unknown embedding function: {function_name}. "
                f"Supported: default, openai, sentence-transformers"
            )
    
    @property
    def model_name(self) -> str:
        """Name of the embedding model."""
        return f"chroma-{self._embedding_function_name}-{self._model_name}"
    
    @property
    def dimension(self) -> int:
        """Dimensionality of embeddings.
        
        This is tricky since Chroma doesn't expose dimension directly.
        We'll embed a test string to determine it.
        """
        if not hasattr(self, "_dimension"):
            test_embedding = self._embedding_function(["test"])[0]
            self._dimension = len(test_embedding)
        return self._dimension
    
    def embed(self, text: str) -> Embedding:
        """Generate embedding for a single text."""
        # Chroma's embedding functions take lists
        embeddings = self._embedding_function([text])
        vector = np.array(embeddings[0], dtype=np.float32)
        
        return Embedding(vector=vector, model=self.model_name)
    
    def embed_batch(self, texts: List[str]) -> List[Embedding]:
        """Generate embeddings for a batch of texts."""
        embeddings = self._embedding_function(texts)
        
        return [
            Embedding(
                vector=np.array(emb, dtype=np.float32),
                model=self.model_name,
            )
            for emb in embeddings
        ]


class ChromaNeighborhoodProvider(NeighborhoodProvider):
    """Neighborhood provider using Chroma's vector search.
    
    Stores videos in Chroma collections and uses built-in ANN search.
    """
    
    def __init__(
        self,
        api_key: str,
        tenant: str,
        database: str,
        collection_name: str = "idearank_videos",
        embedding_function: Optional[Any] = None,
    ):
        """Initialize Chroma neighborhood provider.
        
        Args:
            api_key: Chroma Cloud API key
            tenant: Chroma Cloud tenant ID
            database: Database name
            collection_name: Name for the video collection
            embedding_function: Optional Chroma embedding function
                (if None, uses default)
        """
        if not CHROMA_AVAILABLE:
            raise ImportError(
                "chromadb not installed. Install with: pip install chromadb"
            )
        
        self.api_key = api_key
        self.tenant = tenant
        self.database = database
        self.collection_name = collection_name
        
        # Initialize client
        self.client = chromadb.CloudClient(
            api_key=api_key,
            tenant=tenant,
            database=database,
        )
        
        # Get or create collection
        if embedding_function is None:
            embedding_function = chromadb.utils.embedding_functions.DefaultEmbeddingFunction()
        
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=embedding_function,
            )
            logger.info(f"Using existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=embedding_function,
                metadata={"description": "IdeaRank video embeddings"},
            )
            logger.info(f"Created new collection: {collection_name}")
        
        # Video lookup cache
        self._video_cache: Dict[str, Video] = {}
    
    def find_global_neighbors(
        self,
        embedding: Embedding,
        k: int = 50,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[Tuple[Video, float]]:
        """Find k nearest neighbors from entire corpus."""
        exclude_ids = exclude_ids or []
        
        # Query Chroma
        results = self.collection.query(
            query_embeddings=[embedding.vector.tolist()],
            n_results=k + len(exclude_ids),  # Get extra in case we need to filter
        )
        
        # Parse results
        neighbors = []
        for i, video_id in enumerate(results['ids'][0]):
            if video_id in exclude_ids:
                continue
            
            if len(neighbors) >= k:
                break
            
            # Get video from cache
            if video_id in self._video_cache:
                video = self._video_cache[video_id]
                # Distance in Chroma is typically L2; convert to similarity
                # similarity ≈ 1 / (1 + distance)
                distance = results['distances'][0][i]
                similarity = 1.0 / (1.0 + distance)
                neighbors.append((video, similarity))
        
        return neighbors
    
    def find_intra_channel_neighbors(
        self,
        embedding: Embedding,
        channel_id: str,
        k: int = 15,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[Tuple[Video, float]]:
        """Find k nearest neighbors within a specific channel."""
        exclude_ids = exclude_ids or []
        
        # Query with channel filter
        results = self.collection.query(
            query_embeddings=[embedding.vector.tolist()],
            n_results=k + len(exclude_ids) + 50,  # Get extras for filtering
            where={"channel_id": channel_id},  # Filter by channel
        )
        
        # Parse results
        neighbors = []
        for i, video_id in enumerate(results['ids'][0]):
            if video_id in exclude_ids:
                continue
            
            if len(neighbors) >= k:
                break
            
            if video_id in self._video_cache:
                video = self._video_cache[video_id]
                distance = results['distances'][0][i]
                similarity = 1.0 / (1.0 + distance)
                neighbors.append((video, similarity))
        
        return neighbors
    
    def index_video(self, video: Video) -> None:
        """Add a video to the Chroma collection."""
        if video.embedding is None:
            raise ValueError(f"Video {video.id} has no embedding")
        
        # Add to collection
        self.collection.add(
            ids=[video.id],
            embeddings=[video.embedding.vector.tolist()],
            metadatas=[{
                "channel_id": video.channel_id,
                "title": video.title,
                "published_at": video.published_at.isoformat(),
            }],
            documents=[video.full_text],
        )
        
        # Cache video for retrieval
        self._video_cache[video.id] = video
        
        logger.debug(f"Indexed video: {video.id}")
    
    def index_videos_batch(self, videos: List[Video]) -> None:
        """Add multiple videos to the collection."""
        # Filter videos with embeddings
        valid_videos = [v for v in videos if v.embedding is not None]
        
        if not valid_videos:
            logger.warning("No videos with embeddings to index")
            return
        
        # Prepare batch data
        ids = [v.id for v in valid_videos]
        embeddings = [v.embedding.vector.tolist() for v in valid_videos]
        metadatas = [
            {
                "channel_id": v.channel_id,
                "title": v.title,
                "published_at": v.published_at.isoformat(),
            }
            for v in valid_videos
        ]
        documents = [v.full_text for v in valid_videos]
        
        # Batch add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )
        
        # Cache videos
        for video in valid_videos:
            self._video_cache[video.id] = video
        
        logger.info(f"Indexed {len(valid_videos)} videos in batch")
    
    def clear_collection(self) -> None:
        """Clear all videos from the collection."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "IdeaRank video embeddings"},
        )
        self._video_cache.clear()
        logger.info(f"Cleared collection: {self.collection_name}")


class ChromaProvider:
    """Combined provider for both embedding and neighborhood search.
    
    This is a convenience class that provides both EmbeddingProvider
    and NeighborhoodProvider using the same Chroma Cloud instance.
    """
    
    def __init__(
        self,
        api_key: str,
        tenant: str,
        database: str,
        embedding_function: str = "default",
        model_name: Optional[str] = None,
        collection_name: str = "idearank_videos",
    ):
        """Initialize combined Chroma provider.
        
        Args:
            api_key: Chroma Cloud API key
            tenant: Chroma Cloud tenant ID
            database: Database name
            embedding_function: Chroma embedding function to use
            model_name: Specific model name
            collection_name: Collection name for video storage
        """
        if not CHROMA_AVAILABLE:
            raise ImportError(
                "chromadb not installed. Install with: pip install chromadb"
            )
        
        # Initialize embedding provider
        self.embedding_provider = ChromaEmbeddingProvider(
            api_key=api_key,
            tenant=tenant,
            database=database,
            embedding_function=embedding_function,
            model_name=model_name,
        )
        
        # Initialize neighborhood provider with same embedding function
        self.neighborhood_provider = ChromaNeighborhoodProvider(
            api_key=api_key,
            tenant=tenant,
            database=database,
            collection_name=collection_name,
            embedding_function=self.embedding_provider._embedding_function,
        )
        
        logger.info(
            f"Initialized combined Chroma provider: "
            f"database={database}, collection={collection_name}"
        )
    
    def get_embedding_provider(self) -> ChromaEmbeddingProvider:
        """Get the embedding provider."""
        return self.embedding_provider
    
    def get_neighborhood_provider(self) -> ChromaNeighborhoodProvider:
        """Get the neighborhood provider."""
        return self.neighborhood_provider


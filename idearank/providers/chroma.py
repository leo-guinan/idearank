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

from idearank.models import ContentItem, Embedding, TopicMixture
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
        
        # Initialize client (updated chromadb package supports CloudClient with v2 API)
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
    
    Stores content items in Chroma collections and uses built-in ANN search.
    """
    
    def __init__(
        self,
        api_key: str,
        tenant: str,
        database: str,
        collection_name: str = "idearank_content",
        embedding_function: Optional[Any] = None,
    ):
        """Initialize Chroma neighborhood provider.
        
        Args:
            api_key: Chroma Cloud API key
            tenant: Chroma Cloud tenant ID
            database: Database name
            collection_name: Name for the content collection
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
        
        # Initialize client (updated chromadb package supports CloudClient with v2 API)
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
                metadata={"description": "IdeaRank content item embeddings"},
            )
            logger.info(f"Created new collection: {collection_name}")
        
        # Content item lookup cache
        self._content_cache: Dict[str, ContentItem] = {}
    
    def find_global_neighbors(
        self,
        embedding: Embedding,
        k: int = 50,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[Tuple[ContentItem, float]]:
        """Find k nearest neighbors from entire corpus."""
        exclude_ids = exclude_ids or []
        
        # Query Chroma
        results = self.collection.query(
            query_embeddings=[embedding.vector.tolist()],
            n_results=k + len(exclude_ids),  # Get extra in case we need to filter
        )
        
        # Parse results
        neighbors = []
        for i, content_id in enumerate(results['ids'][0]):
            if content_id in exclude_ids:
                continue
            
            if len(neighbors) >= k:
                break
            
            # Get content item from cache
            if content_id in self._content_cache:
                content_item = self._content_cache[content_id]
                # Distance in Chroma is typically L2; convert to similarity
                # similarity ≈ 1 / (1 + distance)
                distance = results['distances'][0][i]
                similarity = 1.0 / (1.0 + distance)
                neighbors.append((content_item, similarity))
        
        return neighbors
    
    def find_intra_source_neighbors(
        self,
        embedding: Embedding,
        content_source_id: str,
        k: int = 15,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[Tuple[ContentItem, float]]:
        """Find k nearest neighbors within a specific content source."""
        exclude_ids = exclude_ids or []
        
        # Query with source filter
        results = self.collection.query(
            query_embeddings=[embedding.vector.tolist()],
            n_results=k + len(exclude_ids) + 50,  # Get extras for filtering
            where={"content_source_id": content_source_id},  # Filter by source
        )
        
        # Parse results
        neighbors = []
        for i, content_id in enumerate(results['ids'][0]):
            if content_id in exclude_ids:
                continue
            
            if len(neighbors) >= k:
                break
            
            if content_id in self._content_cache:
                content_item = self._content_cache[content_id]
                distance = results['distances'][0][i]
                similarity = 1.0 / (1.0 + distance)
                neighbors.append((content_item, similarity))
        
        return neighbors
    
    def index_content_item(self, content_item: ContentItem) -> None:
        """Add a content item to the Chroma collection."""
        if content_item.embedding is None:
            raise ValueError(f"Content item {content_item.id} has no embedding")
        
        # Add to collection
        self.collection.add(
            ids=[content_item.id],
            embeddings=[content_item.embedding.vector.tolist()],
            metadatas=[{
                "content_source_id": content_item.content_source_id,
                "title": content_item.title,
                "published_at": content_item.published_at.isoformat(),
            }],
            documents=[content_item.full_text],
        )
        
        # Cache content item for retrieval
        self._content_cache[content_item.id] = content_item
        
        logger.debug(f"Indexed content item: {content_item.id}")
    
    def index_content_batch(self, content_items: List[ContentItem]) -> None:
        """Add multiple content items to the collection."""
        # Filter items with embeddings
        valid_items = [item for item in content_items if item.embedding is not None]
        
        if not valid_items:
            logger.warning("No content items with embeddings to index")
            return
        
        # Prepare batch data
        ids = [item.id for item in valid_items]
        embeddings = [item.embedding.vector.tolist() for item in valid_items]
        metadatas = [
            {
                "content_source_id": item.content_source_id,
                "title": item.title,
                "published_at": item.published_at.isoformat(),
            }
            for item in valid_items
        ]
        documents = [item.full_text for item in valid_items]
        
        # Batch add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )
        
        # Cache content items
        for item in valid_items:
            self._content_cache[item.id] = item
        
        logger.info(f"Indexed {len(valid_items)} content items in batch")
    
    def clear_collection(self) -> None:
        """Clear all content items from the collection."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "IdeaRank content item embeddings"},
        )
        self._content_cache.clear()
        logger.info(f"Cleared collection: {self.collection_name}")


class ChromaProvider:
    """Combined provider for both embedding and neighborhood search.
    
    Supports both local (persistent) and cloud modes.
    """
    
    def __init__(
        self,
        collection_name: str = "idearank_content",
        persist_directory: Optional[str] = None,
        embedding_function: str = "default",
        model_name: Optional[str] = None,
        # Cloud-only parameters
        chroma_cloud_api_key: Optional[str] = None,
        chroma_cloud_tenant: str = "default_tenant",
        chroma_cloud_database: str = "default_database",
    ):
        """Initialize combined Chroma provider.
        
        Args:
            collection_name: Collection name for content storage
            persist_directory: Local directory for persistence (None = use cloud)
            embedding_function: Chroma embedding function to use
            model_name: Specific model name
            chroma_cloud_api_key: Chroma Cloud API key (required for cloud mode)
            chroma_cloud_tenant: Chroma Cloud tenant (cloud mode only)
            chroma_cloud_database: Chroma Cloud database (cloud mode only)
        """
        if not CHROMA_AVAILABLE:
            raise ImportError(
                "chromadb not installed. Install with: pip install chromadb"
            )
        
        self.is_cloud = persist_directory is None
        
        if self.is_cloud:
            # Cloud mode
            if not chroma_cloud_api_key:
                raise ValueError("chroma_cloud_api_key required for cloud mode")
            
            logger.info(f"Initializing Chroma Cloud provider: tenant={chroma_cloud_tenant}, db={chroma_cloud_database}")
            
            # Initialize embedding provider (cloud)
            self.embedding_provider = ChromaEmbeddingProvider(
                api_key=chroma_cloud_api_key,
                tenant=chroma_cloud_tenant,
                database=chroma_cloud_database,
                embedding_function=embedding_function,
                model_name=model_name,
            )
            
            # Initialize neighborhood provider (cloud)
            self.neighborhood_provider = ChromaNeighborhoodProvider(
                api_key=chroma_cloud_api_key,
                tenant=chroma_cloud_tenant,
                database=chroma_cloud_database,
                collection_name=collection_name,
                embedding_function=self.embedding_provider._embedding_function,
            )
            
            logger.info(f"Chroma Cloud provider ready: collection={collection_name}")
            
        else:
            # Local mode - use persistent ChromaDB with sentence transformers
            logger.info(f"Initializing local Chroma provider: {persist_directory}")
            
            # Initialize local Chroma client
            self.client = chromadb.PersistentClient(path=persist_directory)
            
            # Set up embedding function for local mode
            if embedding_function == "sentence-transformers":
                if model_name is None:
                    model_name = "all-MiniLM-L6-v2"
                emb_func = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=model_name,
                )
            elif embedding_function == "default":
                emb_func = chromadb.utils.embedding_functions.DefaultEmbeddingFunction()
            else:
                raise ValueError(
                    f"Local mode only supports 'default' or 'sentence-transformers', got: {embedding_function}"
                )
            
            # Create local embedding provider
            from idearank.providers.embeddings import SentenceTransformerEmbeddingProvider
            self.embedding_provider = SentenceTransformerEmbeddingProvider(
                model_name=model_name or "all-MiniLM-L6-v2"
            )
            
            # Create local neighborhood provider
            self.neighborhood_provider = self._create_local_neighborhood_provider(
                collection_name, emb_func
            )
            
            logger.info(
                f"Local Chroma provider ready: {persist_directory}, "
                f"collection={collection_name}, model={model_name or 'all-MiniLM-L6-v2'}"
            )
    
    def get_embedding_provider(self):
        """Get the embedding provider."""
        return self.embedding_provider
    
    def get_neighborhood_provider(self):
        """Get the neighborhood provider."""
        return self.neighborhood_provider
    
    # Delegation methods for neighborhood provider
    def index_content_item(self, content_item) -> None:
        """Delegate to neighborhood provider."""
        self.neighborhood_provider.index_content_item(content_item)
    
    def index_content_batch(self, content_items) -> None:
        """Delegate to neighborhood provider."""
        self.neighborhood_provider.index_content_batch(content_items)
    
    def find_global_neighbors(self, embedding, k: int = 50, exclude_ids=None):
        """Delegate to neighborhood provider."""
        return self.neighborhood_provider.find_global_neighbors(embedding, k, exclude_ids)
    
    def find_intra_source_neighbors(self, embedding, content_source_id: str, k: int = 15, exclude_ids=None):
        """Delegate to neighborhood provider."""
        return self.neighborhood_provider.find_intra_source_neighbors(embedding, content_source_id, k, exclude_ids)
    
    def _create_local_neighborhood_provider(self, collection_name: str, embedding_function):
        """Create a local neighborhood provider using persistent Chroma client."""
        
        class LocalChromaNeighborhoodProvider(NeighborhoodProvider):
            """Local ChromaDB neighborhood provider."""
            
            def __init__(self, client, collection_name: str, embedding_function):
                self.client = client
                self.collection_name = collection_name
                self.embedding_function = embedding_function
                self._content_cache: Dict[str, ContentItem] = {}
                
                # Get or create collection
                try:
                    self.collection = self.client.get_collection(
                        name=collection_name,
                        embedding_function=embedding_function,
                    )
                    logger.info(f"Using existing local collection: {collection_name}")
                except Exception:
                    self.collection = self.client.create_collection(
                        name=collection_name,
                        embedding_function=embedding_function,
                        metadata={"description": "IdeaRank content item embeddings"},
                    )
                    logger.info(f"Created new local collection: {collection_name}")
            
            def find_global_neighbors(
                self,
                embedding: Embedding,
                k: int = 50,
                exclude_ids: Optional[List[str]] = None,
            ) -> List[Tuple[ContentItem, float]]:
                """Find k nearest neighbors from entire corpus."""
                exclude_ids = exclude_ids or []
                
                # Query Chroma
                results = self.collection.query(
                    query_embeddings=[embedding.vector.tolist()],
                    n_results=k + len(exclude_ids),
                )
                
                # Parse results
                neighbors = []
                for i, content_id in enumerate(results['ids'][0]):
                    if content_id in exclude_ids:
                        continue
                    
                    if len(neighbors) >= k:
                        break
                    
                    if content_id in self._content_cache:
                        content_item = self._content_cache[content_id]
                        distance = results['distances'][0][i]
                        similarity = 1.0 / (1.0 + distance)
                        neighbors.append((content_item, similarity))
                
                return neighbors
            
            def find_intra_source_neighbors(
                self,
                embedding: Embedding,
                content_source_id: str,
                k: int = 15,
                exclude_ids: Optional[List[str]] = None,
            ) -> List[Tuple[ContentItem, float]]:
                """Find k nearest neighbors within a specific content source."""
                exclude_ids = exclude_ids or []
                
                # Query with source filter
                results = self.collection.query(
                    query_embeddings=[embedding.vector.tolist()],
                    n_results=k + len(exclude_ids) + 50,
                    where={"content_source_id": content_source_id},
                )
                
                # Parse results
                neighbors = []
                for i, content_id in enumerate(results['ids'][0]):
                    if content_id in exclude_ids:
                        continue
                    
                    if len(neighbors) >= k:
                        break
                    
                    if content_id in self._content_cache:
                        content_item = self._content_cache[content_id]
                        distance = results['distances'][0][i]
                        similarity = 1.0 / (1.0 + distance)
                        neighbors.append((content_item, similarity))
                
                return neighbors
            
            def index_content_item(self, content_item: ContentItem) -> None:
                """Add a content item to the Chroma collection."""
                if content_item.embedding is None:
                    raise ValueError(f"Content item {content_item.id} has no embedding")
                
                self.collection.add(
                    ids=[content_item.id],
                    embeddings=[content_item.embedding.vector.tolist()],
                    metadatas=[{
                        "content_source_id": content_item.content_source_id,
                        "title": content_item.title,
                        "published_at": content_item.published_at.isoformat(),
                    }],
                    documents=[content_item.full_text],
                )
                
                self._content_cache[content_item.id] = content_item
                logger.debug(f"Indexed content item: {content_item.id}")
            
            def index_content_batch(self, content_items: List[ContentItem]) -> None:
                """Add multiple content items to the collection."""
                valid_items = [item for item in content_items if item.embedding is not None]
                
                if not valid_items:
                    logger.warning("No content items with embeddings to index")
                    return
                
                ids = [item.id for item in valid_items]
                embeddings = [item.embedding.vector.tolist() for item in valid_items]
                metadatas = [
                    {
                        "content_source_id": item.content_source_id,
                        "title": item.title,
                        "published_at": item.published_at.isoformat(),
                    }
                    for item in valid_items
                ]
                documents = [item.full_text for item in valid_items]
                
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents,
                )
                
                for item in valid_items:
                    self._content_cache[item.id] = item
                
                logger.info(f"Indexed {len(valid_items)} content items in batch")
        
        return LocalChromaNeighborhoodProvider(self.client, collection_name, embedding_function)


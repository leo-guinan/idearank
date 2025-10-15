"""Dual Chroma provider for parallel comparison of semantic units vs regular chunks.

This provider manages TWO separate Chroma collections:
1. Semantic Units Collection - stores extracted actors, events, changes
2. Chunks Collection - stores regular document chunks

This enables A/B testing different embedding strategies.
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

from idearank.models import ContentItem, Embedding
from idearank.providers.embeddings import EmbeddingProvider
from idearank.providers.neighborhoods import NeighborhoodProvider
from idearank.utils.chunking import DocumentChunker, DocumentChunk
from idearank.semantic_extractor import SemanticExtractor, FallbackSemanticExtractor, SemanticStructure

logger = logging.getLogger(__name__)


class DualChromaProvider:
    """Manages two parallel Chroma collections for comparison.
    
    Collections:
    - semantic_units: Embeddings of extracted semantic primitives
    - document_chunks: Embeddings of regular text chunks
    
    Both collections use the same embedding model for fair comparison.
    """
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        embedding_function: str = "sentence-transformers",
        model_name: Optional[str] = None,
        # Cloud-only parameters
        chroma_cloud_api_key: Optional[str] = None,
        chroma_cloud_tenant: str = "default_tenant",
        chroma_cloud_database: str = "default_database",
        # Semantic extraction
        semantic_extractor: Optional[SemanticExtractor] = None,
        # Chunking parameters
        chunk_size: int = 8000,
        chunk_overlap: int = 500,
    ):
        """Initialize dual Chroma provider.
        
        Args:
            persist_directory: Local directory for persistence (None = use cloud)
            embedding_function: Chroma embedding function
            model_name: Specific model name
            chroma_cloud_api_key: Cloud API key (required for cloud mode)
            chroma_cloud_tenant: Cloud tenant
            chroma_cloud_database: Cloud database
            semantic_extractor: Extractor for semantic units
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
        """
        if not CHROMA_AVAILABLE:
            raise ImportError("chromadb not installed. Install with: pip install chromadb")
        
        self.is_cloud = persist_directory is None
        
        # Initialize embedding provider
        from idearank.providers.embeddings import SentenceTransformerEmbeddingProvider
        self.embedding_provider = SentenceTransformerEmbeddingProvider(
            model_name=model_name or "all-MiniLM-L6-v2"
        )
        
        # Initialize semantic extractor
        self.semantic_extractor = semantic_extractor or FallbackSemanticExtractor()
        
        # Initialize chunker
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        # Initialize Chroma client
        if self.is_cloud:
            if not chroma_cloud_api_key:
                raise ValueError("chroma_cloud_api_key required for cloud mode")
            
            self.client = chromadb.CloudClient(
                api_key=chroma_cloud_api_key,
                tenant=chroma_cloud_tenant,
                database=chroma_cloud_database,
            )
            logger.info(f"Using Chroma Cloud: {chroma_cloud_tenant}/{chroma_cloud_database}")
        else:
            self.client = chromadb.PersistentClient(path=persist_directory)
            logger.info(f"Using local Chroma: {persist_directory}")
        
        # Set up embedding function for Chroma
        if embedding_function == "sentence-transformers":
            self.chroma_embedding_func = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name or "all-MiniLM-L6-v2"
            )
        elif embedding_function == "default":
            self.chroma_embedding_func = chromadb.utils.embedding_functions.DefaultEmbeddingFunction()
        else:
            raise ValueError(f"Unsupported embedding function: {embedding_function}")
        
        # Create or get collections
        self.semantic_collection = self._get_or_create_collection("idearank_semantic_units")
        self.chunks_collection = self._get_or_create_collection("idearank_document_chunks")
        
        # Storage for semantic structures and chunks
        self.semantic_structures: Dict[str, SemanticStructure] = {}
        self.chunks_by_content: Dict[str, List[DocumentChunk]] = {}
        
        logger.info(
            f"Dual Chroma provider initialized with collections: "
            f"semantic_units, document_chunks"
        )
    
    def _get_or_create_collection(self, name: str):
        """Get or create a Chroma collection."""
        try:
            collection = self.client.get_collection(
                name=name,
                embedding_function=self.chroma_embedding_func,
            )
            logger.info(f"Using existing collection: {name}")
        except Exception:
            collection = self.client.create_collection(
                name=name,
                embedding_function=self.chroma_embedding_func,
                metadata={"description": f"IdeaRank {name}"},
            )
            logger.info(f"Created new collection: {name}")
        
        return collection
    
    def process_and_index_content(
        self,
        content_item: ContentItem,
        mode: str = "both"  # "semantic", "chunks", or "both"
    ) -> Dict[str, Any]:
        """Process content item and index in both collections.
        
        Args:
            content_item: Content item to process
            mode: Which collections to index ("semantic", "chunks", or "both")
            
        Returns:
            Dict with processing stats
        """
        stats = {
            "content_id": content_item.id,
            "semantic_units_count": 0,
            "chunks_count": 0,
            "semantic_indexed": False,
            "chunks_indexed": False,
        }
        
        # Process semantic units
        if mode in ["semantic", "both"]:
            semantic_result = self._process_semantic_units(content_item)
            stats.update(semantic_result)
        
        # Process regular chunks
        if mode in ["chunks", "both"]:
            chunks_result = self._process_chunks(content_item)
            stats.update(chunks_result)
        
        return stats
    
    def _process_semantic_units(self, content_item: ContentItem) -> Dict[str, Any]:
        """Extract and index semantic units."""
        logger.info(f"Extracting semantic units from {content_item.id}")
        
        # Extract semantic structure
        semantic_structure = self.semantic_extractor.extract(
            content_id=content_item.id,
            text=content_item.full_text,
            title=content_item.title if hasattr(content_item, 'title') else "",
        )
        
        # Store structure
        self.semantic_structures[content_item.id] = semantic_structure
        
        # Get embeddable units
        units = semantic_structure.get_embeddable_units()
        
        if not units:
            logger.warning(f"No semantic units extracted from {content_item.id}")
            return {"semantic_units_count": 0, "semantic_indexed": False}
        
        # Embed units
        unit_texts = [text for _, _, text in units]
        unit_embeddings = self.embedding_provider.embed_batch(unit_texts)
        
        # Index in Chroma
        unit_ids = [unit_id for _, unit_id, _ in units]
        unit_types = [unit_type for unit_type, _, _ in units]
        
        self.semantic_collection.add(
            ids=unit_ids,
            embeddings=[emb.vector.tolist() for emb in unit_embeddings],
            documents=unit_texts,
            metadatas=[
                {
                    "content_id": content_item.id,
                    "content_source_id": content_item.content_source_id,
                    "unit_type": unit_type,
                    "title": content_item.title,
                }
                for unit_type in unit_types
            ],
        )
        
        logger.info(
            f"Indexed {len(units)} semantic units from {content_item.id}: "
            f"{len(semantic_structure.actors)} actors, "
            f"{len(semantic_structure.events)} events, "
            f"{len(semantic_structure.changes)} changes"
        )
        
        return {
            "semantic_units_count": len(units),
            "semantic_indexed": True,
        }
    
    def _process_chunks(self, content_item: ContentItem) -> Dict[str, Any]:
        """Chunk and index regular document chunks."""
        logger.info(f"Chunking document {content_item.id}")
        
        # Chunk document
        chunks = self.chunker.chunk_document(
            text=content_item.full_text,
            parent_id=content_item.id,
        )
        
        # Store chunks
        self.chunks_by_content[content_item.id] = chunks
        
        if not chunks:
            logger.warning(f"No chunks created from {content_item.id}")
            return {"chunks_count": 0, "chunks_indexed": False}
        
        # Embed chunks
        chunk_texts = [chunk.text for chunk in chunks]
        chunk_embeddings = self.embedding_provider.embed_batch(chunk_texts)
        
        # Index in Chroma
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        
        self.chunks_collection.add(
            ids=chunk_ids,
            embeddings=[emb.vector.tolist() for emb in chunk_embeddings],
            documents=chunk_texts,
            metadatas=[
                {
                    "content_id": content_item.id,
                    "content_source_id": content_item.content_source_id,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    "title": content_item.title,
                }
                for chunk in chunks
            ],
        )
        
        logger.info(f"Indexed {len(chunks)} chunks from {content_item.id}")
        
        return {
            "chunks_count": len(chunks),
            "chunks_indexed": True,
        }
    
    def query_semantic(
        self,
        query_text: str,
        k: int = 10,
        content_source_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query the semantic units collection.
        
        Args:
            query_text: Query text
            k: Number of results
            content_source_id: Optional filter by content source
            
        Returns:
            List of result dicts with metadata
        """
        query_embedding = self.embedding_provider.embed(query_text)
        
        where = {"content_source_id": content_source_id} if content_source_id else None
        
        results = self.semantic_collection.query(
            query_embeddings=[query_embedding.vector.tolist()],
            n_results=k,
            where=where,
        )
        
        return self._format_results(results, "semantic")
    
    def query_chunks(
        self,
        query_text: str,
        k: int = 10,
        content_source_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query the chunks collection.
        
        Args:
            query_text: Query text
            k: Number of results
            content_source_id: Optional filter by content source
            
        Returns:
            List of result dicts with metadata
        """
        query_embedding = self.embedding_provider.embed(query_text)
        
        where = {"content_source_id": content_source_id} if content_source_id else None
        
        results = self.chunks_collection.query(
            query_embeddings=[query_embedding.vector.tolist()],
            n_results=k,
            where=where,
        )
        
        return self._format_results(results, "chunks")
    
    def compare_queries(
        self,
        query_text: str,
        k: int = 10,
        content_source_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Query both collections and compare results.
        
        Args:
            query_text: Query text
            k: Number of results per collection
            content_source_id: Optional filter by content source
            
        Returns:
            Dict with results from both collections and comparison metrics
        """
        semantic_results = self.query_semantic(query_text, k, content_source_id)
        chunk_results = self.query_chunks(query_text, k, content_source_id)
        
        # Calculate overlap
        semantic_content_ids = {r["content_id"] for r in semantic_results}
        chunk_content_ids = {r["content_id"] for r in chunk_results}
        overlap = semantic_content_ids & chunk_content_ids
        
        return {
            "query": query_text,
            "semantic_results": semantic_results,
            "chunk_results": chunk_results,
            "overlap_count": len(overlap),
            "overlap_percentage": len(overlap) / k * 100 if k > 0 else 0,
            "unique_to_semantic": len(semantic_content_ids - chunk_content_ids),
            "unique_to_chunks": len(chunk_content_ids - semantic_content_ids),
        }
    
    def _format_results(self, results: Dict, source: str) -> List[Dict[str, Any]]:
        """Format Chroma results for output."""
        formatted = []
        
        for i in range(len(results['ids'][0])):
            formatted.append({
                "id": results['ids'][0][i],
                "document": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i],
                "similarity": 1.0 / (1.0 + results['distances'][0][i]),
                "source": source,
                "content_id": results['metadatas'][0][i].get("content_id"),
            })
        
        return formatted
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about both collections."""
        semantic_count = self.semantic_collection.count()
        chunks_count = self.chunks_collection.count()
        
        return {
            "semantic_units_total": semantic_count,
            "chunks_total": chunks_count,
            "content_items_with_semantics": len(self.semantic_structures),
            "content_items_with_chunks": len(self.chunks_by_content),
            "embedding_model": self.embedding_provider.model_name,
            "embedding_dimension": self.embedding_provider.dimension,
        }
    
    def clear_collections(self):
        """Clear both collections."""
        self.client.delete_collection("idearank_semantic_units")
        self.client.delete_collection("idearank_document_chunks")
        
        self.semantic_collection = self._get_or_create_collection("idearank_semantic_units")
        self.chunks_collection = self._get_or_create_collection("idearank_document_chunks")
        
        self.semantic_structures.clear()
        self.chunks_by_content.clear()
        
        logger.info("Cleared both semantic and chunks collections")


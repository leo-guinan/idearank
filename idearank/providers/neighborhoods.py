"""Neighborhood provider interface for ANN search."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np

from idearank.models import ContentItem, Embedding


class NeighborhoodProvider(ABC):
    """Abstract interface for approximate nearest neighbor search."""
    
    @abstractmethod
    def find_global_neighbors(
        self,
        embedding: Embedding,
        k: int = 50,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[Tuple[ContentItem, float]]:
        """Find k nearest neighbors from global corpus.
        
        Args:
            embedding: Query embedding
            k: Number of neighbors to return
            exclude_ids: Content item IDs to exclude from results
            
        Returns:
            List of (content_item, similarity) tuples, sorted by decreasing similarity
        """
        pass
    
    @abstractmethod
    def find_intra_source_neighbors(
        self,
        embedding: Embedding,
        content_source_id: str,
        k: int = 15,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[Tuple[ContentItem, float]]:
        """Find k nearest neighbors within the same content source.
        
        Args:
            embedding: Query embedding
            content_source_id: Content source to search within
            k: Number of neighbors to return
            exclude_ids: Content item IDs to exclude from results
            
        Returns:
            List of (content_item, similarity) tuples, sorted by decreasing similarity
        """
        pass
    
    @abstractmethod
    def index_content_item(self, content_item: ContentItem) -> None:
        """Add a content item to the index."""
        pass
    
    @abstractmethod
    def index_content_batch(self, content_items: List[ContentItem]) -> None:
        """Add multiple content items to the index."""
        pass


class DummyNeighborhoodProvider(NeighborhoodProvider):
    """Dummy provider using brute-force search - for testing only."""
    
    def __init__(self):
        """Initialize empty index."""
        self.content_items: List[ContentItem] = []
        self.items_by_source: dict[str, List[ContentItem]] = {}
    
    def find_global_neighbors(
        self,
        embedding: Embedding,
        k: int = 50,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[Tuple[ContentItem, float]]:
        """Brute-force search over all content items."""
        exclude_ids = exclude_ids or []
        
        candidates = [
            item for item in self.content_items
            if item.id not in exclude_ids and item.embedding is not None
        ]
        
        # Compute similarities
        similarities = []
        for item in candidates:
            if item.embedding is not None:
                sim = embedding.cosine_similarity(item.embedding)
                similarities.append((item, float(sim)))
        
        # Sort by similarity (descending) and take top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def find_intra_source_neighbors(
        self,
        embedding: Embedding,
        content_source_id: str,
        k: int = 15,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[Tuple[ContentItem, float]]:
        """Search within a specific content source."""
        exclude_ids = exclude_ids or []
        
        candidates = [
            item for item in self.items_by_source.get(content_source_id, [])
            if item.id not in exclude_ids and item.embedding is not None
        ]
        
        # Compute similarities
        similarities = []
        for item in candidates:
            if item.embedding is not None:
                sim = embedding.cosine_similarity(item.embedding)
                similarities.append((item, float(sim)))
        
        # Sort and take top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def index_content_item(self, content_item: ContentItem) -> None:
        """Add content item to index."""
        self.content_items.append(content_item)
        
        if content_item.content_source_id not in self.items_by_source:
            self.items_by_source[content_item.content_source_id] = []
        self.items_by_source[content_item.content_source_id].append(content_item)
    
    def index_content_batch(self, content_items: List[ContentItem]) -> None:
        """Add multiple content items."""
        for item in content_items:
            self.index_content_item(item)


class FAISSNeighborhoodProvider(NeighborhoodProvider):
    """Provider using FAISS for efficient ANN search.
    
    NOTE: Requires faiss package.
    This is a stub - implement when needed.
    """
    
    def __init__(self, dimension: int = 384):
        """Initialize FAISS index."""
        self.dimension = dimension
        # TODO: Initialize FAISS index
        # self.index_global = faiss.IndexFlatIP(dimension)
        # self.index_by_source = {}
    
    def find_global_neighbors(
        self,
        embedding: Embedding,
        k: int = 50,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[Tuple[ContentItem, float]]:
        """Search using FAISS."""
        raise NotImplementedError("FAISS provider not yet implemented")
    
    def find_intra_source_neighbors(
        self,
        embedding: Embedding,
        content_source_id: str,
        k: int = 15,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[Tuple[ContentItem, float]]:
        """Search within content source using FAISS."""
        raise NotImplementedError("FAISS provider not yet implemented")
    
    def index_content_item(self, content_item: ContentItem) -> None:
        """Add to FAISS index."""
        raise NotImplementedError("FAISS provider not yet implemented")
    
    def index_content_batch(self, content_items: List[ContentItem]) -> None:
        """Batch add to FAISS."""
        raise NotImplementedError("FAISS provider not yet implemented")


"""Neighborhood provider interface for ANN search."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np

from idearank.models import Video, Embedding


class NeighborhoodProvider(ABC):
    """Abstract interface for approximate nearest neighbor search."""
    
    @abstractmethod
    def find_global_neighbors(
        self,
        embedding: Embedding,
        k: int = 50,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[Tuple[Video, float]]:
        """Find k nearest neighbors from global corpus.
        
        Args:
            embedding: Query embedding
            k: Number of neighbors to return
            exclude_ids: Video IDs to exclude from results
            
        Returns:
            List of (video, similarity) tuples, sorted by decreasing similarity
        """
        pass
    
    @abstractmethod
    def find_intra_channel_neighbors(
        self,
        embedding: Embedding,
        channel_id: str,
        k: int = 15,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[Tuple[Video, float]]:
        """Find k nearest neighbors within the same channel.
        
        Args:
            embedding: Query embedding
            channel_id: Channel to search within
            k: Number of neighbors to return
            exclude_ids: Video IDs to exclude from results
            
        Returns:
            List of (video, similarity) tuples, sorted by decreasing similarity
        """
        pass
    
    @abstractmethod
    def index_video(self, video: Video) -> None:
        """Add a video to the index."""
        pass
    
    @abstractmethod
    def index_videos_batch(self, videos: List[Video]) -> None:
        """Add multiple videos to the index."""
        pass


class DummyNeighborhoodProvider(NeighborhoodProvider):
    """Dummy provider using brute-force search - for testing only."""
    
    def __init__(self):
        """Initialize empty index."""
        self.videos: List[Video] = []
        self.videos_by_channel: dict[str, List[Video]] = {}
    
    def find_global_neighbors(
        self,
        embedding: Embedding,
        k: int = 50,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[Tuple[Video, float]]:
        """Brute-force search over all videos."""
        exclude_ids = exclude_ids or []
        
        candidates = [
            v for v in self.videos
            if v.id not in exclude_ids and v.embedding is not None
        ]
        
        # Compute similarities
        similarities = []
        for video in candidates:
            if video.embedding is not None:
                sim = embedding.cosine_similarity(video.embedding)
                similarities.append((video, float(sim)))
        
        # Sort by similarity (descending) and take top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def find_intra_channel_neighbors(
        self,
        embedding: Embedding,
        channel_id: str,
        k: int = 15,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[Tuple[Video, float]]:
        """Search within a specific channel."""
        exclude_ids = exclude_ids or []
        
        candidates = [
            v for v in self.videos_by_channel.get(channel_id, [])
            if v.id not in exclude_ids and v.embedding is not None
        ]
        
        # Compute similarities
        similarities = []
        for video in candidates:
            if video.embedding is not None:
                sim = embedding.cosine_similarity(video.embedding)
                similarities.append((video, float(sim)))
        
        # Sort and take top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def index_video(self, video: Video) -> None:
        """Add video to index."""
        self.videos.append(video)
        
        if video.channel_id not in self.videos_by_channel:
            self.videos_by_channel[video.channel_id] = []
        self.videos_by_channel[video.channel_id].append(video)
    
    def index_videos_batch(self, videos: List[Video]) -> None:
        """Add multiple videos."""
        for video in videos:
            self.index_video(video)


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
        # self.index_by_channel = {}
    
    def find_global_neighbors(
        self,
        embedding: Embedding,
        k: int = 50,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[Tuple[Video, float]]:
        """Search using FAISS."""
        raise NotImplementedError("FAISS provider not yet implemented")
    
    def find_intra_channel_neighbors(
        self,
        embedding: Embedding,
        channel_id: str,
        k: int = 15,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[Tuple[Video, float]]:
        """Search within channel using FAISS."""
        raise NotImplementedError("FAISS provider not yet implemented")
    
    def index_video(self, video: Video) -> None:
        """Add to FAISS index."""
        raise NotImplementedError("FAISS provider not yet implemented")
    
    def index_videos_batch(self, videos: List[Video]) -> None:
        """Batch add to FAISS."""
        raise NotImplementedError("FAISS provider not yet implemented")


"""Content source management for IdeaRank.

Manages multiple content sources (YouTube, Ghost, etc.) and auto-detects types.
"""

import json
from pathlib import Path
from typing import List, Optional, Literal
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


ContentType = Literal["youtube", "ghost_export", "ghost_api", "medium", "twitter", "unknown"]


@dataclass
class ContentSource:
    """Represents a content source to be processed."""
    
    id: str  # Unique identifier
    type: ContentType  # Type of content source
    url_or_path: str  # URL or file path
    name: Optional[str] = None  # Optional display name
    max_items: int = 50  # Max videos/posts to process
    filter_query: Optional[str] = None  # Optional filter
    enabled: bool = True  # Whether to process this source
    last_processed: Optional[str] = None  # ISO timestamp
    added_at: str = None  # ISO timestamp when added
    
    def __post_init__(self):
        if self.added_at is None:
            self.added_at = datetime.now().isoformat()
    
    @staticmethod
    def detect_type(url_or_path: str) -> ContentType:
        """Auto-detect content source type.
        
        Args:
            url_or_path: URL or file path
            
        Returns:
            ContentType
        """
        # Check if it's a file
        if Path(url_or_path).exists():
            # Check file extension and name patterns
            filename_lower = url_or_path.lower()
            
            # Twitter archive JSON files
            if filename_lower.endswith('.json') and any(indicator in filename_lower for indicator in ['twitter', 'tweet', 'archive']):
                return "twitter"
            
            # Ghost export files
            if url_or_path.endswith('.json') or '.ghost.' in url_or_path:
                return "ghost_export"
            
            # Medium archives (ZIP)
            if url_or_path.endswith('.zip') and 'medium' in filename_lower:
                return "medium"
            
            # Twitter archives (ZIP) - check for twitter indicators
            if url_or_path.endswith('.zip') and any(indicator in filename_lower for indicator in ['twitter', 'tweet', 'archive']):
                return "twitter"
            
            # For other ZIP files, default to unknown rather than medium
            if url_or_path.endswith('.zip'):
                return "unknown"
            
            return "unknown"
        
        # Check if it's a URL
        url_lower = url_or_path.lower()
        
        if 'youtube.com' in url_lower or 'youtu.be' in url_lower:
            return "youtube"
        
        # Assume it's a Ghost blog URL if it's a URL we don't recognize
        if url_lower.startswith('http'):
            return "ghost_api"
        
        return "unknown"
    
    @classmethod
    def create(
        cls,
        url_or_path: str,
        name: Optional[str] = None,
        max_items: int = 50,
        filter_query: Optional[str] = None,
    ) -> "ContentSource":
        """Create a content source with auto-detected type.
        
        Args:
            url_or_path: URL or file path
            name: Optional display name
            max_items: Max items to process
            filter_query: Optional filter
            
        Returns:
            ContentSource instance
        """
        content_type = cls.detect_type(url_or_path)
        
        # Generate ID
        source_id = f"{content_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Auto-generate name if not provided
        if name is None:
            if content_type == "youtube":
                # Extract channel name from URL
                if '@' in url_or_path:
                    name = url_or_path.split('@')[1].split('/')[0]
                else:
                    name = url_or_path
            elif content_type == "ghost_export":
                name = Path(url_or_path).stem
            elif content_type == "medium":
                name = Path(url_or_path).stem.replace('medium-export', '').replace('-', ' ').strip()
            elif content_type == "twitter":
                # Extract name from file
                name = Path(url_or_path).stem
            else:
                name = url_or_path
        
        return cls(
            id=source_id,
            type=content_type,
            url_or_path=url_or_path,
            name=name,
            max_items=max_items,
            filter_query=filter_query,
        )


class SourcesConfig:
    """Manages list of content sources."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize sources config.
        
        Args:
            config_path: Path to sources file. Defaults to ~/.idearank/sources.json
        """
        if config_path is None:
            config_dir = Path.home() / ".idearank"
            config_dir.mkdir(exist_ok=True)
            config_path = config_dir / "sources.json"
        
        self.config_path = config_path
        self.sources = self._load_sources()
    
    def _load_sources(self) -> List[ContentSource]:
        """Load sources from file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                return [ContentSource(**s) for s in data.get('sources', [])]
            except Exception as e:
                logger.error(f"Failed to load sources: {e}")
                return []
        return []
    
    def _save_sources(self) -> None:
        """Save sources to file."""
        self.config_path.parent.mkdir(exist_ok=True, parents=True)
        
        data = {
            'sources': [asdict(s) for s in self.sources],
            'updated_at': datetime.now().isoformat(),
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_source(self, source: ContentSource) -> None:
        """Add a content source."""
        # Check if URL/path already exists
        existing = self.get_source_by_url(source.url_or_path)
        if existing:
            raise ValueError(f"Source already exists: {existing.name} ({existing.id})")
        
        self.sources.append(source)
        self._save_sources()
        logger.info(f"Added source: {source.name} ({source.type})")
    
    def remove_source(self, source_id: str) -> bool:
        """Remove a content source.
        
        Returns:
            True if source was removed, False if not found
        """
        original_count = len(self.sources)
        self.sources = [s for s in self.sources if s.id != source_id]
        
        if len(self.sources) < original_count:
            self._save_sources()
            logger.info(f"Removed source: {source_id}")
            return True
        
        return False
    
    def get_source(self, source_id: str) -> Optional[ContentSource]:
        """Get source by ID."""
        for source in self.sources:
            if source.id == source_id:
                return source
        return None
    
    def get_source_by_url(self, url_or_path: str) -> Optional[ContentSource]:
        """Get source by URL or path."""
        for source in self.sources:
            if source.url_or_path == url_or_path:
                return source
        return None
    
    def list_sources(self, enabled_only: bool = False) -> List[ContentSource]:
        """List all sources.
        
        Args:
            enabled_only: If True, only return enabled sources
            
        Returns:
            List of ContentSource objects
        """
        if enabled_only:
            return [s for s in self.sources if s.enabled]
        return self.sources
    
    def enable_source(self, source_id: str) -> bool:
        """Enable a source."""
        source = self.get_source(source_id)
        if source:
            source.enabled = True
            self._save_sources()
            return True
        return False
    
    def disable_source(self, source_id: str) -> bool:
        """Disable a source (won't be processed)."""
        source = self.get_source(source_id)
        if source:
            source.enabled = False
            self._save_sources()
            return True
        return False
    
    def update_source(
        self,
        source_id: str,
        name: Optional[str] = None,
        max_items: Optional[int] = None,
        filter_query: Optional[str] = None,
    ) -> bool:
        """Update source settings.
        
        Returns:
            True if updated, False if not found
        """
        source = self.get_source(source_id)
        if not source:
            return False
        
        if name is not None:
            source.name = name
        if max_items is not None:
            source.max_items = max_items
        if filter_query is not None:
            source.filter_query = filter_query
        
        self._save_sources()
        return True
    
    def mark_processed(self, source_id: str) -> bool:
        """Mark a source as processed."""
        source = self.get_source(source_id)
        if source:
            source.last_processed = datetime.now().isoformat()
            self._save_sources()
            return True
        return False
    
    def clear_all(self) -> int:
        """Clear all sources.
        
        Returns:
            Number of sources removed
        """
        count = len(self.sources)
        self.sources = []
        self._save_sources()
        return count


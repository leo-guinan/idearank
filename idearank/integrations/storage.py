"""SQLite storage for IdeaRank pipeline data."""

from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
import json

logger = logging.getLogger(__name__)

try:
    from sqlalchemy import (
        create_engine,
        Column,
        Integer,
        String,
        Float,
        DateTime,
        Text,
        Boolean,
        JSON,
    )
    from sqlalchemy.orm import declarative_base, sessionmaker
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    logger.warning("SQLAlchemy not installed. Install with: pip install -e '.[pipeline]'")

if SQLALCHEMY_AVAILABLE:
    Base = declarative_base()
    
    class ContentItemRecord(Base):
        """Content item metadata and analytics."""
        __tablename__ = 'content_items'
        
        id = Column(String, primary_key=True)  # content_item_id
        content_source_id = Column(String, index=True)
        title = Column(String)
        description = Column(Text)
        body = Column(Text, nullable=True)  # Main content (transcript, post text, tweet, etc.)
        body_source = Column(String)  # "youtube", "gladia", "ghost", "twitter", "none"
        
        published_at = Column(DateTime)
        fetched_at = Column(DateTime, default=datetime.utcnow)
        
        # Analytics
        view_count = Column(Integer)
        like_count = Column(Integer, default=0)
        comment_count = Column(Integer, default=0)
        duration_seconds = Column(Integer)  # For videos/audio content
        
        # Computed impressions (estimate: views * 5)
        impression_count = Column(Integer)
        watch_time_seconds = Column(Float)
        avg_view_duration = Column(Float)
        
        # Trust signals
        has_citations = Column(Boolean, default=False)
        citation_count = Column(Integer, default=0)
        source_diversity_score = Column(Float, default=0.0)
        correction_count = Column(Integer, default=0)
        
        # Tags as JSON
        tags = Column(JSON)
    
    class ContentSourceRecord(Base):
        """Content source metadata."""
        __tablename__ = 'content_sources'
        
        id = Column(String, primary_key=True)  # content_source_id
        name = Column(String)
        description = Column(Text)
        created_at = Column(DateTime, default=datetime.utcnow)
        
        subscriber_count = Column(Integer, default=0)
        total_views = Column(Integer, default=0)
    
    class IdeaRankScoreRecord(Base):
        """IdeaRank scores for content items."""
        __tablename__ = 'idearank_scores'
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        content_item_id = Column(String, index=True)
        content_source_id = Column(String, index=True)
        
        computed_at = Column(DateTime, default=datetime.utcnow)
        
        # Overall score
        score = Column(Float)
        passes_gates = Column(Boolean)
        
        # Factor scores
        uniqueness_score = Column(Float)
        cohesion_score = Column(Float)
        learning_score = Column(Float)
        quality_score = Column(Float)
        trust_score = Column(Float)
        
        # Factor components (as JSON for debugging)
        uniqueness_components = Column(JSON)
        cohesion_components = Column(JSON)
        learning_components = Column(JSON)
        quality_components = Column(JSON)
        trust_components = Column(JSON)
        
        # Configuration used
        weights = Column(JSON)
    
    class ContentSourceRankScoreRecord(Base):
        """Content source-level IdeaRank scores."""
        __tablename__ = 'content_source_rank_scores'
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        content_source_id = Column(String, index=True)
        
        computed_at = Column(DateTime, default=datetime.utcnow)
        
        score = Column(Float)
        mean_content_score = Column(Float)
        aul_bonus = Column(Float)
        
        content_count = Column(Integer)
        window_days = Column(Integer)
        crystallization_detected = Column(Boolean)


class SQLiteStorage:
    """SQLite storage manager for IdeaRank pipeline."""
    
    def __init__(self, db_path: str = "idearank.db"):
        """Initialize SQLite storage.
        
        Args:
            db_path: Path to SQLite database file
        """
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError(
                "SQLAlchemy not installed. "
                "Install with: pip install -e '.[pipeline]'"
            )
        
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        logger.info(f"Initialized SQLite storage at {db_path}")
    
    def get_content_body(self, content_item_id: str) -> Optional[tuple[str, str]]:
        """Get cached content body from database.
        
        Returns:
            (body_text, source) or None if not found
        """
        if not SQLALCHEMY_AVAILABLE:
            return None
        
        record = self.session.query(ContentItemRecord).filter_by(id=content_item_id).first()
        if record and record.body:
            logger.info(f"Found cached content for {content_item_id} (source: {record.body_source})")
            return record.body, record.body_source
        
        return None
    
    def save_content_item(self, content_item_data: Any, source_data: Optional[Any] = None) -> None:
        """Save content item to database.
        
        Args:
            content_item_data: ContentItem object from idearank.models
            source_data: Optional source-specific data for additional info
        """
        # Check if content item already exists
        existing = self.session.query(ContentItemRecord).filter_by(id=content_item_data.id).first()
        
        if existing:
            # Update existing record
            existing.title = content_item_data.title
            existing.description = content_item_data.description
            existing.body = content_item_data.body
            existing.view_count = content_item_data.view_count
            existing.like_count = getattr(content_item_data, 'like_count', 0)
            existing.comment_count = getattr(content_item_data, 'comment_count', 0)
            existing.fetched_at = datetime.utcnow()
        else:
            # Create new record
            record = ContentItemRecord(
                id=content_item_data.id,
                content_source_id=content_item_data.content_source_id,
                title=content_item_data.title,
                description=content_item_data.description,
                body=content_item_data.body or "",
                body_source=getattr(source_data, 'body_source', 'none') if source_data else 'none',
                published_at=content_item_data.published_at,
                view_count=content_item_data.view_count,
                like_count=getattr(content_item_data, 'like_count', 0),
                comment_count=getattr(content_item_data, 'comment_count', 0),
                duration_seconds=int(content_item_data.content_duration),
                impression_count=content_item_data.impression_count,
                watch_time_seconds=content_item_data.watch_time_seconds,
                avg_view_duration=content_item_data.avg_view_duration,
                has_citations=content_item_data.has_citations,
                citation_count=content_item_data.citation_count,
                source_diversity_score=content_item_data.source_diversity_score,
                correction_count=content_item_data.correction_count,
                tags=content_item_data.tags,
            )
            self.session.add(record)
        
        self.session.commit()
        logger.debug(f"Saved content item: {content_item_data.id}")
    
    def save_content_source(self, content_source: Any) -> None:
        """Save content source to database."""
        existing = self.session.query(ContentSourceRecord).filter_by(id=content_source.id).first()
        
        if existing:
            existing.name = content_source.name
            existing.description = content_source.description
        else:
            record = ContentSourceRecord(
                id=content_source.id,
                name=content_source.name,
                description=content_source.description,
                subscriber_count=content_source.subscriber_count,
                total_views=content_source.total_views,
            )
            self.session.add(record)
        
        self.session.commit()
        logger.debug(f"Saved content source: {content_source.id}")
    
    def save_content_score(
        self,
        content_item_id: str,
        content_source_id: str,
        score_result: Any,
    ) -> None:
        """Save IdeaRank score for a content item.
        
        Args:
            content_item_id: Content item ID
            content_source_id: Content source ID
            score_result: IdeaRankScore object
        """
        record = IdeaRankScoreRecord(
            content_item_id=content_item_id,
            content_source_id=content_source_id,
            score=score_result.score,
            passes_gates=score_result.passes_gates,
            uniqueness_score=score_result.uniqueness.score,
            cohesion_score=score_result.cohesion.score,
            learning_score=score_result.learning.score,
            quality_score=score_result.quality.score,
            trust_score=score_result.trust.score,
            uniqueness_components=score_result.uniqueness.components,
            cohesion_components=score_result.cohesion.components,
            learning_components=score_result.learning.components,
            quality_components=score_result.quality.components,
            trust_components=score_result.trust.components,
            weights=score_result.weights_used,
        )
        
        self.session.add(record)
        self.session.commit()
        logger.debug(f"Saved IdeaRank score for content item: {content_item_id}")
    
    def save_source_score(
        self,
        content_source_id: str,
        score_result: Any,
    ) -> None:
        """Save content source-level score."""
        record = ContentSourceRankScoreRecord(
            content_source_id=content_source_id,
            score=score_result.score,
            mean_content_score=score_result.mean_content_score,
            aul_bonus=score_result.aul_bonus,
            content_count=score_result.content_count,
            window_days=score_result.window_days,
            crystallization_detected=score_result.crystallization_detected,
        )
        
        self.session.add(record)
        self.session.commit()
        logger.debug(f"Saved content source score for: {content_source_id}")
    
    def get_content_item(self, content_item_id: str) -> Optional[ContentItemRecord]:
        """Get content item by ID."""
        return self.session.query(ContentItemRecord).filter_by(id=content_item_id).first()
    
    def get_source_content_items(self, content_source_id: str) -> List[ContentItemRecord]:
        """Get all content items for a source."""
        return self.session.query(ContentItemRecord).filter_by(content_source_id=content_source_id).all()
    
    def get_latest_scores(self, content_source_id: str, limit: int = 10) -> List[IdeaRankScoreRecord]:
        """Get latest IdeaRank scores for a content source."""
        return (
            self.session.query(IdeaRankScoreRecord)
            .filter_by(content_source_id=content_source_id)
            .order_by(IdeaRankScoreRecord.computed_at.desc())
            .limit(limit)
            .all()
        )
    
    def close(self):
        """Close database session."""
        self.session.close()


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
    
    class VideoRecord(Base):
        """Video metadata and analytics."""
        __tablename__ = 'videos'
        
        id = Column(String, primary_key=True)  # video_id
        channel_id = Column(String, index=True)
        title = Column(String)
        description = Column(Text)
        transcript = Column(Text, nullable=True)
        transcript_source = Column(String)  # "youtube", "gladia", "none"
        
        published_at = Column(DateTime)
        fetched_at = Column(DateTime, default=datetime.utcnow)
        
        # Analytics
        view_count = Column(Integer)
        like_count = Column(Integer, default=0)
        comment_count = Column(Integer, default=0)
        duration_seconds = Column(Integer)
        
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
    
    class ChannelRecord(Base):
        """Channel metadata."""
        __tablename__ = 'channels'
        
        id = Column(String, primary_key=True)  # channel_id
        name = Column(String)
        description = Column(Text)
        created_at = Column(DateTime, default=datetime.utcnow)
        
        subscriber_count = Column(Integer, default=0)
        total_views = Column(Integer, default=0)
    
    class IdeaRankScoreRecord(Base):
        """IdeaRank scores for videos."""
        __tablename__ = 'idearank_scores'
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        video_id = Column(String, index=True)
        channel_id = Column(String, index=True)
        
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
    
    class ChannelRankScoreRecord(Base):
        """Channel-level IdeaRank scores."""
        __tablename__ = 'channel_rank_scores'
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        channel_id = Column(String, index=True)
        
        computed_at = Column(DateTime, default=datetime.utcnow)
        
        score = Column(Float)
        mean_video_score = Column(Float)
        aul_bonus = Column(Float)
        
        video_count = Column(Integer)
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
    
    def save_video(self, video_data: Any, youtube_data: Optional[Any] = None) -> None:
        """Save video to database.
        
        Args:
            video_data: Video object from idearank.models
            youtube_data: Optional YouTubeVideoData for additional info
        """
        # Check if video already exists
        existing = self.session.query(VideoRecord).filter_by(id=video_data.id).first()
        
        if existing:
            # Update existing record
            existing.title = video_data.title
            existing.description = video_data.description
            existing.transcript = video_data.transcript
            existing.view_count = video_data.view_count
            existing.like_count = getattr(video_data, 'like_count', 0)
            existing.comment_count = getattr(video_data, 'comment_count', 0)
            existing.fetched_at = datetime.utcnow()
        else:
            # Create new record
            record = VideoRecord(
                id=video_data.id,
                channel_id=video_data.channel_id,
                title=video_data.title,
                description=video_data.description,
                transcript=video_data.transcript or "",
                transcript_source=getattr(youtube_data, 'transcript_source', 'none') if youtube_data else 'none',
                published_at=video_data.published_at,
                view_count=video_data.view_count,
                like_count=getattr(video_data, 'like_count', 0),
                comment_count=getattr(video_data, 'comment_count', 0),
                duration_seconds=int(video_data.video_duration),
                impression_count=video_data.impression_count,
                watch_time_seconds=video_data.watch_time_seconds,
                avg_view_duration=video_data.avg_view_duration,
                has_citations=video_data.has_citations,
                citation_count=video_data.citation_count,
                source_diversity_score=video_data.source_diversity_score,
                correction_count=video_data.correction_count,
                tags=video_data.tags,
            )
            self.session.add(record)
        
        self.session.commit()
        logger.debug(f"Saved video: {video_data.id}")
    
    def save_channel(self, channel: Any) -> None:
        """Save channel to database."""
        existing = self.session.query(ChannelRecord).filter_by(id=channel.id).first()
        
        if existing:
            existing.name = channel.name
            existing.description = channel.description
        else:
            record = ChannelRecord(
                id=channel.id,
                name=channel.name,
                description=channel.description,
                subscriber_count=channel.subscriber_count,
                total_views=channel.total_views,
            )
            self.session.add(record)
        
        self.session.commit()
        logger.debug(f"Saved channel: {channel.id}")
    
    def save_video_score(
        self,
        video_id: str,
        channel_id: str,
        score_result: Any,
    ) -> None:
        """Save IdeaRank score for a video.
        
        Args:
            video_id: Video ID
            channel_id: Channel ID
            score_result: IdeaRankScore object
        """
        record = IdeaRankScoreRecord(
            video_id=video_id,
            channel_id=channel_id,
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
        logger.debug(f"Saved IdeaRank score for video: {video_id}")
    
    def save_channel_score(
        self,
        channel_id: str,
        score_result: Any,
    ) -> None:
        """Save channel-level score."""
        record = ChannelRankScoreRecord(
            channel_id=channel_id,
            score=score_result.score,
            mean_video_score=score_result.mean_video_score,
            aul_bonus=score_result.aul_bonus,
            video_count=score_result.video_count,
            window_days=score_result.window_days,
            crystallization_detected=score_result.crystallization_detected,
        )
        
        self.session.add(record)
        self.session.commit()
        logger.debug(f"Saved channel score for: {channel_id}")
    
    def get_video(self, video_id: str) -> Optional[VideoRecord]:
        """Get video by ID."""
        return self.session.query(VideoRecord).filter_by(id=video_id).first()
    
    def get_channel_videos(self, channel_id: str) -> List[VideoRecord]:
        """Get all videos for a channel."""
        return self.session.query(VideoRecord).filter_by(channel_id=channel_id).all()
    
    def get_latest_scores(self, channel_id: str, limit: int = 10) -> List[IdeaRankScoreRecord]:
        """Get latest IdeaRank scores for a channel."""
        return (
            self.session.query(IdeaRankScoreRecord)
            .filter_by(channel_id=channel_id)
            .order_by(IdeaRankScoreRecord.computed_at.desc())
            .limit(limit)
            .all()
        )
    
    def close(self):
        """Close database session."""
        self.session.close()


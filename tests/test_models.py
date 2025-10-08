"""Tests for core data models."""

import numpy as np
from datetime import datetime, timedelta

from idearank.models import Embedding, TopicMixture, Video, Channel


def test_embedding_cosine_similarity():
    """Test embedding similarity computation."""
    # Identical embeddings
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    e1 = Embedding(vector=v1, model="test")
    e2 = Embedding(vector=v1.copy(), model="test")
    
    assert abs(e1.cosine_similarity(e2) - 1.0) < 1e-6
    
    # Orthogonal embeddings
    v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    e3 = Embedding(vector=v2, model="test")
    
    assert abs(e1.cosine_similarity(e3)) < 1e-6


def test_topic_mixture_entropy():
    """Test topic mixture entropy calculation."""
    # Uniform distribution (high entropy)
    uniform = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    tm1 = TopicMixture(distribution=uniform, topic_model="test")
    
    # Peaked distribution (low entropy)
    peaked = np.array([0.97, 0.01, 0.01, 0.01], dtype=np.float32)
    tm2 = TopicMixture(distribution=peaked, topic_model="test")
    
    assert tm1.entropy() > tm2.entropy()


def test_video_metrics():
    """Test video analytics metrics."""
    video = Video(
        id="test_video",
        channel_id="test_channel",
        title="Test",
        description="Test",
        transcript="Test",
        published_at=datetime.utcnow(),
        snapshot_time=datetime.utcnow(),
        view_count=1000,
        impression_count=5000,
        watch_time_seconds=100000.0,
        avg_view_duration=200.0,
        video_duration=300.0,
    )
    
    # WTPI
    assert video.watch_time_per_impression == 100000.0 / 5000
    
    # Completion rate
    assert abs(video.completion_rate - (200.0 / 300.0)) < 1e-6


def test_channel_video_window():
    """Test channel video windowing."""
    base_time = datetime(2024, 1, 1)
    videos = [
        Video(
            id=f"video_{i}",
            channel_id="test",
            title=f"Video {i}",
            description="",
            transcript="",
            published_at=base_time + timedelta(days=i * 30),
            snapshot_time=datetime.utcnow(),
        )
        for i in range(12)  # 12 months of videos
    ]
    
    channel = Channel(
        id="test",
        name="Test",
        description="",
        created_at=base_time,
        videos=videos,
    )
    
    # Get videos in 180-day window ending at month 11
    end_time = base_time + timedelta(days=11 * 30)
    windowed = channel.get_videos_in_window(end_time, window_days=180)
    
    # Should get roughly 6 months of videos
    assert 5 <= len(windowed) <= 7


def test_channel_prior_video():
    """Test finding prior video."""
    base_time = datetime(2024, 1, 1)
    videos = [
        Video(
            id=f"video_{i}",
            channel_id="test",
            title=f"Video {i}",
            description="",
            transcript="",
            published_at=base_time + timedelta(days=i * 10),
            snapshot_time=datetime.utcnow(),
        )
        for i in range(10)
    ]
    
    channel = Channel(
        id="test",
        name="Test",
        description="",
        created_at=base_time,
        videos=videos,
    )
    
    # Get prior to video 5
    prior = channel.get_prior_video(videos[5])
    assert prior is not None
    assert prior.id == "video_4"
    
    # First video has no prior
    prior_first = channel.get_prior_video(videos[0])
    assert prior_first is None


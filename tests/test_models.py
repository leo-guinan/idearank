"""Tests for core data models."""

import numpy as np
from datetime import datetime, timedelta

from idearank.models import Embedding, TopicMixture, ContentItem, ContentSource


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


def test_content_item_metrics():
    """Test content item analytics metrics."""
    item = ContentItem(
        id="test_item",
        content_source_id="test_source",
        title="Test",
        description="Test",
        body="Test",
        published_at=datetime.utcnow(),
        captured_at=datetime.utcnow(),
        view_count=1000,
        impression_count=5000,
        watch_time_seconds=100000.0,
        avg_view_duration=200.0,
        content_duration=300.0,
    )
    
    # WTPI
    assert item.watch_time_per_impression == 100000.0 / 5000
    
    # Completion rate
    assert abs(item.completion_rate - (200.0 / 300.0)) < 1e-6


def test_source_content_window():
    """Test content source windowing."""
    base_time = datetime(2024, 1, 1)
    content_items = [
        ContentItem(
            id=f"item_{i}",
            content_source_id="test",
            title=f"Content {i}",
            description="",
            body="",
            published_at=base_time + timedelta(days=i * 30),
            captured_at=datetime.utcnow(),
        )
        for i in range(12)  # 12 months of content
    ]
    
    content_source = ContentSource(
        id="test",
        name="Test",
        description="",
        created_at=base_time,
        content_items=content_items,
    )
    
    # Get content in 180-day window ending at month 11
    end_time = base_time + timedelta(days=11 * 30)
    windowed = content_source.get_content_in_window(end_time, window_days=180)
    
    # Should get roughly 6 months of content
    assert 5 <= len(windowed) <= 7


def test_source_prior_content():
    """Test finding prior content."""
    base_time = datetime(2024, 1, 1)
    content_items = [
        ContentItem(
            id=f"item_{i}",
            content_source_id="test",
            title=f"Content {i}",
            description="",
            body="",
            published_at=base_time + timedelta(days=i * 10),
            captured_at=datetime.utcnow(),
        )
        for i in range(10)
    ]
    
    content_source = ContentSource(
        id="test",
        name="Test",
        description="",
        created_at=base_time,
        content_items=content_items,
    )
    
    # Get prior to item 5
    prior = content_source.get_prior_content(content_items[5])
    assert prior is not None
    assert prior.id == "item_4"
    
    # First item has no prior
    prior_first = content_source.get_prior_content(content_items[0])
    assert prior_first is None


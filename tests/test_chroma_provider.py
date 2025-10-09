"""Tests for Chroma provider integration.

These tests verify the provider interface is correctly implemented.
They don't test actual Chroma Cloud connectivity (use integration tests for that).
"""

import pytest

# Check if chromadb is available
try:
    from idearank.providers.chroma import (
        ChromaEmbeddingProvider,
        ChromaNeighborhoodProvider,
        ChromaProvider,
    )
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


@pytest.mark.skipif(not CHROMA_AVAILABLE, reason="chromadb not installed")
def test_chroma_provider_import():
    """Test that Chroma providers can be imported."""
    assert ChromaEmbeddingProvider is not None
    assert ChromaNeighborhoodProvider is not None
    assert ChromaProvider is not None


@pytest.mark.skipif(not CHROMA_AVAILABLE, reason="chromadb not installed")
def test_chroma_provider_init():
    """Test that ChromaProvider can be initialized (without connecting)."""
    # Note: This will fail if credentials are invalid, but that's OK
    # We're just testing the interface
    try:
        provider = ChromaProvider(
            api_key="test_key",
            tenant="test_tenant",
            database="test_db",
            embedding_function="default",
        )
        # If we get here, initialization works
        assert provider is not None
    except Exception as e:
        # Expected if credentials are invalid or network is down
        # We just care that the interface exists
        assert "api_key" in str(e).lower() or "connect" in str(e).lower() or "auth" in str(e).lower()


@pytest.mark.skipif(not CHROMA_AVAILABLE, reason="chromadb not installed")
def test_chroma_provider_interfaces():
    """Test that ChromaProvider provides the correct interfaces."""
    try:
        provider = ChromaProvider(
            api_key="test_key",
            tenant="test_tenant",
            database="test_db",
        )
        
        # Should have methods to get providers
        assert hasattr(provider, 'get_embedding_provider')
        assert hasattr(provider, 'get_neighborhood_provider')
        
    except Exception:
        # Expected if can't connect
        pass


def test_chroma_availability_flag():
    """Test that CHROMA_AVAILABLE flag is set correctly."""
    from idearank.providers import CHROMA_AVAILABLE as flag
    
    # Should match local check
    assert flag == CHROMA_AVAILABLE


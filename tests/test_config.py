"""Tests for configuration system."""

import pytest
from idearank.config import IdeaRankConfig, FactorWeights


def test_default_config():
    """Test default configuration."""
    config = IdeaRankConfig.default()
    config.validate()
    
    # Check default weights
    assert config.weights.uniqueness == 0.35
    assert config.weights.cohesion == 0.20
    assert config.weights.learning == 0.25
    assert config.weights.quality == 0.15
    assert config.weights.trust == 0.05


def test_experimental_config():
    """Test experimental configuration."""
    config = IdeaRankConfig.experimental()
    config.validate()
    
    # Should have smaller k values for faster experimentation
    assert config.uniqueness.k_global < 50
    assert config.cohesion.k_intra < 15


def test_invalid_weights():
    """Test validation catches invalid weights."""
    with pytest.raises(ValueError):
        weights = FactorWeights(
            uniqueness=-0.1,  # Negative!
            cohesion=0.2,
            learning=0.2,
            quality=0.2,
            trust=0.1,
        )
        weights.validate()


def test_all_zero_weights():
    """Test validation catches all-zero weights."""
    with pytest.raises(ValueError):
        weights = FactorWeights(
            uniqueness=0.0,
            cohesion=0.0,
            learning=0.0,
            quality=0.0,
            trust=0.0,
        )
        weights.validate()


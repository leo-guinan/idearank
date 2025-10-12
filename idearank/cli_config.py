"""Configuration management for IdeaRank CLI.

Stores API keys and settings in user's home directory.
"""

import json
import os
from pathlib import Path
from typing import Optional, Literal


class CLIConfig:
    """Manages CLI configuration and API keys."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize config manager.
        
        Args:
            config_path: Path to config file. Defaults to ~/.idearank/config.json
        """
        if config_path is None:
            config_dir = Path.home() / ".idearank"
            config_dir.mkdir(exist_ok=True)
            config_path = config_dir / "config.json"
        
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> dict:
        """Load config from file or create default."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            return {
            "youtube_api_key": None,
            "whisper_model": "small",  # tiny, base, small, medium, large
            "whisper_device": "auto",  # cpu, cuda, auto
            "chroma_mode": "local",  # "local" or "cloud"
            "chroma_cloud_api_key": None,
            "chroma_cloud_tenant": None,
            "chroma_cloud_database": None,
        }
    
    def _save_config(self) -> None:
        """Save config to file."""
        self.config_path.parent.mkdir(exist_ok=True, parents=True)
        with open(self.config_path, 'w') as f:
            json.dump(self._config, f, indent=2)
    
    # YouTube API Key
    def set_youtube_api_key(self, api_key: str) -> None:
        """Set YouTube API key."""
        self._config["youtube_api_key"] = api_key
        self._save_config()
    
    def get_youtube_api_key(self) -> Optional[str]:
        """Get YouTube API key."""
        return self._config.get("youtube_api_key")
    
    # Whisper Configuration
    def set_whisper_model(self, model: str) -> None:
        """Set Whisper model size (tiny, base, small, medium, large)."""
        valid_models = ["tiny", "base", "small", "medium", "large"]
        if model not in valid_models:
            raise ValueError(f"Model must be one of: {', '.join(valid_models)}")
        self._config["whisper_model"] = model
        self._save_config()
    
    def get_whisper_model(self) -> str:
        """Get Whisper model size."""
        return self._config.get("whisper_model", "small")
    
    def set_whisper_device(self, device: str) -> None:
        """Set Whisper device (cpu, cuda, auto)."""
        valid_devices = ["cpu", "cuda", "auto"]
        if device not in valid_devices:
            raise ValueError(f"Device must be one of: {', '.join(valid_devices)}")
        self._config["whisper_device"] = device
        self._save_config()
    
    def get_whisper_device(self) -> str:
        """Get Whisper device."""
        return self._config.get("whisper_device", "auto")
    
    # Chroma Mode
    def set_chroma_mode(self, mode: Literal["local", "cloud"]) -> None:
        """Set Chroma storage mode."""
        if mode not in ["local", "cloud"]:
            raise ValueError("Mode must be 'local' or 'cloud'")
        self._config["chroma_mode"] = mode
        self._save_config()
    
    def get_chroma_mode(self) -> str:
        """Get Chroma storage mode."""
        return self._config.get("chroma_mode", "local")
    
    # Chroma Cloud Settings
    def set_chroma_cloud(
        self,
        api_key: str,
        tenant: str = "default_tenant",
        database: str = "default_database",
    ) -> None:
        """Set Chroma Cloud credentials.
        
        Args:
            api_key: Chroma Cloud API key
            tenant: Chroma Cloud tenant (default: "default_tenant")
            database: Chroma Cloud database (default: "default_database")
        """
        self._config["chroma_cloud_api_key"] = api_key
        self._config["chroma_cloud_tenant"] = tenant
        self._config["chroma_cloud_database"] = database
        self._save_config()
    
    def get_chroma_cloud_config(self) -> dict:
        """Get Chroma Cloud configuration."""
        return {
            "api_key": self._config.get("chroma_cloud_api_key"),
            "tenant": self._config.get("chroma_cloud_tenant", "default_tenant"),
            "database": self._config.get("chroma_cloud_database", "default_database"),
        }
    
    def is_configured(self) -> tuple[bool, list[str]]:
        """Check if minimum configuration is present.
        
        Returns:
            (is_configured, missing_keys)
        """
        missing = []
        
        if not self._config.get("youtube_api_key"):
            missing.append("youtube_api_key")
        
        if self._config.get("chroma_mode") == "cloud":
            if not self._config.get("chroma_cloud_api_key"):
                missing.append("chroma_cloud_api_key")
        
        return len(missing) == 0, missing
    
    def show(self) -> dict:
        """Get all config (with masked sensitive values)."""
        config = self._config.copy()
        
        # Mask API keys
        if config.get("youtube_api_key"):
            config["youtube_api_key"] = self._mask_key(config["youtube_api_key"])
        if config.get("chroma_cloud_api_key"):
            config["chroma_cloud_api_key"] = self._mask_key(config["chroma_cloud_api_key"])
        
        return config
    
    @staticmethod
    def _mask_key(key: str) -> str:
        """Mask API key for display."""
        if not key:
            return None
        if len(key) <= 8:
            return "***"
        return f"{key[:4]}...{key[-4:]}"
    
    def clear(self) -> None:
        """Clear all configuration."""
        self._config = {
            "youtube_api_key": None,
            "whisper_model": "small",
            "whisper_device": "auto",
            "chroma_mode": "local",
            "chroma_cloud_api_key": None,
            "chroma_cloud_tenant": None,
            "chroma_cloud_database": None,
        }
        self._save_config()


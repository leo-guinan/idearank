"""Twitter integration for local archive JSON files.

Processes Twitter archive JSON files for IdeaRank analysis.
"""

import json
import logging
import zipfile
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TwitterPost:
    """Represents a single Twitter post."""
    
    id: str
    text: str
    created_at: datetime
    author: str
    retweet_count: int = 0
    favorite_count: int = 0
    reply_count: int = 0
    is_retweet: bool = False
    is_quote_tweet: bool = False
    media_urls: List[str] = None
    hashtags: List[str] = None
    mentions: List[str] = None
    
    def __post_init__(self):
        if self.media_urls is None:
            self.media_urls = []
        if self.hashtags is None:
            self.hashtags = []
        if self.mentions is None:
            self.mentions = []


@dataclass
class TwitterArchive:
    """Represents a Twitter archive."""
    
    username: str
    posts: List[TwitterPost]
    total_posts: int
    date_range: tuple[datetime, datetime]
    source_file: str
    loaded_at: datetime


@dataclass
class TwitterUser:
    """Represents a Twitter user."""
    
    username: str
    display_name: str
    description: str
    follower_count: int
    following_count: int
    tweet_count: int


class TwitterArchiveParser:
    """Parser for Twitter archive JSON files."""
    
    def __init__(self):
        """Initialize the parser."""
        self.logger = logging.getLogger(__name__)
    
    def load_archive(self, file_path: str) -> TwitterArchive:
        """Load a Twitter archive from a JSON file or ZIP archive.
        
        Supports multiple formats:
        - Official Twitter archive format (tweets.js or tweet.js)
        - Community archive format (archive.json)
        - Simple JSON array of tweets
        - ZIP archives containing JSON files
        
        Args:
            file_path: Path to the JSON file or ZIP archive
            
        Returns:
            TwitterArchive with parsed posts
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Archive file not found: {file_path}")
        
        self.logger.info(f"Loading Twitter archive from: {file_path}")
        
        # Handle ZIP files
        if file_path.suffix.lower() == '.zip':
            data = self._load_from_zip(file_path)
        else:
            # Handle JSON files
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON file: {e}")
        
        # Detect format and parse
        posts, username = self._parse_archive_format(data, file_path)
        
        if not posts:
            raise ValueError("No posts found in archive")
        
        # Calculate date range
        dates = [p.created_at for p in posts if p.created_at]
        date_range = (min(dates), max(dates)) if dates else (datetime.now(), datetime.now())
        
        archive = TwitterArchive(
            username=username,
            posts=posts,
            total_posts=len(posts),
            date_range=date_range,
            source_file=str(file_path),
            loaded_at=datetime.now(),
        )
        
        self.logger.info(f"Loaded {len(posts)} posts from @{username}")
        return archive
    
    def _load_from_zip(self, zip_path: Path) -> Any:
        """Load Twitter data from a ZIP archive.
        
        Args:
            zip_path: Path to ZIP file
            
        Returns:
            Parsed JSON data
        """
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_file:
                # Look for common Twitter archive files
                possible_files = [
                    'tweets.js',
                    'tweet.js', 
                    'archive.json',
                    'data/tweets.js',
                    'data/tweet.js',
                    'data/archive.json',
                    'Twitter Archive/data/tweets.js',
                    'Twitter Archive/data/tweet.js'
                ]
                
                json_file = None
                for possible_file in possible_files:
                    if possible_file in zip_file.namelist():
                        json_file = possible_file
                        break
                
                if not json_file:
                    # List available files for debugging
                    available_files = zip_file.namelist()
                    raise ValueError(f"No Twitter JSON file found in ZIP. Available files: {available_files[:10]}")
                
                # Read the JSON file from ZIP
                with zip_file.open(json_file) as f:
                    content = f.read()
                    
                    # Try different encodings
                    for encoding in ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']:
                        try:
                            text = content.decode(encoding)
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        raise ValueError("Could not decode ZIP file content with any supported encoding")
                    
                    # Parse JSON - handle both pure JSON and JavaScript-wrapped JSON
                    try:
                        # Try parsing as pure JSON first
                        data = json.loads(text)
                        self.logger.info(f"Successfully loaded {json_file} from ZIP archive")
                        return data
                    except json.JSONDecodeError:
                        # Try parsing as JavaScript-wrapped JSON (Twitter archive format)
                        try:
                            # Remove JavaScript wrapper and extract JSON
                            if text.strip().startswith('window.YTD.tweets.part0 = '):
                                # Extract JSON array from JavaScript assignment
                                json_start = text.find('[')
                                json_end = text.rfind(']') + 1
                                if json_start != -1 and json_end != -1:
                                    json_content = text[json_start:json_end]
                                    data = json.loads(json_content)
                                    self.logger.info(f"Successfully loaded {json_file} from ZIP archive (JavaScript format)")
                                    return data
                            
                            # Try other JavaScript patterns
                            if ' = [' in text and text.strip().endswith(';'):
                                # Find the array content
                                array_start = text.find('[')
                                array_end = text.rfind(']') + 1
                                if array_start != -1 and array_end != -1:
                                    array_content = text[array_start:array_end]
                                    data = json.loads(array_content)
                                    self.logger.info(f"Successfully loaded {json_file} from ZIP archive (JavaScript array format)")
                                    return data
                            
                            raise ValueError(f"Could not extract JSON from JavaScript file {json_file}")
                        except (json.JSONDecodeError, ValueError) as e:
                            raise ValueError(f"Invalid JSON in ZIP file {json_file}: {e}")
                        
        except zipfile.BadZipFile:
            raise ValueError("Invalid ZIP file format")
        except Exception as e:
            raise ValueError(f"Error reading ZIP file: {e}")
    
    def _parse_archive_format(self, data: Any, file_path: Path) -> tuple[List[TwitterPost], str]:
        """Detect and parse the archive format.
        
        Returns:
            Tuple of (posts, username)
        """
        # Format 1: Array of tweets directly
        if isinstance(data, list):
            return self._parse_tweet_array(data), self._extract_username_from_tweets(data)
        
        # Format 2: Official Twitter archive (has specific structure)
        if isinstance(data, dict):
            # Check for official archive format
            if 'tweets' in data or 'tweet' in data:
                tweets = data.get('tweets', data.get('tweet', []))
                username = data.get('user', {}).get('screen_name', 'unknown')
                return self._parse_tweet_array(tweets), username
            
            # Format 3: Community archive format (has metadata)
            if 'data' in data and 'user' in data:
                username = data['user'].get('username', 'unknown')
                tweets = data['data']
                return self._parse_tweet_array(tweets), username
            
            # Format 4: Single level with posts key
            if 'posts' in data:
                username = data.get('username', 'unknown')
                return self._parse_tweet_array(data['posts']), username
        
        # Couldn't detect format
        raise ValueError("Unrecognized Twitter archive format")
    
    def _parse_tweet_array(self, tweets: List[Dict]) -> List[TwitterPost]:
        """Parse an array of tweet objects.
        
        Args:
            tweets: List of tweet dictionaries
            
        Returns:
            List of TwitterPost objects
        """
        posts = []
        
        for tweet_data in tweets:
            try:
                # Handle nested 'tweet' key (official format)
                if 'tweet' in tweet_data:
                    tweet_data = tweet_data['tweet']
                
                post = self._parse_single_tweet(tweet_data)
                if post:
                    posts.append(post)
            except Exception as e:
                self.logger.warning(f"Failed to parse tweet: {e}")
                continue
        
        return posts
    
    def _parse_single_tweet(self, data: Dict) -> Optional[TwitterPost]:
        """Parse a single tweet object.
        
        Args:
            data: Tweet dictionary
            
        Returns:
            TwitterPost or None if parsing fails
        """
        # Extract ID
        tweet_id = str(data.get('id', data.get('id_str', '')))
        if not tweet_id:
            return None
        
        # Extract text (various possible keys)
        text = data.get('full_text', data.get('text', data.get('content', '')))
        if not text:
            return None
        
        # Parse created_at
        created_at = self._parse_date(data.get('created_at'))
        
        # Extract author
        author = data.get('user', {}).get('screen_name', data.get('author', 'unknown'))
        
        # Extract engagement metrics (ensure they're integers)
        retweet_count = int(data.get('retweet_count', 0) or 0)
        favorite_count = int(data.get('favorite_count', data.get('like_count', 0)) or 0)
        reply_count = int(data.get('reply_count', 0) or 0)
        
        # Check if retweet or quote
        is_retweet = 'retweeted_status' in data or text.startswith('RT @')
        is_quote_tweet = 'quoted_status' in data
        
        # Extract entities
        entities = data.get('entities', {})
        hashtags = [h.get('text', '') for h in entities.get('hashtags', [])]
        mentions = [m.get('screen_name', '') for m in entities.get('user_mentions', [])]
        
        # Extract media
        media = entities.get('media', [])
        media_urls = [m.get('media_url', '') for m in media if m.get('media_url')]
        
        return TwitterPost(
            id=tweet_id,
            text=text,
            created_at=created_at,
            author=author,
            retweet_count=retweet_count,
            favorite_count=favorite_count,
            reply_count=reply_count,
            is_retweet=is_retweet,
            is_quote_tweet=is_quote_tweet,
            media_urls=media_urls,
            hashtags=hashtags,
            mentions=mentions,
        )
    
    def _parse_date(self, date_str: Any) -> datetime:
        """Parse a date string into datetime.
        
        Args:
            date_str: Date string in various formats
            
        Returns:
            datetime object
        """
        if not date_str:
            return datetime.now()
        
        if isinstance(date_str, datetime):
            return date_str
        
        # Try various formats
        formats = [
            '%a %b %d %H:%M:%S %z %Y',  # Twitter API format
            '%Y-%m-%dT%H:%M:%S.%fZ',     # ISO format
            '%Y-%m-%d %H:%M:%S',          # Simple format
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except (ValueError, TypeError):
                continue
        
        # Fallback
        self.logger.warning(f"Could not parse date: {date_str}")
        return datetime.now()
    
    def _extract_username_from_tweets(self, tweets: List[Dict]) -> str:
        """Extract username from tweet array if not provided elsewhere.
        
        Args:
            tweets: List of tweet dictionaries
            
        Returns:
            Username string
        """
        for tweet in tweets:
            if 'tweet' in tweet:
                tweet = tweet['tweet']
            
            user = tweet.get('user', {})
            if user.get('screen_name'):
                return user['screen_name']
            
            if tweet.get('author'):
                return tweet['author']
        
        return 'unknown'


def load_twitter_archive(file_path: str) -> TwitterArchive:
    """Convenience function to load a Twitter archive.
    
    Args:
        file_path: Path to Twitter archive JSON file
        
    Returns:
        TwitterArchive object
    """
    parser = TwitterArchiveParser()
    return parser.load_archive(file_path)

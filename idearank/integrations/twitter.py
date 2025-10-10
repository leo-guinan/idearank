"""Twitter integration via community archive.

Fetches Twitter data from community-archive.org and processes it for IdeaRank.
"""

import requests
import json
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import re

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
    """Represents a Twitter archive from community-archive.org."""
    
    username: str
    posts: List[TwitterPost]
    total_posts: int
    date_range: tuple[datetime, datetime]
    archive_url: str
    download_date: datetime


class CommunityArchiveClient:
    """Client for fetching Twitter data from community-archive.org."""
    
    BASE_URL = "https://www.community-archive.org"
    API_ENDPOINT = "/api/v1"
    
    def __init__(self, timeout: int = 30):
        """Initialize the community archive client.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'IdeaRank/1.0 (Content Analysis Tool)',
            'Accept': 'application/json',
        })
    
    def check_archive_availability(self, username: str) -> Dict[str, Any]:
        """Check if a Twitter archive is available.
        
        Args:
            username: Twitter username (with or without @)
            
        Returns:
            Dict with availability info and metadata
        """
        # Clean username
        clean_username = self._clean_username(username)
        if not clean_username:
            return {
                'available': False,
                'username': username,
                'error': f'Invalid Twitter username format: {username}',
            }
        
        logger.info(f"Checking archive availability for @{clean_username}")
        
        try:
            # Community Archive doesn't have a public API, so we can't programmatically check
            # Instead, we'll provide helpful guidance to the user
            logger.info(f"Checking archive availability for @{clean_username}")
            
            # For now, we assume archives might be available but can't verify programmatically
            # This is a limitation of community-archive.org not providing a public API
            return {
                'available': None,  # Unknown - can't check programmatically
                'username': clean_username,
                'message': f"Can't programmatically verify archive availability for @{clean_username}",
                'archive_url': f"{self.BASE_URL}/search?q={clean_username}",
                'upload_url': f"{self.BASE_URL}/upload",
                'user_directory_url': f"{self.BASE_URL}/user-dir",
                'note': "Community Archive doesn't provide a public API. Check manually at the archive_url.",
            }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for @{clean_username}: {e}")
            return {
                'available': False,
                'username': clean_username,
                'error': f"Request failed: {e}",
            }
    
    def fetch_archive(self, username: str, limit: Optional[int] = None) -> Optional[TwitterArchive]:
        """Fetch a Twitter archive from community-archive.org.
        
        Args:
            username: Twitter username (with or without @)
            limit: Maximum number of posts to fetch (None for all)
            
        Returns:
            TwitterArchive object or None if not available
        """
        clean_username = self._clean_username(username)
        if not clean_username:
            logger.warning(f"Invalid Twitter username format: {username}")
            return None
        
        # Check availability (but we can't verify programmatically)
        availability = self.check_archive_availability(clean_username)
        if availability['available'] is False:
            logger.warning(f"Archive confirmed not available for @{clean_username}")
            return None
        
        # If we can't check programmatically, we'll try to fetch anyway
        # This will likely fail, but gives better error messages
        logger.info(f"Attempting to fetch archive for @{clean_username} (availability unknown)")
        
        logger.info(f"Fetching archive for @{clean_username}...")
        
        try:
            # Fetch posts
            posts = self._fetch_posts(clean_username, limit)
            if not posts:
                logger.warning(f"No posts found for @{clean_username}")
                return None
            
            # Calculate date range
            dates = [post.created_at for post in posts]
            date_range = (min(dates), max(dates))
            
            archive = TwitterArchive(
                username=clean_username,
                posts=posts,
                total_posts=len(posts),
                date_range=date_range,
                archive_url=availability['archive_url'],
                download_date=datetime.now(),
            )
            
            logger.info(f"Successfully fetched {len(posts)} posts for @{clean_username}")
            return archive
            
        except Exception as e:
            logger.error(f"Failed to fetch archive for @{clean_username}: {e}")
            return None
    
    def _fetch_posts(self, username: str, limit: Optional[int] = None) -> List[TwitterPost]:
        """Fetch posts for a username.
        
        Args:
            username: Clean username
            limit: Maximum posts to fetch
            
        Returns:
            List of TwitterPost objects
        """
        posts = []
        page = 1
        page_size = 100  # API page size
        
        while True:
            try:
                url = f"{self.BASE_URL}{self.API_ENDPOINT}/archive/{username}/posts"
                params = {
                    'page': page,
                    'per_page': page_size,
                    'format': 'json',
                }
                
                logger.debug(f"Fetching page {page} for @{username}")
                response = self.session.get(url, params=params, timeout=self.timeout)
                
                if response.status_code != 200:
                    logger.warning(f"Failed to fetch page {page}: {response.status_code}")
                    break
                
                data = response.json()
                page_posts = data.get('posts', [])
                
                if not page_posts:
                    logger.debug(f"No more posts on page {page}")
                    break
                
                # Parse posts
                for post_data in page_posts:
                    post = self._parse_post(post_data)
                    if post:
                        posts.append(post)
                    
                    # Check limit
                    if limit and len(posts) >= limit:
                        logger.info(f"Reached limit of {limit} posts")
                        return posts[:limit]
                
                # Check if there are more pages
                if len(page_posts) < page_size:
                    logger.debug(f"Last page reached (got {len(page_posts)} posts)")
                    break
                
                page += 1
                
            except Exception as e:
                logger.error(f"Error fetching page {page} for @{username}: {e}")
                break
        
        return posts
    
    def _parse_post(self, post_data: Dict[str, Any]) -> Optional[TwitterPost]:
        """Parse a single post from API data.
        
        Args:
            post_data: Raw post data from API
            
        Returns:
            TwitterPost object or None if parsing fails
        """
        try:
            # Parse timestamp
            created_at_str = post_data.get('created_at', '')
            if created_at_str:
                # Try different timestamp formats
                for fmt in ['%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%d %H:%M:%S']:
                    try:
                        created_at = datetime.strptime(created_at_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    logger.warning(f"Could not parse timestamp: {created_at_str}")
                    created_at = datetime.now()
            else:
                created_at = datetime.now()
            
            # Extract text
            text = post_data.get('text', '').strip()
            if not text:
                return None
            
            # Extract hashtags and mentions
            hashtags = self._extract_hashtags(text)
            mentions = self._extract_mentions(text)
            
            # Parse media URLs
            media_urls = []
            media_data = post_data.get('media', [])
            for media in media_data:
                if isinstance(media, dict):
                    url = media.get('url') or media.get('media_url')
                    if url:
                        media_urls.append(url)
            
            return TwitterPost(
                id=str(post_data.get('id', '')),
                text=text,
                created_at=created_at,
                author=post_data.get('author', ''),
                retweet_count=post_data.get('retweet_count', 0),
                favorite_count=post_data.get('favorite_count', 0),
                reply_count=post_data.get('reply_count', 0),
                is_retweet=post_data.get('is_retweet', False),
                is_quote_tweet=post_data.get('is_quote_tweet', False),
                media_urls=media_urls,
                hashtags=hashtags,
                mentions=mentions,
            )
            
        except Exception as e:
            logger.error(f"Failed to parse post: {e}")
            return None
    
    def _clean_username(self, username: str) -> str:
        """Clean and normalize a Twitter username.
        
        Args:
            username: Username with or without @
            
        Returns:
            Clean username without @
        """
        # Remove @ if present
        clean = username.lstrip('@')
        
        # Remove any whitespace
        clean = clean.strip()
        
        # Validate format (alphanumeric, underscore, max 15 chars)
        if not re.match(r'^[A-Za-z0-9_]{1,15}$', clean):
            # Don't raise error, just return None to indicate invalid
            logger.warning(f"Invalid Twitter username format: {username}")
            return None
        
        return clean
    
    def _extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from tweet text.
        
        Args:
            text: Tweet text
            
        Returns:
            List of hashtags (without #)
        """
        hashtag_pattern = r'#([A-Za-z0-9_]+)'
        hashtags = re.findall(hashtag_pattern, text)
        return [tag.lower() for tag in hashtags]
    
    def _extract_mentions(self, text: str) -> List[str]:
        """Extract mentions from tweet text.
        
        Args:
            text: Tweet text
            
        Returns:
            List of mentioned usernames (without @)
        """
        mention_pattern = r'@([A-Za-z0-9_]+)'
        mentions = re.findall(mention_pattern, text)
        return [mention.lower() for mention in mentions]


def check_twitter_archive(username: str) -> Dict[str, Any]:
    """Quick check if a Twitter archive is available.
    
    Args:
        username: Twitter username (with or without @)
        
    Returns:
        Dict with availability info
    """
    client = CommunityArchiveClient()
    return client.check_archive_availability(username)


def fetch_twitter_archive(username: str, limit: Optional[int] = None, archive_url: Optional[str] = None) -> Optional[TwitterArchive]:
    """Fetch a Twitter archive.
    
    Args:
        username: Twitter username (with or without @)
        limit: Maximum posts to fetch
        archive_url: Direct URL to archive JSON file (optional)
        
    Returns:
        TwitterArchive or None if not available
    """
    if archive_url:
        # Fetch from direct URL
        return _fetch_from_url(archive_url, username, limit)
    else:
        # Try Community Archive (limited functionality)
        client = CommunityArchiveClient()
        return client.fetch_archive(username, limit)


def _fetch_from_url(archive_url: str, username: Optional[str] = None, limit: Optional[int] = None) -> Optional[TwitterArchive]:
    """Fetch Twitter archive from direct URL.
    
    Args:
        archive_url: Direct URL to archive JSON
        username: Username for the archive (optional, will extract from URL if not provided)
        limit: Maximum posts to fetch
        
    Returns:
        TwitterArchive or None if failed
    """
    import requests
    from urllib.parse import urlparse
    
    logger.info(f"Fetching Twitter archive from URL: {archive_url}")
    
    try:
        response = requests.get(archive_url, timeout=30)
        response.raise_for_status()
        
        # Parse the JSON data
        archive_data = response.json()
        
        # Extract username from URL if not provided
        if not username:
            parsed_url = urlparse(archive_url)
            path_parts = parsed_url.path.strip('/').split('/')
            # Look for username patterns in the URL path
            for part in reversed(path_parts):
                if part and not part.startswith('archive') and not part.endswith('.json') and not part.startswith('storage'):
                    username = part.lower()
                    break
            
            if not username:
                username = "unknown_user"
        
        # Extract tweets from the archive
        tweets = []
        if isinstance(archive_data, dict):
            # Look for common tweet data structures
            if 'tweets' in archive_data:
                tweets_data = archive_data['tweets']
                # Check if tweets are nested under 'tweet' key (Twitter export format)
                if tweets_data and isinstance(tweets_data[0], dict) and 'tweet' in tweets_data[0]:
                    tweets = [item['tweet'] for item in tweets_data]
                else:
                    tweets = tweets_data
            elif 'data' in archive_data:
                tweets = archive_data['data']
            elif 'statuses' in archive_data:
                tweets = archive_data['statuses']
            else:
                # Try to find any array of tweet-like objects
                for key, value in archive_data.items():
                    if isinstance(value, list) and value and isinstance(value[0], dict):
                        if any(field in value[0] for field in ['text', 'full_text', 'content', 'tweet']):
                            tweets = value
                            break
        
        if not tweets:
            logger.warning(f"No tweets found in archive data")
            return None
        
        # Limit tweets if specified
        if limit:
            tweets = tweets[:limit]
        
        # Convert to TwitterPost objects
        posts = []
        for tweet_data in tweets:
            post = _parse_tweet_from_archive(tweet_data)
            if post:
                posts.append(post)
        
        if not posts:
            logger.warning(f"No valid tweets parsed from archive")
            return None
        
        # Calculate date range
        dates = [post.created_at for post in posts]
        date_range = (min(dates), max(dates))
        
        archive = TwitterArchive(
            username=username,
            posts=posts,
            total_posts=len(posts),
            date_range=date_range,
            archive_url=archive_url,
            download_date=datetime.now(),
        )
        
        logger.info(f"Successfully parsed {len(posts)} tweets from archive for user @{username}")
        return archive
        
    except Exception as e:
        logger.error(f"Failed to fetch archive from URL: {e}")
        return None


def _parse_tweet_from_archive(tweet_data: Dict[str, Any]) -> Optional[TwitterPost]:
    """Parse a single tweet from archive data.
    
    Args:
        tweet_data: Raw tweet data from archive
        
    Returns:
        TwitterPost object or None if parsing fails
    """
    try:
        # Extract text content
        text = (
            tweet_data.get('full_text') or 
            tweet_data.get('text') or 
            tweet_data.get('content') or 
            ''
        ).strip()
        
        if not text:
            return None
        
        # Parse timestamp
        created_at_str = (
            tweet_data.get('created_at') or 
            tweet_data.get('date') or 
            tweet_data.get('timestamp') or 
            ''
        )
        
        if created_at_str:
            # Try different timestamp formats
            for fmt in [
                '%a %b %d %H:%M:%S %z %Y',  # Twitter format
                '%Y-%m-%dT%H:%M:%S.%fZ',    # ISO format
                '%Y-%m-%dT%H:%M:%SZ',       # ISO format without microseconds
                '%Y-%m-%d %H:%M:%S',        # Simple format
            ]:
                try:
                    created_at = datetime.strptime(created_at_str, fmt)
                    break
                except ValueError:
                    continue
            else:
                created_at = datetime.now()
        else:
            created_at = datetime.now()
        
        # Extract engagement metrics
        retweet_count = (
            tweet_data.get('retweet_count') or 
            tweet_data.get('retweets') or 
            0
        )
        favorite_count = (
            tweet_data.get('favorite_count') or 
            tweet_data.get('likes') or 
            tweet_data.get('favourites_count') or 
            0
        )
        reply_count = (
            tweet_data.get('reply_count') or 
            tweet_data.get('replies') or 
            0
        )
        
        # Extract hashtags and mentions
        hashtags = []
        mentions = []
        
        if 'entities' in tweet_data:
            entities = tweet_data['entities']
            if 'hashtags' in entities:
                hashtags = [h.get('text', '') for h in entities['hashtags'] if h.get('text')]
            if 'user_mentions' in entities:
                mentions = [m.get('screen_name', '') for m in entities['user_mentions'] if m.get('screen_name')]
        
        # Also extract from text using regex
        if not hashtags:
            hashtags = re.findall(r'#([A-Za-z0-9_]+)', text)
        if not mentions:
            mentions = re.findall(r'@([A-Za-z0-9_]+)', text)
        
        # Extract media URLs
        media_urls = []
        if 'entities' in tweet_data and 'media' in tweet_data['entities']:
            for media in tweet_data['entities']['media']:
                if media.get('media_url'):
                    media_urls.append(media['media_url'])
        
        # Get author
        author = (
            tweet_data.get('user', {}).get('screen_name') or 
            tweet_data.get('author') or 
            tweet_data.get('username') or 
            'unknown'
        )
        
        return TwitterPost(
            id=str(tweet_data.get('id', tweet_data.get('id_str', ''))),
            text=text,
            created_at=created_at,
            author=author,
            retweet_count=int(retweet_count),
            favorite_count=int(favorite_count),
            reply_count=int(reply_count),
            is_retweet=tweet_data.get('retweeted', False),
            is_quote_tweet='quoted_status' in tweet_data,
            media_urls=media_urls,
            hashtags=[tag.lower() for tag in hashtags],
            mentions=[mention.lower() for mention in mentions],
        )
        
    except Exception as e:
        logger.error(f"Failed to parse tweet: {e}")
        return None


# Convenience functions for CLI
def is_twitter_handle(text: str) -> bool:
    """Check if text looks like a Twitter handle.
    
    Args:
        text: Text to check
        
    Returns:
        True if it looks like a Twitter handle
    """
    if not text:
        return False
    
    # Clean the text
    clean = text.lstrip('@').strip()
    
    # Check format
    return bool(re.match(r'^[A-Za-z0-9_]{1,15}$', clean))


def normalize_twitter_handle(text: str) -> str:
    """Normalize a Twitter handle.
    
    Args:
        text: Twitter handle (with or without @)
        
    Returns:
        Normalized handle without @
    """
    return text.lstrip('@').strip()


# Example usage
if __name__ == "__main__":
    # Test with a known username
    test_username = "elonmusk"
    
    print(f"Checking archive for @{test_username}...")
    availability = check_twitter_archive(test_username)
    
    if availability['available']:
        print(f"✓ Archive available!")
        print(f"  URL: {availability['archive_url']}")
        
        # Fetch first 10 posts
        print("\nFetching first 10 posts...")
        archive = fetch_twitter_archive(test_username, limit=10)
        
        if archive:
            print(f"✓ Fetched {len(archive.posts)} posts")
            print(f"  Date range: {archive.date_range[0]} to {archive.date_range[1]}")
            
            # Show first post
            if archive.posts:
                first_post = archive.posts[0]
                print(f"\nFirst post:")
                print(f"  Text: {first_post.text[:100]}...")
                print(f"  Date: {first_post.created_at}")
                print(f"  Likes: {first_post.favorite_count}")
        else:
            print("✗ Failed to fetch posts")
    else:
        print(f"✗ Archive not available")
        if 'upload_url' in availability:
            print(f"  Upload your archive at: {availability['upload_url']}")

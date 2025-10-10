"""Ghost blog crawling and content fetching.

Fetches blog posts from Ghost CMS blogs via their Content API.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# Optional imports
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests library not installed. Install with: pip install requests")


@dataclass
class GhostPostData:
    """Raw data from Ghost Content API."""
    
    post_id: str
    blog_url: str  # Base URL of the blog
    title: str
    slug: str
    html: str  # Full HTML content
    plaintext: str  # Plain text content (no HTML)
    feature_image: Optional[str]
    featured: bool
    
    published_at: datetime
    updated_at: Optional[datetime]
    
    # Metadata
    excerpt: Optional[str]
    custom_excerpt: Optional[str]
    meta_title: Optional[str]
    meta_description: Optional[str]
    
    # Author info
    authors: List[str]  # List of author names
    primary_author: Optional[str]
    
    # Tags
    tags: List[str]
    primary_tag: Optional[str]
    
    # Engagement (if available)
    reading_time: int  # Minutes
    word_count: int
    
    # URLs
    url: str  # Full post URL
    canonical_url: Optional[str]


class GhostClient:
    """Client for fetching Ghost blog content."""
    
    def __init__(
        self,
        blog_url: str,
        content_api_key: Optional[str] = None,
        storage=None,  # Optional SQLiteStorage for content caching
    ):
        """Initialize Ghost client.
        
        Args:
            blog_url: Base URL of Ghost blog (e.g., "https://blog.example.com")
            content_api_key: Ghost Content API key (optional for public blogs)
            storage: SQLiteStorage instance for content caching (optional)
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError(
                "requests library not installed. "
                "Install with: pip install requests"
            )
        
        self.blog_url = blog_url.rstrip('/')
        self.content_api_key = content_api_key
        self.storage = storage
        
        # Ghost Content API endpoint
        self.api_base = f"{self.blog_url}/ghost/api/content"
        
        logger.info(f"Initialized Ghost client for: {self.blog_url}")
    
    def get_posts(
        self,
        limit: int = 50,
        include_tags: bool = True,
        include_authors: bool = True,
        filter_query: Optional[str] = None,
    ) -> List[GhostPostData]:
        """Fetch posts from Ghost blog.
        
        Args:
            limit: Maximum number of posts to fetch
            include_tags: Include tag information
            include_authors: Include author information
            filter_query: Optional Ghost filter query (e.g., "tag:python")
            
        Returns:
            List of GhostPostData objects
        """
        logger.info(f"Fetching posts from {self.blog_url} (limit: {limit})")
        
        all_posts = []
        page = 1
        per_page = min(limit, 15)  # Ghost API typically limits to 15 per page
        
        while len(all_posts) < limit:
            # Build API request
            params = {
                'key': self.content_api_key,
                'limit': per_page,
                'page': page,
                'formats': 'html,plaintext',  # Get both HTML and plain text
            }
            
            # Include related data
            include_parts = []
            if include_tags:
                include_parts.append('tags')
            if include_authors:
                include_parts.append('authors')
            if include_parts:
                params['include'] = ','.join(include_parts)
            
            # Add filter if provided
            if filter_query:
                params['filter'] = filter_query
            
            # Make request
            try:
                response = requests.get(
                    f"{self.api_base}/posts/",
                    params=params,
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                logger.error(f"Failed to fetch posts (page {page}): {e}")
                break
            
            posts = data.get('posts', [])
            if not posts:
                logger.info(f"No more posts found (page {page})")
                break
            
            # Convert to GhostPostData
            for post in posts:
                if len(all_posts) >= limit:
                    break
                
                try:
                    post_data = self._parse_post(post)
                    all_posts.append(post_data)
                except Exception as e:
                    logger.error(f"Failed to parse post {post.get('id')}: {e}")
                    continue
            
            # Check if there are more pages
            meta = data.get('meta', {}).get('pagination', {})
            if page >= meta.get('pages', 1):
                break
            
            page += 1
        
        logger.info(f"Successfully fetched {len(all_posts)} posts")
        return all_posts
    
    def get_post_by_slug(self, slug: str) -> Optional[GhostPostData]:
        """Fetch a specific post by its slug.
        
        Args:
            slug: Post slug (URL-friendly identifier)
            
        Returns:
            GhostPostData or None if not found
        """
        # Check cache first
        if self.storage:
            cached = self.storage.get_post_content(slug)
            if cached:
                # TODO: Convert cached data back to GhostPostData
                pass
        
        params = {
            'key': self.content_api_key,
            'formats': 'html,plaintext',
            'include': 'tags,authors',
        }
        
        try:
            response = requests.get(
                f"{self.api_base}/posts/slug/{slug}/",
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            
            posts = data.get('posts', [])
            if posts:
                return self._parse_post(posts[0])
            
        except Exception as e:
            logger.error(f"Failed to fetch post {slug}: {e}")
        
        return None
    
    def _parse_post(self, post_dict: dict) -> GhostPostData:
        """Parse Ghost API post response into GhostPostData."""
        
        # Parse dates
        published_at = datetime.fromisoformat(
            post_dict['published_at'].replace('Z', '+00:00')
        )
        updated_at = None
        if post_dict.get('updated_at'):
            updated_at = datetime.fromisoformat(
                post_dict['updated_at'].replace('Z', '+00:00')
            )
        
        # Extract authors
        authors = []
        primary_author = None
        if post_dict.get('authors'):
            authors = [author['name'] for author in post_dict['authors']]
            if authors:
                primary_author = authors[0]
        elif post_dict.get('primary_author'):
            primary_author = post_dict['primary_author'].get('name')
            if primary_author:
                authors = [primary_author]
        
        # Extract tags
        tags = []
        primary_tag = None
        if post_dict.get('tags'):
            tags = [tag['name'] for tag in post_dict['tags'] if tag.get('name')]
            if tags:
                primary_tag = tags[0]
        elif post_dict.get('primary_tag'):
            primary_tag = post_dict['primary_tag'].get('name')
            if primary_tag:
                tags = [primary_tag]
        
        # Calculate word count and reading time
        plaintext = post_dict.get('plaintext', '')
        word_count = len(plaintext.split())
        reading_time = post_dict.get('reading_time', max(1, word_count // 200))
        
        return GhostPostData(
            post_id=post_dict['id'],
            blog_url=self.blog_url,
            title=post_dict['title'],
            slug=post_dict['slug'],
            html=post_dict.get('html', ''),
            plaintext=plaintext,
            feature_image=post_dict.get('feature_image'),
            featured=post_dict.get('featured', False),
            published_at=published_at,
            updated_at=updated_at,
            excerpt=post_dict.get('excerpt'),
            custom_excerpt=post_dict.get('custom_excerpt'),
            meta_title=post_dict.get('meta_title'),
            meta_description=post_dict.get('meta_description'),
            authors=authors,
            primary_author=primary_author,
            tags=tags,
            primary_tag=primary_tag,
            reading_time=reading_time,
            word_count=word_count,
            url=post_dict.get('url', f"{self.blog_url}/{post_dict['slug']}"),
            canonical_url=post_dict.get('canonical_url'),
        )
    
    def get_blog_info(self) -> dict:
        """Get blog metadata (title, description, etc.)."""
        params = {'key': self.content_api_key}
        
        try:
            response = requests.get(
                f"{self.api_base}/settings/",
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            
            settings = data.get('settings', {})
            return {
                'title': settings.get('title', 'Unknown Blog'),
                'description': settings.get('description', ''),
                'logo': settings.get('logo'),
                'icon': settings.get('icon'),
                'cover_image': settings.get('cover_image'),
                'timezone': settings.get('timezone', 'UTC'),
                'url': self.blog_url,
            }
        
        except Exception as e:
            logger.error(f"Failed to fetch blog info: {e}")
            return {
                'title': self.blog_url,
                'description': '',
                'url': self.blog_url,
            }


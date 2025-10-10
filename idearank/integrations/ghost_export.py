"""Ghost export file parser.

Parses Ghost export JSON files (free for all users via Settings → Export).
"""

import json
from datetime import datetime
from typing import List, Optional
import logging
from pathlib import Path

from idearank.integrations.ghost import GhostPostData

logger = logging.getLogger(__name__)


class GhostExportParser:
    """Parser for Ghost export JSON files."""
    
    def __init__(self, export_file_path: str):
        """Initialize parser with export file.
        
        Args:
            export_file_path: Path to Ghost export JSON file
        """
        self.export_file_path = Path(export_file_path)
        
        if not self.export_file_path.exists():
            raise FileNotFoundError(f"Export file not found: {export_file_path}")
        
        # Load and parse export
        logger.info(f"Loading Ghost export from: {export_file_path}")
        with open(self.export_file_path, 'r', encoding='utf-8') as f:
            self.export_data = json.load(f)
        
        # Extract main data
        self.db = self.export_data.get('db', [{}])[0]
        self.data = self.db.get('data', {})
        
        # Parse entities
        self.posts = self.data.get('posts', [])
        self.tags = {tag['id']: tag for tag in self.data.get('tags', [])}
        self.users = {user['id']: user for user in self.data.get('users', [])}
        self.posts_tags = self.data.get('posts_tags', [])
        self.posts_authors = self.data.get('posts_authors', [])
        
        # Get blog settings
        self.settings = {s['key']: s['value'] for s in self.data.get('settings', [])}
        self.blog_url = self.settings.get('url', 'unknown')
        self.blog_title = self.settings.get('title', 'Unknown Blog')
        
        logger.info(f"Loaded {len(self.posts)} posts from {self.blog_title}")
    
    def get_posts(
        self,
        limit: Optional[int] = None,
        status: str = 'published',
        tag: Optional[str] = None,
        author: Optional[str] = None,
    ) -> List[GhostPostData]:
        """Get posts from export.
        
        Args:
            limit: Maximum number of posts to return
            status: Post status ('published', 'draft', 'scheduled')
            tag: Filter by tag name (optional)
            author: Filter by author name (optional)
            
        Returns:
            List of GhostPostData objects
        """
        filtered_posts = []
        
        for post in self.posts:
            # Filter by status
            if post.get('status') != status:
                continue
            
            # Filter by tag
            if tag:
                post_tag_ids = [
                    pt['tag_id'] for pt in self.posts_tags 
                    if pt['post_id'] == post['id']
                ]
                post_tags = [
                    self.tags[tid]['name'] for tid in post_tag_ids 
                    if tid in self.tags
                ]
                if tag not in post_tags:
                    continue
            
            # Filter by author
            if author:
                post_author_ids = [
                    pa['author_id'] for pa in self.posts_authors 
                    if pa['post_id'] == post['id']
                ]
                post_authors = [
                    self.users[aid]['name'] for aid in post_author_ids 
                    if aid in self.users
                ]
                if author not in post_authors:
                    continue
            
            # Convert to GhostPostData
            try:
                post_data = self._parse_post(post)
                filtered_posts.append(post_data)
            except Exception as e:
                logger.error(f"Failed to parse post {post.get('id')}: {e}")
                continue
            
            # Check limit
            if limit and len(filtered_posts) >= limit:
                break
        
        logger.info(f"Filtered to {len(filtered_posts)} posts")
        return filtered_posts
    
    def get_blog_info(self) -> dict:
        """Get blog metadata from export."""
        return {
            'title': self.blog_title,
            'description': self.settings.get('description', ''),
            'logo': self.settings.get('logo'),
            'icon': self.settings.get('icon'),
            'cover_image': self.settings.get('cover_image'),
            'timezone': self.settings.get('timezone', 'UTC'),
            'url': self.blog_url,
        }
    
    def _parse_post(self, post_dict: dict) -> GhostPostData:
        """Parse Ghost export post into GhostPostData."""
        
        # Parse dates
        published_at = None
        if post_dict.get('published_at'):
            try:
                published_at = datetime.fromisoformat(
                    post_dict['published_at'].replace('Z', '+00:00')
                )
            except:
                published_at = datetime.now()
        
        updated_at = None
        if post_dict.get('updated_at'):
            try:
                updated_at = datetime.fromisoformat(
                    post_dict['updated_at'].replace('Z', '+00:00')
                )
            except:
                pass
        
        # Get post tags
        post_tag_ids = [
            pt['tag_id'] for pt in self.posts_tags 
            if pt['post_id'] == post_dict['id']
        ]
        tags = [
            self.tags[tid]['name'] for tid in post_tag_ids 
            if tid in self.tags
        ]
        primary_tag = tags[0] if tags else None
        
        # Get post authors
        post_author_ids = [
            pa['author_id'] for pa in self.posts_authors 
            if pa['post_id'] == post_dict['id']
        ]
        authors = [
            self.users[aid]['name'] for aid in post_author_ids 
            if aid in self.users
        ]
        primary_author = authors[0] if authors else None
        
        # Get plaintext content
        plaintext = post_dict.get('plaintext', '')
        if not plaintext:
            # If no plaintext, strip HTML from html field
            html = post_dict.get('html', '')
            import re
            plaintext = re.sub(r'<[^>]+>', '', html)
        
        # Calculate reading time and word count
        word_count = len(plaintext.split())
        reading_time = max(1, word_count // 200)  # ~200 words per minute
        
        # Construct full URL
        url = f"{self.blog_url}/{post_dict.get('slug', '')}"
        
        return GhostPostData(
            post_id=post_dict['id'],
            blog_url=self.blog_url,
            title=post_dict.get('title', 'Untitled'),
            slug=post_dict.get('slug', ''),
            html=post_dict.get('html', ''),
            plaintext=plaintext,
            feature_image=post_dict.get('feature_image'),
            featured=post_dict.get('featured', False),
            published_at=published_at or datetime.now(),
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
            url=url,
            canonical_url=post_dict.get('canonical_url'),
        )
    
    def get_stats(self) -> dict:
        """Get statistics about the export."""
        return {
            'total_posts': len(self.posts),
            'published': len([p for p in self.posts if p.get('status') == 'published']),
            'drafts': len([p for p in self.posts if p.get('status') == 'draft']),
            'total_tags': len(self.tags),
            'total_authors': len(self.users),
            'blog_title': self.blog_title,
            'blog_url': self.blog_url,
        }


def create_ghost_client_from_export(
    export_file_path: str,
    storage=None,
) -> 'GhostExportClient':
    """Create a Ghost client that reads from an export file.
    
    Args:
        export_file_path: Path to Ghost export JSON file
        storage: Optional SQLiteStorage for caching
        
    Returns:
        GhostExportClient instance
    """
    return GhostExportClient(export_file_path, storage)


class GhostExportClient:
    """Ghost client that reads from export files instead of API.
    
    Compatible interface with GhostClient for use in pipelines.
    """
    
    def __init__(self, export_file_path: str, storage=None):
        """Initialize from export file.
        
        Args:
            export_file_path: Path to Ghost export JSON file
            storage: Optional SQLiteStorage for caching
        """
        self.parser = GhostExportParser(export_file_path)
        self.storage = storage
        self.blog_url = self.parser.blog_url
        
        logger.info(f"Initialized Ghost export client for: {self.blog_url}")
    
    def get_posts(
        self,
        limit: int = 50,
        include_tags: bool = True,
        include_authors: bool = True,
        filter_query: Optional[str] = None,
    ) -> List[GhostPostData]:
        """Get posts from export file.
        
        Args:
            limit: Maximum number of posts
            include_tags: Include tag info (always True for exports)
            include_authors: Include author info (always True for exports)
            filter_query: Simple filter: "tag:NAME" or "author:NAME"
            
        Returns:
            List of GhostPostData objects
        """
        # Parse simple filter queries
        tag = None
        author = None
        if filter_query:
            if filter_query.startswith('tag:'):
                tag = filter_query.replace('tag:', '').strip()
            elif filter_query.startswith('author:'):
                author = filter_query.replace('author:', '').strip()
        
        return self.parser.get_posts(
            limit=limit,
            status='published',
            tag=tag,
            author=author,
        )
    
    def get_blog_info(self) -> dict:
        """Get blog metadata from export."""
        return self.parser.get_blog_info()
    
    def get_export_stats(self) -> dict:
        """Get statistics about the export."""
        return self.parser.get_stats()


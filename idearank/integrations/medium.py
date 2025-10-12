"""
Medium Archive Parser

Parses Medium export archives (ZIP files containing HTML articles).
Medium exports include posts, drafts, and publication data.
"""

import logging
import json
import zipfile
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional
from dataclasses import dataclass
from html.parser import HTMLParser
import re

logger = logging.getLogger(__name__)


@dataclass
class MediumPost:
    """Represents a single Medium post."""
    id: str
    title: str
    content: str
    published_at: Optional[datetime]
    url: Optional[str]
    canonical_url: Optional[str]
    claps: int
    word_count: int
    tags: List[str]
    is_draft: bool
    is_comment: bool = False  # Short responses/comments on other posts


@dataclass
class MediumUser:
    """Represents a Medium user/author."""
    username: str
    name: str
    bio: str
    follower_count: int
    following_count: int


class MediumHTMLParser(HTMLParser):
    """Parse Medium HTML export files to extract content."""
    
    def __init__(self):
        super().__init__()
        self.title = ""
        self.content = []
        self.in_article = False
        self.in_title = False
        self.current_tag = None
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Look for article content
        if tag == 'article':
            self.in_article = True
        elif tag == 'h1' and self.in_article and not self.title:
            self.in_title = True
        elif tag in ['p', 'h2', 'h3', 'h4', 'li', 'blockquote'] and self.in_article:
            self.current_tag = tag
            
    def handle_endtag(self, tag):
        if tag == 'article':
            self.in_article = False
        elif tag == 'h1':
            self.in_title = False
        elif tag in ['p', 'h2', 'h3', 'h4', 'li', 'blockquote']:
            self.current_tag = None
            
    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
            
        if self.in_title:
            self.title = data
        elif self.current_tag and self.in_article:
            self.content.append(data)
            
    def get_title(self) -> str:
        return self.title
        
    def get_content(self) -> str:
        return '\n\n'.join(self.content)


class MediumArchiveClient:
    """Client for parsing Medium archive exports."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def load_archive(self, archive_path: str) -> tuple[MediumUser, List[MediumPost]]:
        """
        Load and parse a Medium archive ZIP file.
        
        Args:
            archive_path: Path to the Medium export ZIP file
            
        Returns:
            Tuple of (user info, list of posts)
        """
        archive_path = Path(archive_path)
        
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")
            
        if archive_path.suffix.lower() != '.zip':
            raise ValueError("Medium archive must be a ZIP file")
            
        self.logger.info(f"Loading Medium archive from: {archive_path}")
        
        with zipfile.ZipFile(archive_path, 'r') as zf:
            # Get list of files
            file_list = zf.namelist()
            
            # Parse user profile
            user = self._parse_profile(zf, file_list)
            
            # Parse posts
            posts = self._parse_posts(zf, file_list)
            
        self.logger.info(f"Loaded {len(posts)} posts from {user.name}")
        return user, posts
        
    def _parse_profile(self, zf: zipfile.ZipFile, file_list: List[str]) -> MediumUser:
        """Parse user profile from archive."""
        # Look for profile.html or similar
        profile_files = [f for f in file_list if 'profile' in f.lower() and f.endswith('.html')]
        
        username = "unknown"
        name = "Medium User"
        bio = ""
        follower_count = 0
        following_count = 0
        
        if profile_files:
            try:
                with zf.open(profile_files[0]) as f:
                    html = f.read().decode('utf-8')
                    # Try to extract username from HTML
                    username_match = re.search(r'@(\w+)', html)
                    if username_match:
                        username = username_match.group(1)
                    # Try to extract name
                    name_match = re.search(r'<h1[^>]*>([^<]+)</h1>', html)
                    if name_match:
                        name = name_match.group(1).strip()
            except Exception as e:
                self.logger.warning(f"Could not parse profile: {e}")
                
        return MediumUser(
            username=username,
            name=name,
            bio=bio,
            follower_count=follower_count,
            following_count=following_count,
        )
        
    def _parse_posts(self, zf: zipfile.ZipFile, file_list: List[str]) -> List[MediumPost]:
        """Parse all posts from archive."""
        posts = []
        
        # Look for post HTML files (usually in posts/ directory)
        post_files = [
            f for f in file_list 
            if f.endswith('.html') and 
            ('posts/' in f or 'drafts/' in f) and
            'profile' not in f.lower()
        ]
        
        for post_file in post_files:
            try:
                post = self._parse_post_file(zf, post_file)
                if post:
                    posts.append(post)
            except Exception as e:
                self.logger.warning(f"Could not parse {post_file}: {e}")
                
        return posts
        
    def _parse_post_file(self, zf: zipfile.ZipFile, file_path: str) -> Optional[MediumPost]:
        """Parse a single post HTML file."""
        with zf.open(file_path) as f:
            html = f.read().decode('utf-8')
            
        # Use custom parser
        parser = MediumHTMLParser()
        parser.feed(html)
        
        title = parser.get_title()
        content = parser.get_content()
        
        if not title or not content:
            return None
            
        # Extract metadata from HTML
        published_at = self._extract_publish_date(html)
        url = self._extract_url(html)
        canonical_url = self._extract_canonical_url(html)
        claps = self._extract_claps(html)
        tags = self._extract_tags(html)
        is_draft = 'drafts/' in file_path
        
        # Generate ID from filename
        post_id = Path(file_path).stem
        
        # Count words
        word_count = len(content.split())
        
        # Detect comments/responses (heuristics)
        is_comment = self._is_likely_comment(file_path, title, content, word_count)
        
        return MediumPost(
            id=post_id,
            title=title,
            content=content,
            published_at=published_at,
            url=url,
            canonical_url=canonical_url,
            claps=claps,
            word_count=word_count,
            tags=tags,
            is_draft=is_draft,
            is_comment=is_comment,
        )
        
    def _extract_publish_date(self, html: str) -> Optional[datetime]:
        """Extract publish date from HTML (returns timezone-aware datetime)."""
        # Try various patterns
        patterns = [
            r'<time[^>]*datetime="([^"]+)"',
            r'"datePublished":"([^"]+)"',
            r'Published on ([A-Za-z]+ \d+, \d{4})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html)
            if match:
                date_str = match.group(1)
                try:
                    # Try ISO format first
                    dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    # Ensure timezone-aware
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt
                except:
                    try:
                        # Try common formats (returns naive, so make it aware)
                        dt = datetime.strptime(date_str, '%B %d, %Y')
                        return dt.replace(tzinfo=timezone.utc)
                    except:
                        pass
                        
        return None
        
    def _extract_url(self, html: str) -> Optional[str]:
        """Extract post URL from HTML."""
        match = re.search(r'<link rel="canonical" href="([^"]+)"', html)
        if match:
            return match.group(1)
        return None
        
    def _extract_canonical_url(self, html: str) -> Optional[str]:
        """Extract canonical URL from HTML."""
        match = re.search(r'"canonicalUrl":"([^"]+)"', html)
        if match:
            return match.group(1)
        return None
        
    def _extract_claps(self, html: str) -> int:
        """Extract clap count from HTML."""
        match = re.search(r'"claps":(\d+)', html)
        if match:
            return int(match.group(1))
        return 0
        
    def _extract_tags(self, html: str) -> List[str]:
        """Extract tags from HTML."""
        tags = []
        # Try to find tags in meta or content
        tag_matches = re.findall(r'"tag":"([^"]+)"', html)
        if tag_matches:
            tags = tag_matches
        else:
            # Try alternative format
            tag_matches = re.findall(r'<a[^>]*class="[^"]*tag[^"]*"[^>]*>([^<]+)</a>', html)
            tags = [t.strip() for t in tag_matches]
        return tags[:10]  # Limit to 10 tags
        
    def _is_likely_comment(self, file_path: str, title: str, content: str, word_count: int) -> bool:
        """
        Detect if a post is likely a comment/response rather than a full article.
        
        Heuristics:
        - Located in 'comments' or 'responses' directory
        - Very short (< 200 words)
        - Title indicates response (starts with common response patterns)
        - Title is very short and generic
        """
        # Check directory structure
        path_lower = file_path.lower()
        if any(indicator in path_lower for indicator in ['comment', 'response', 'replies']):
            return True
        
        # Very short content is likely a comment
        if word_count < 200:
            return True
        
        # Check for comment-like title patterns
        title_lower = title.lower().strip()
        comment_patterns = [
            'thanks', 'thank you', 'great', 'nice', 'awesome', 'love',
            'this is', 're:', 'reply', 'response', '👍', '🙏', '❤️',
            'agreed', 'exactly', 'yes!', 'no.', 'well said', 'good point',
        ]
        
        # Title starts with a comment pattern
        for pattern in comment_patterns:
            if title_lower.startswith(pattern):
                return True
        
        # Very short title (< 4 words) combined with short content (< 300 words)
        if len(title.split()) < 4 and word_count < 300:
            # Check if title is generic
            if any(word in title_lower for word in ['thanks', 'great', 'nice', 'this', 'that']):
                return True
        
        return False


def load_medium_archive(archive_path: str) -> tuple[MediumUser, List[MediumPost]]:
    """
    Convenience function to load a Medium archive.
    
    Args:
        archive_path: Path to the Medium export ZIP file
        
    Returns:
        Tuple of (user info, list of posts)
    """
    client = MediumArchiveClient()
    return client.load_archive(archive_path)


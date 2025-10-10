"""Example of processing a Ghost export file with IdeaRank.

This shows how to use the Ghost export parser to analyze blog content.
"""

import logging
import json
from datetime import datetime
from pathlib import Path

from idearank.integrations.ghost_export import GhostExportParser, GhostExportClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_ghost_export(output_path: str = "sample_blog.ghost.json"):
    """Create a minimal sample Ghost export for testing."""
    
    export_data = {
        "db": [{
            "meta": {
                "exported_on": datetime.now().timestamp(),
                "version": "5.0.0"
            },
            "data": {
                "posts": [
                    {
                        "id": "post_1",
                        "title": "Introduction to Machine Learning",
                        "slug": "intro-to-ml",
                        "html": "<p>Machine learning is transforming software development...</p>",
                        "plaintext": "Machine learning is transforming software development. This post covers the fundamentals of supervised learning, neural networks, and practical applications.",
                        "feature_image": None,
                        "featured": True,
                        "status": "published",
                        "published_at": "2024-01-15T10:00:00.000Z",
                        "updated_at": "2024-01-15T10:00:00.000Z",
                        "excerpt": "Learn the fundamentals of machine learning",
                        "meta_description": "A comprehensive introduction to machine learning concepts",
                    },
                    {
                        "id": "post_2",
                        "title": "Advanced Neural Networks",
                        "slug": "advanced-neural-networks",
                        "html": "<p>Building on our ML fundamentals, let's explore deep learning...</p>",
                        "plaintext": "Building on our ML fundamentals, let's explore deep learning. We'll cover CNNs, RNNs, and transformers with practical examples.",
                        "feature_image": None,
                        "featured": False,
                        "status": "published",
                        "published_at": "2024-02-20T10:00:00.000Z",
                        "updated_at": "2024-02-20T10:00:00.000Z",
                        "excerpt": "Deep dive into neural network architectures",
                        "meta_description": "Advanced neural network techniques and architectures",
                    },
                    {
                        "id": "post_3",
                        "title": "Building Production ML Systems",
                        "slug": "production-ml",
                        "html": "<p>Taking ML models to production requires careful engineering...</p>",
                        "plaintext": "Taking ML models to production requires careful engineering. This post covers deployment, monitoring, and scaling ML systems in real-world applications.",
                        "feature_image": None,
                        "featured": False,
                        "status": "published",
                        "published_at": "2024-03-10T10:00:00.000Z",
                        "updated_at": "2024-03-10T10:00:00.000Z",
                        "excerpt": "Production ML best practices",
                        "meta_description": "How to deploy and scale machine learning systems",
                    },
                ],
                "tags": [
                    {"id": "tag_1", "name": "machine-learning", "slug": "machine-learning"},
                    {"id": "tag_2", "name": "tutorial", "slug": "tutorial"},
                    {"id": "tag_3", "name": "advanced", "slug": "advanced"},
                ],
                "users": [
                    {"id": "user_1", "name": "Jane Doe", "slug": "jane"},
                    {"id": "user_2", "name": "John Smith", "slug": "john"},
                ],
                "posts_tags": [
                    {"post_id": "post_1", "tag_id": "tag_1"},
                    {"post_id": "post_1", "tag_id": "tag_2"},
                    {"post_id": "post_2", "tag_id": "tag_1"},
                    {"post_id": "post_2", "tag_id": "tag_3"},
                    {"post_id": "post_3", "tag_id": "tag_1"},
                ],
                "posts_authors": [
                    {"post_id": "post_1", "author_id": "user_1"},
                    {"post_id": "post_2", "author_id": "user_1"},
                    {"post_id": "post_3", "author_id": "user_2"},
                ],
                "settings": [
                    {"key": "title", "value": "Sample ML Blog"},
                    {"key": "description", "value": "A blog about machine learning"},
                    {"key": "url", "value": "https://ml-blog.example.com"},
                ],
            }
        }]
    }
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2)
    
    logger.info(f"Created sample export: {output_path}")
    return output_path


def main():
    """Demonstrate Ghost export parsing."""
    
    print("=" * 70)
    print("Ghost Export Parser Example")
    print("=" * 70)
    
    # Create sample export file
    print("\n[1/4] Creating sample Ghost export...")
    export_file = create_sample_ghost_export()
    print(f"✓ Created: {export_file}")
    
    # Parse export
    print("\n[2/4] Parsing Ghost export...")
    parser = GhostExportParser(export_file)
    print(f"✓ Loaded blog: {parser.blog_title}")
    
    # Get stats
    print("\n[3/4] Export statistics...")
    stats = parser.get_stats()
    print(f"  Total posts: {stats['total_posts']}")
    print(f"  Published: {stats['published']}")
    print(f"  Drafts: {stats['drafts']}")
    print(f"  Tags: {stats['total_tags']}")
    print(f"  Authors: {stats['total_authors']}")
    print(f"  Blog URL: {stats['blog_url']}")
    
    # Get posts
    print("\n[4/4] Fetching posts...")
    posts = parser.get_posts(limit=10, status='published')
    print(f"✓ Found {len(posts)} published posts")
    
    # Display posts
    print("\nPosts found:")
    print("-" * 70)
    for i, post in enumerate(posts, 1):
        print(f"\n{i}. {post.title}")
        print(f"   Published: {post.published_at.strftime('%Y-%m-%d')}")
        print(f"   Tags: {', '.join(post.tags)}")
        print(f"   Author: {post.primary_author}")
        print(f"   Words: {post.word_count} | Reading time: {post.reading_time} min")
        print(f"   URL: {post.url}")
    
    # Test filtering
    print("\n" + "=" * 70)
    print("Testing Filters")
    print("=" * 70)
    
    # Filter by tag
    tutorial_posts = parser.get_posts(tag="tutorial")
    print(f"\nPosts with tag 'tutorial': {len(tutorial_posts)}")
    for post in tutorial_posts:
        print(f"  - {post.title}")
    
    # Filter by author
    jane_posts = parser.get_posts(author="Jane Doe")
    print(f"\nPosts by 'Jane Doe': {len(jane_posts)}")
    for post in jane_posts:
        print(f"  - {post.title}")
    
    print("\n" + "=" * 70)
    print("Example Complete!")
    print("=" * 70)
    
    print("\nTo process this with IdeaRank:")
    print(f"  idearank process-ghost-export {export_file}")
    
    # Clean up
    Path(export_file).unlink()
    print(f"\nCleaned up: {export_file}")


if __name__ == "__main__":
    main()


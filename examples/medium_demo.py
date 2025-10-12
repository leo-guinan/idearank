"""
Medium Integration Demo

Demonstrates how to use the Medium archive parser.
"""

from idearank.integrations.medium import MediumArchiveClient

def demo_medium_parser():
    """
    Demo of Medium archive parsing.
    
    To use this:
    1. Export your Medium archive: Settings → Security and apps → Download your information
    2. Save the ZIP file
    3. Update the path below
    """
    
    print("=" * 70)
    print("Medium Archive Parser Demo")
    print("=" * 70)
    print()
    
    # Example usage (you'll need a real Medium export ZIP)
    # archive_path = "~/Downloads/medium-export.zip"
    # 
    # client = MediumArchiveClient()
    # user, posts = client.load_archive(archive_path)
    # 
    # print(f"User: {user.name} (@{user.username})")
    # print(f"Total posts: {len(posts)}")
    # print()
    # 
    # # Show first few posts
    # for post in posts[:5]:
    #     print(f"Title: {post.title}")
    #     print(f"  Published: {post.published_at}")
    #     print(f"  Claps: {post.claps:,}")
    #     print(f"  Words: {post.word_count:,}")
    #     print(f"  Tags: {', '.join(post.tags)}")
    #     print(f"  Draft: {post.is_draft}")
    #     print()
    
    print("To use this demo:")
    print("1. Export your Medium archive:")
    print("   Settings → Security and apps → Download your information")
    print()
    print("2. Uncomment the code above and update the archive_path")
    print()
    print("3. Run: python examples/medium_demo.py")
    print()
    print("Or use the CLI directly:")
    print("  idearank process-medium ~/Downloads/medium-export.zip")
    print()


if __name__ == "__main__":
    demo_medium_parser()


"""
Twitter Integration Demo

Demonstrates how to process Twitter archive JSON files.
"""

from idearank.integrations.twitter import load_twitter_archive

def demo_twitter_parser():
    """
    Demo of Twitter archive parsing.
    
    To use this:
    1. Download your Twitter archive: https://x.com/settings/download_your_data
    2. Extract the ZIP file
    3. Find the tweet.js or tweets.json file
    4. Update the path below
    """
    
    print("=" * 70)
    print("Twitter Archive Parser Demo")
    print("=" * 70)
    print()
    
    # Example usage (you'll need a real Twitter archive JSON)
    # archive_path = "~/Downloads/twitter-archive/tweet.js"
    # 
    # archive = load_twitter_archive(archive_path)
    # 
    # print(f"User: @{archive.username}")
    # print(f"Total posts: {len(archive.posts)}")
    # print(f"Date range: {archive.date_range[0]} to {archive.date_range[1]}")
    # print()
    # 
    # # Show first few tweets
    # for post in archive.posts[:5]:
    #     print(f"Tweet: {post.text[:80]}...")
    #     print(f"  Date: {post.created_at}")
    #     print(f"  Engagement: {post.favorite_count + post.retweet_count}")
    #     print(f"  Hashtags: {', '.join(post.hashtags) if post.hashtags else 'None'}")
    #     print()
    
    print("To use this demo:")
    print("1. Download your Twitter archive:")
    print("   https://x.com/settings/download_your_data")
    print()
    print("2. Extract the ZIP and find tweets.json or tweet.js")
    print()
    print("3. Uncomment the code above and update the archive_path")
    print()
    print("4. Run: python examples/twitter_demo.py")
    print()
    print("Or use the CLI directly:")
    print("  idearank source add ~/Downloads/twitter-archive.json twitter")
    print("  idearank process-all")
    print()


if __name__ == "__main__":
    demo_twitter_parser()

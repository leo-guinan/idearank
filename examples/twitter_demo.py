"""Demo of Twitter integration with community archive.

Shows how to check for and fetch Twitter archives from community-archive.org.
"""

import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)


def main():
    """Demonstrate Twitter archive integration."""
    
    print("=" * 70)
    print("Twitter Community Archive Integration Demo")
    print("=" * 70)
    
    from idearank.integrations.twitter import (
        CommunityArchiveClient,
        check_twitter_archive,
        fetch_twitter_archive,
        is_twitter_handle,
        normalize_twitter_handle
    )
    
    # Test Twitter handle detection
    print("\n[1/4] Testing Twitter Handle Detection:")
    print("-" * 50)
    
    test_handles = [
        "@elonmusk",
        "elonmusk", 
        "BarackObama",
        "not_a_handle",
        "this_is_too_long_to_be_a_handle_123",
        "valid_handle_123"
    ]
    
    for handle in test_handles:
        is_valid = is_twitter_handle(handle)
        normalized = normalize_twitter_handle(handle) if is_valid else "N/A"
        print(f"  '{handle}' -> Valid: {is_valid}, Normalized: {normalized}")
    
    # Test archive availability check
    print("\n[2/4] Testing Archive Availability:")
    print("-" * 50)
    
    test_usernames = ["elonmusk", "BarackObama", "non_existent_user_12345"]
    
    for username in test_usernames:
        print(f"\nChecking @{username}...")
        availability = check_twitter_archive(username)
        
        if availability['available']:
            print(f"  ✓ Archive available!")
            print(f"    URL: {availability['archive_url']}")
            if 'metadata' in availability:
                metadata = availability['metadata']
                print(f"    Posts: {metadata.get('total_posts', 'Unknown')}")
                print(f"    Date range: {metadata.get('date_range', 'Unknown')}")
        else:
            print(f"  ✗ Archive not available")
            if 'upload_url' in availability:
                print(f"    Upload at: {availability['upload_url']}")
            if 'error' in availability:
                print(f"    Error: {availability['error']}")
    
    # Test fetching a small archive (if available)
    print("\n[3/4] Testing Archive Fetching:")
    print("-" * 50)
    
    # Try to fetch a small sample
    test_username = "elonmusk"  # Usually has archives
    print(f"Fetching first 5 posts from @{test_username}...")
    
    archive = fetch_twitter_archive(test_username, limit=5)
    
    if archive:
        print(f"  ✓ Successfully fetched archive!")
        print(f"    Username: @{archive.username}")
        print(f"    Total posts: {archive.total_posts}")
        print(f"    Date range: {archive.date_range[0]} to {archive.date_range[1]}")
        print(f"    Archive URL: {archive.archive_url}")
        
        print(f"\n  Sample posts:")
        for i, post in enumerate(archive.posts[:3], 1):
            print(f"    {i}. {post.text[:80]}...")
            print(f"       Likes: {post.favorite_count}, RTs: {post.retweet_count}")
            print(f"       Hashtags: {post.hashtags}")
            print(f"       Mentions: {post.mentions}")
    else:
        print(f"  ✗ Failed to fetch archive")
        print(f"    This could mean:")
        print(f"    - No archive exists for @{test_username}")
        print(f"    - Network issues")
        print(f"    - API changes")
    
    # Test CLI integration
    print("\n[4/4] CLI Integration:")
    print("-" * 50)
    
    print("To use Twitter integration in IdeaRank CLI:")
    print()
    print("1. Add a Twitter source:")
    print("   idearank source add @username")
    print("   idearank source add username  # @ is optional")
    print()
    print("2. Check what sources you have:")
    print("   idearank source list")
    print()
    print("3. Process all sources (including Twitter):")
    print("   idearank process-all")
    print()
    print("4. View results:")
    print("   idearank viz dashboard")
    print("   idearank diagnose")
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    
    print("\nKey Features:")
    print("✓ Auto-detects Twitter handles")
    print("✓ Checks community-archive.org for availability")
    print("✓ Fetches full archives when available")
    print("✓ Processes tweets through IdeaRank pipeline")
    print("✓ Integrates with multi-source management")
    print("✓ Provides helpful error messages")
    
    print("\nWhat happens if no archive is found:")
    print("- Source is still added to your list")
    print("- You get instructions to upload to community-archive.org")
    print("- Source will be processed when archive becomes available")
    
    print("\nCommunity Archive Benefits:")
    print("- Preserves Twitter data before it's lost")
    print("- Makes data available for analysis")
    print("- Free service for researchers")
    print("- Respects user privacy")


if __name__ == "__main__":
    main()

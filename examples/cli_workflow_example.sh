#!/bin/bash
# IdeaRank CLI Workflow Example
# This script demonstrates the complete workflow for using IdeaRank CLI

set -e  # Exit on error

echo "================================================"
echo "IdeaRank CLI Workflow Example"
echo "================================================"
echo ""

# Step 1: Install IdeaRank with CLI support
echo "Step 1: Installing IdeaRank with CLI support..."
echo "$ pip install -e \".[pipeline]\""
# pip install -e ".[pipeline]"
echo "✓ Installed"
echo ""

# Step 2: Check if CLI is available
echo "Step 2: Verifying CLI installation..."
echo "$ idearank --help"
idearank --help
echo ""
echo "✓ CLI is working"
echo ""

# Step 3: Show current configuration
echo "Step 3: Checking current configuration..."
echo "$ idearank config show"
idearank config show
echo ""

# Step 4: Interactive setup (commented out - requires user input)
echo "Step 4: Interactive setup (run manually)..."
echo "$ idearank setup"
echo ""
echo "This will prompt you for:"
echo "  - YouTube API Key (required)"
echo "  - Gladia API Key (optional)"
echo "  - Chroma mode (local/cloud)"
echo ""
echo "For this example, we'll use individual config commands instead."
echo ""

# Step 5: Set YouTube API key (example - use your real key)
echo "Step 5: Setting API keys..."
echo "$ idearank config set-youtube-key YOUR_YOUTUBE_API_KEY"
echo ""
echo "Note: Replace YOUR_YOUTUBE_API_KEY with your actual key from:"
echo "https://console.cloud.google.com/apis/credentials"
echo ""
# idearank config set-youtube-key YOUR_YOUTUBE_API_KEY
echo "(Skipped in example - add your key to test)"
echo ""

# Step 6: Optionally set Gladia key
echo "Step 6: (Optional) Setting Gladia API key for transcription..."
echo "$ idearank config set-gladia-key YOUR_GLADIA_API_KEY"
echo ""
echo "Note: Get your Gladia key from: https://gladia.io"
echo "(Skip this if you only want to use YouTube auto-captions)"
echo ""
# idearank config set-gladia-key YOUR_GLADIA_API_KEY
echo "(Skipped in example - add your key to test)"
echo ""

# Step 7: Set Chroma mode (default is local)
echo "Step 7: Setting Chroma mode..."
echo "$ idearank config set-chroma-mode local"
# idearank config set-chroma-mode local
echo "✓ Using local Chroma storage at ~/.idearank/chroma_db/"
echo ""

# Step 8: Process a YouTube channel
echo "Step 8: Processing a YouTube channel..."
echo "$ idearank process https://youtube.com/@channelname --max-videos 10"
echo ""
echo "This will:"
echo "  1. Fetch channel videos from YouTube API"
echo "  2. Get video metadata (views, likes, duration, etc.)"
echo "  3. Fetch transcripts (YouTube auto-captions)"
echo "  4. Convert to IdeaRank format"
echo "  5. Save to SQLite database"
echo "  6. Generate embeddings"
echo "  7. Index in Chroma vector database"
echo "  8. Compute IdeaRank scores (U, C, L, Q, T)"
echo "  9. Display results in a beautiful table"
echo ""
echo "Example output:"
echo "┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━┳━━━━━┓"
echo "┃ Rank ┃ Title                ┃ Score   ┃ U   ┃ ... ┃"
echo "┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━╇━━━━━┩"
echo "│ 1    │ Best Video Title     │ 0.8234  │ 0.9 │ ... │"
echo "│ 2    │ Second Best Video... │ 0.7891  │ 0.8 │ ... │"
echo "└──────┴──────────────────────┴─────────┴─────┴─────┘"
echo ""
echo "(Skipped in example - requires API keys)"
echo ""

# Step 9: Process with all features
echo "Step 9: Advanced processing with all features..."
echo "$ export OPENAI_API_KEY=\"sk-...\""
echo "$ idearank process https://youtube.com/@channelname \\"
echo "    --max-videos 50 \\"
echo "    --use-gladia \\"
echo "    --openai-key \$OPENAI_API_KEY \\"
echo "    --output my_results.db \\"
echo "    --collection my_collection"
echo ""
echo "This enables:"
echo "  - Gladia transcription for videos without captions"
echo "  - OpenAI embeddings for better semantic understanding"
echo "  - Custom output path and collection name"
echo ""
echo "(Skipped in example - requires API keys)"
echo ""

# Step 10: Query results
echo "Step 10: Querying results from SQLite database..."
echo "$ sqlite3 idearank_results.db"
echo ""
echo "Example queries:"
echo "  SELECT title, idearank_score FROM videos ORDER BY idearank_score DESC LIMIT 10;"
echo "  SELECT AVG(uniqueness_score) FROM videos;"
echo "  SELECT title FROM videos WHERE passes_gates = 1;"
echo ""
echo "(Results are stored in SQLite for further analysis)"
echo ""

# Summary
echo "================================================"
echo "Workflow Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Get your YouTube API key: https://console.cloud.google.com/apis/credentials"
echo "  2. Run: idearank setup"
echo "  3. Run: idearank process https://youtube.com/@yourchannel"
echo "  4. Analyze results in the SQLite database"
echo ""
echo "For more information:"
echo "  - CLI Guide: cat CLI_GUIDE.md"
echo "  - Python API: cat QUICKSTART.md"
echo "  - Architecture: cat ARCHITECTURE.md"
echo ""
echo "Happy ranking! 🚀"


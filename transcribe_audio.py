#!/usr/bin/env python3
"""Quick script to transcribe audio file using Whisper."""

import sys
from pathlib import Path

try:
    import whisper
except ImportError:
    print("Error: whisper package not installed")
    print("Install with: pip install openai-whisper")
    sys.exit(1)

def transcribe_to_markdown(audio_path: str, output_path: str = None, model_name: str = "base"):
    """Transcribe audio file and save to markdown.
    
    Args:
        audio_path: Path to audio file
        output_path: Optional output path for markdown file
        model_name: Whisper model to use (tiny, base, small, medium, large)
    """
    audio_path = Path(audio_path).expanduser()
    
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)
    
    # Default output path: same name as audio file but with .md extension
    if output_path is None:
        output_path = audio_path.with_suffix('.md')
    else:
        output_path = Path(output_path).expanduser()
    
    print(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)
    
    print(f"Transcribing: {audio_path}")
    result = model.transcribe(str(audio_path))
    
    # Create markdown content
    markdown_content = f"""# Transcription: {audio_path.name}

**Source:** `{audio_path}`  
**Date:** {audio_path.stat().st_mtime}  
**Model:** {model_name}

---

## Transcript

{result['text']}

---

## Segments

"""
    
    # Add timestamped segments
    for segment in result['segments']:
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text'].strip()
        
        # Format time as MM:SS
        start_min, start_sec = divmod(int(start_time), 60)
        end_min, end_sec = divmod(int(end_time), 60)
        
        markdown_content += f"**[{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}]** {text}\n\n"
    
    # Write to file
    output_path.write_text(markdown_content)
    print(f"\n✓ Transcription saved to: {output_path}")
    print(f"  Total duration: {int(result['segments'][-1]['end'] // 60)} minutes")
    print(f"  Word count: ~{len(result['text'].split())} words")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe_audio.py <audio_file> [output_file] [model]")
        print("\nModels: tiny, base (default), small, medium, large")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    model = sys.argv[3] if len(sys.argv) > 3 else "base"
    
    transcribe_to_markdown(audio_file, output_file, model)


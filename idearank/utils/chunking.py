"""Document chunking utilities for handling long-form content.

Splits large documents into smaller chunks that can be individually embedded
while maintaining parent-child relationships.
"""

import re
from typing import List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A chunk of a larger document."""
    
    text: str
    parent_id: str  # ID of the parent content item
    chunk_index: int  # Position in the sequence (0-based)
    total_chunks: int  # Total number of chunks for this parent
    char_start: int  # Character offset in original document
    char_end: int  # Character offset in original document
    
    @property
    def chunk_id(self) -> str:
        """Generate a unique ID for this chunk."""
        return f"{self.parent_id}_chunk_{self.chunk_index}"


class DocumentChunker:
    """Splits long documents into smaller, embeddable chunks."""
    
    def __init__(
        self,
        chunk_size: int = 8000,  # ~2000 tokens for OpenAI
        chunk_overlap: int = 500,  # Overlap between chunks for context
        respect_paragraphs: bool = True,
    ):
        """Initialize document chunker.
        
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            respect_paragraphs: Try to break at paragraph boundaries
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.respect_paragraphs = respect_paragraphs
    
    def chunk_document(
        self,
        text: str,
        parent_id: str,
    ) -> List[DocumentChunk]:
        """Split a document into chunks.
        
        Args:
            text: The full document text
            parent_id: ID of the parent content item
            
        Returns:
            List of DocumentChunk objects
        """
        if not text or len(text) <= self.chunk_size:
            # Document is small enough, return as single chunk
            return [DocumentChunk(
                text=text,
                parent_id=parent_id,
                chunk_index=0,
                total_chunks=1,
                char_start=0,
                char_end=len(text),
            )]
        
        # Split into chunks
        if self.respect_paragraphs:
            chunks = self._chunk_by_paragraphs(text)
        else:
            chunks = self._chunk_by_size(text)
        
        # Create DocumentChunk objects
        total_chunks = len(chunks)
        document_chunks = []
        
        for i, (chunk_text, char_start, char_end) in enumerate(chunks):
            # Safety check: if chunk is still too large (in tokens), force split it
            chunk_tokens = estimate_tokens(chunk_text)
            if chunk_tokens > 7000:  # 7000 tokens = ~28k chars worst case
                logger.warning(
                    f"Chunk {i} of {parent_id} has {chunk_tokens} tokens, "
                    f"force-splitting to stay under token limit"
                )
                # Force split by fixed character size (no respect for boundaries)
                sub_chunks = self._force_split_large_chunk(chunk_text, char_start)
                for sub_text, sub_start, sub_end in sub_chunks:
                    document_chunks.append(DocumentChunk(
                        text=sub_text,
                        parent_id=parent_id,
                        chunk_index=len(document_chunks),
                        total_chunks=0,  # Will update later
                        char_start=sub_start,
                        char_end=sub_end,
                    ))
            else:
                document_chunks.append(DocumentChunk(
                    text=chunk_text,
                    parent_id=parent_id,
                    chunk_index=i,
                    total_chunks=total_chunks,
                    char_start=char_start,
                    char_end=char_end,
                ))
        
        # Update total_chunks count for all chunks
        actual_total = len(document_chunks)
        for chunk in document_chunks:
            chunk.total_chunks = actual_total
            chunk.chunk_index = document_chunks.index(chunk)
        
        logger.info(f"Split document {parent_id} into {actual_total} chunks")
        return document_chunks
    
    def _chunk_by_paragraphs(self, text: str) -> List[tuple[str, int, int]]:
        """Split text into chunks, respecting paragraph boundaries.
        
        Returns:
            List of (chunk_text, char_start, char_end) tuples
        """
        # Split into paragraphs (double newline or single newline followed by indent)
        paragraphs = re.split(r'\n\s*\n|\n(?=\s{2,})', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        current_start = 0
        char_offset = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            # If a single paragraph is too large, split it by sentences
            if para_size > self.chunk_size:
                # First, save any accumulated content
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append((chunk_text, current_start, char_offset))
                    current_chunk = []
                    current_size = 0
                
                # Split large paragraph by sentences
                sentence_chunks = self._split_large_paragraph(para, char_offset)
                chunks.extend(sentence_chunks)
                char_offset += para_size + 2  # +2 for \n\n
                current_start = char_offset
                continue
            
            # Check if adding this paragraph would exceed chunk size
            if current_size + para_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append((chunk_text, current_start, char_offset))
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk:
                    # Include last paragraph from previous chunk for context
                    overlap_text = current_chunk[-1]
                    if len(overlap_text) <= self.chunk_overlap:
                        current_chunk = [overlap_text]
                        current_size = len(overlap_text)
                        current_start = char_offset - len(overlap_text) - 2
                    else:
                        current_chunk = []
                        current_size = 0
                        current_start = char_offset
                else:
                    current_chunk = []
                    current_size = 0
                    current_start = char_offset
            
            # Add paragraph to current chunk
            current_chunk.append(para)
            current_size += para_size
            char_offset += para_size + 2  # +2 for \n\n separator
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append((chunk_text, current_start, len(text)))
        
        return chunks
    
    def _split_large_paragraph(
        self,
        para: str,
        start_offset: int,
    ) -> List[tuple[str, int, int]]:
        """Split a large paragraph by sentences.
        
        Args:
            para: Paragraph text
            start_offset: Character offset in original document
            
        Returns:
            List of (chunk_text, char_start, char_end) tuples
        """
        # Split by sentences (rough approximation)
        sentences = re.split(r'(?<=[.!?])\s+', para)
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_start = start_offset
        char_offset = start_offset
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append((chunk_text, chunk_start, char_offset))
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-1]
                    if len(overlap_text) <= self.chunk_overlap:
                        current_chunk = [overlap_text]
                        current_size = len(overlap_text)
                        chunk_start = char_offset - len(overlap_text) - 1
                    else:
                        current_chunk = []
                        current_size = 0
                        chunk_start = char_offset
                else:
                    current_chunk = []
                    current_size = 0
                    chunk_start = char_offset
            
            current_chunk.append(sentence)
            current_size += sentence_size + 1  # +1 for space
            char_offset += sentence_size + 1
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append((chunk_text, chunk_start, start_offset + len(para)))
        
        return chunks
    
    def _chunk_by_size(self, text: str) -> List[tuple[str, int, int]]:
        """Split text into fixed-size chunks with overlap.
        
        Args:
            text: Full document text
            
        Returns:
            List of (chunk_text, char_start, char_end) tuples
        """
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            
            # Try to break at a word boundary if not at end
            if end < text_len:
                # Look backwards for whitespace
                space_pos = text.rfind(' ', start, end)
                if space_pos > start:
                    end = space_pos
            
            chunk_text = text[start:end].strip()
            if chunk_text:  # Only add non-empty chunks
                chunks.append((chunk_text, start, end))
            
            # Move start position with overlap
            start = end - self.chunk_overlap if end < text_len else text_len
            
            # Ensure we make progress
            if start <= chunks[-1][1] if chunks else -1:
                start = end
        
        return chunks
    
    def _force_split_large_chunk(
        self,
        text: str,
        start_offset: int,
    ) -> List[tuple[str, int, int]]:
        """Force split an oversized chunk into smaller pieces.
        
        This is a last resort when a chunk is too large even after
        paragraph and sentence splitting. Splits by fixed character size.
        
        Args:
            text: Chunk text to split
            start_offset: Character offset in original document
            
        Returns:
            List of (chunk_text, char_start, char_end) tuples
        """
        # Use a very conservative size: 2500 chars (~625 tokens minimum)
        max_size = 2500
        chunks = []
        pos = 0
        text_len = len(text)
        
        while pos < text_len:
            end = min(pos + max_size, text_len)
            
            # Try to break at a word boundary
            if end < text_len:
                space_pos = text.rfind(' ', pos, end)
                if space_pos > pos:
                    end = space_pos
            
            chunk_text = text[pos:end].strip()
            if chunk_text:
                chunks.append((
                    chunk_text,
                    start_offset + pos,
                    start_offset + end
                ))
            
            pos = end
            # Ensure we make progress even if no word boundary found
            if pos == chunks[-1][1] - start_offset if chunks else -1:
                pos = end + 1
        
        logger.info(f"Force-split large chunk into {len(chunks)} smaller chunks")
        return chunks


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Estimate the number of tokens in a text.
    
    Args:
        text: Input text
        chars_per_token: Average characters per token (4 for English)
        
    Returns:
        Estimated token count
    """
    return int(len(text) / chars_per_token)


def should_chunk_for_model(
    text: str,
    model_name: str = "text-embedding-3-small",
) -> bool:
    """Determine if text needs chunking for a given embedding model.
    
    Args:
        text: Input text
        model_name: Embedding model name
        
    Returns:
        True if text should be chunked
    """
    # Model token limits (conservative estimates)
    model_limits = {
        "text-embedding-3-small": 8000,
        "text-embedding-3-large": 8000,
        "text-embedding-ada-002": 8000,
        "sentence-transformers": 512,  # Most ST models have 512 token limit
    }
    
    # Default to conservative limit
    token_limit = model_limits.get(model_name, 512)
    
    # Use 80% of limit as threshold for safety
    threshold = int(token_limit * 0.8)
    
    estimated_tokens = estimate_tokens(text)
    return estimated_tokens > threshold


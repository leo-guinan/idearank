"""Semantic content extraction - decompose content into actors, events, and changes.

Instead of chunking text arbitrarily by character count, we extract meaningful
semantic units that represent the actual structure of knowledge:
- Actors: entities, people, concepts, companies
- Events: things that happen, actions taken
- Changes: transformations experienced by actors

This enables:
1. Better embedding (semantic units vs arbitrary chunks)
2. Knowledge graph construction
3. Content reconstruction from semantic primitives
4. No token limit issues (semantic units are naturally smaller)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class Actor:
    """An entity that experiences changes or participates in events."""
    
    name: str
    type: str  # person, company, concept, technology, etc.
    description: str
    aliases: List[str] = None
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []


@dataclass
class Event:
    """Something that happens involving actors."""
    
    description: str
    actors: List[str]  # Names of actors involved
    timestamp: Optional[str] = None  # When it happened (if mentioned)
    context: str = ""  # Additional context
    
    def to_text(self) -> str:
        """Convert event to embeddable text."""
        text = f"Event: {self.description}\n"
        if self.actors:
            text += f"Involving: {', '.join(self.actors)}\n"
        if self.timestamp:
            text += f"When: {self.timestamp}\n"
        if self.context:
            text += f"Context: {self.context}"
        return text


@dataclass
class Change:
    """A transformation or change experienced by an actor."""
    
    actor: str
    before_state: str
    after_state: str
    trigger: str  # What caused the change
    context: str = ""
    
    def to_text(self) -> str:
        """Convert change to embeddable text."""
        return (
            f"Change for {self.actor}:\n"
            f"Before: {self.before_state}\n"
            f"After: {self.after_state}\n"
            f"Trigger: {self.trigger}\n"
            f"{f'Context: {self.context}' if self.context else ''}"
        )


@dataclass
class SemanticStructure:
    """Complete semantic decomposition of content."""
    
    content_id: str
    actors: List[Actor]
    events: List[Event]
    changes: List[Change]
    summary: str
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def get_embeddable_units(self) -> List[tuple[str, str, str]]:
        """Get all semantic units as embeddable text.
        
        Returns:
            List of (unit_type, unit_id, text) tuples
        """
        units = []
        
        # Summary as primary unit
        units.append(("summary", f"{self.content_id}_summary", self.summary))
        
        # Each actor as a unit
        for i, actor in enumerate(self.actors):
            text = f"Actor: {actor.name} ({actor.type})\n{actor.description}"
            if actor.aliases:
                text += f"\nAlso known as: {', '.join(actor.aliases)}"
            units.append(("actor", f"{self.content_id}_actor_{i}", text))
        
        # Each event as a unit
        for i, event in enumerate(self.events):
            units.append(("event", f"{self.content_id}_event_{i}", event.to_text()))
        
        # Each change as a unit
        for i, change in enumerate(self.changes):
            units.append(("change", f"{self.content_id}_change_{i}", change.to_text()))
        
        return units


class SemanticExtractor:
    """Extract semantic structure from content using LLM."""
    
    def __init__(self, api_key: str, model: str = "gpt-5-nano"):
        """Initialize with OpenAI API key."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def extract(self, content_id: str, text: str, title: str = "") -> SemanticStructure:
        """Extract semantic structure from content.
        
        Args:
            content_id: Unique identifier for this content
            text: Full text to analyze
            title: Optional title/heading
            
        Returns:
            SemanticStructure with actors, events, and changes
        """
        logger.info(f"Extracting semantic structure from {content_id}")
        
        # Build prompt for extraction
        prompt = self._build_extraction_prompt(text, title)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing content and extracting semantic structure. You identify actors (entities, people, concepts), events (things that happen), and changes (transformations). Always respond with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return self._parse_extraction(content_id, result)
            
        except Exception as e:
            logger.error(f"Failed to extract semantic structure: {e}")
            # Return minimal structure on failure
            return SemanticStructure(
                content_id=content_id,
                actors=[],
                events=[],
                changes=[],
                summary=text[:500] if len(text) > 500 else text,
                metadata={"extraction_failed": True, "error": str(e)}
            )
    
    def _build_extraction_prompt(self, text: str, title: str = "") -> str:
        """Build the extraction prompt."""
        # Truncate text if too long for the extraction prompt
        max_chars = 50000  # ~12k tokens
        if len(text) > max_chars:
            text = text[:max_chars] + "\n[...content truncated...]"
        
        prompt = f"""Analyze the following content and extract its semantic structure.

{f'Title: {title}' if title else ''}

Content:
{text}

Extract:

1. **Actors** - Key entities, people, concepts, companies, or technologies mentioned. For each:
   - name: The primary name
   - type: person/company/concept/technology/other
   - description: What/who they are (1-2 sentences)
   - aliases: Other names they're referred to as

2. **Events** - Significant things that happen. For each:
   - description: What happened (1-2 sentences)
   - actors: Names of actors involved
   - timestamp: When it happened (if mentioned)
   - context: Additional relevant context

3. **Changes** - Transformations or changes experienced by actors. For each:
   - actor: Name of actor who changed
   - before_state: Their state before
   - after_state: Their state after
   - trigger: What caused the change
   - context: Additional context

4. **Summary** - A 2-3 sentence summary of the entire content

Return valid JSON in this exact structure:
{{
    "summary": "string",
    "actors": [
        {{"name": "string", "type": "string", "description": "string", "aliases": ["string"]}}
    ],
    "events": [
        {{"description": "string", "actors": ["string"], "timestamp": "string or null", "context": "string"}}
    ],
    "changes": [
        {{"actor": "string", "before_state": "string", "after_state": "string", "trigger": "string", "context": "string"}}
    ]
}}

Focus on extracting meaningful semantic units. Each unit should be self-contained and make sense on its own."""
        
        return prompt
    
    def _parse_extraction(self, content_id: str, result: Dict) -> SemanticStructure:
        """Parse extraction result into SemanticStructure."""
        actors = [
            Actor(
                name=a.get("name", "Unknown"),
                type=a.get("type", "unknown"),
                description=a.get("description", ""),
                aliases=a.get("aliases", [])
            )
            for a in result.get("actors", [])
        ]
        
        events = [
            Event(
                description=e.get("description", ""),
                actors=e.get("actors", []),
                timestamp=e.get("timestamp"),
                context=e.get("context", "")
            )
            for e in result.get("events", [])
        ]
        
        changes = [
            Change(
                actor=c.get("actor", "Unknown"),
                before_state=c.get("before_state", ""),
                after_state=c.get("after_state", ""),
                trigger=c.get("trigger", ""),
                context=c.get("context", "")
            )
            for c in result.get("changes", [])
        ]
        
        return SemanticStructure(
            content_id=content_id,
            actors=actors,
            events=events,
            changes=changes,
            summary=result.get("summary", ""),
            metadata={"extraction_method": "gpt-5-nano"}
        )


class FallbackSemanticExtractor:
    """Simple rule-based fallback when LLM extraction isn't available."""
    
    def extract(self, content_id: str, text: str, title: str = "") -> SemanticStructure:
        """Extract basic semantic structure using simple heuristics."""
        logger.info(f"Using fallback semantic extraction for {content_id}")
        
        # Create a simple summary (first paragraph or first 500 chars)
        paragraphs = text.split('\n\n')
        summary = paragraphs[0] if paragraphs else text[:500]
        
        # Simple actor extraction: look for proper nouns (capitalized words)
        import re
        potential_actors = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        # Deduplicate and take most frequent
        from collections import Counter
        actor_counts = Counter(potential_actors)
        top_actors = [
            Actor(
                name=name,
                type="unknown",
                description=f"Mentioned {count} times in content",
                aliases=[]
            )
            for name, count in actor_counts.most_common(10)
        ]
        
        return SemanticStructure(
            content_id=content_id,
            actors=top_actors,
            events=[],  # Would need NLP for this
            changes=[],  # Would need NLP for this
            summary=summary,
            metadata={"extraction_method": "fallback_heuristic"}
        )


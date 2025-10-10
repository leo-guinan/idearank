"""Citation parser for IdeaRank Trust factor.

Instead of counting links, this parser:
1. Identifies references to people and institutions
2. Extracts the ideas attributed to them
3. Optionally validates connections using AI

This rewards intellectual honesty and proper attribution.
"""

import re
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Optional imports
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spacy not installed. Install with: pip install spacy && python -m spacy download en_core_web_sm")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai not installed. Install with: pip install openai")


@dataclass
class EntityCitation:
    """Represents a citation of an entity (person/institution) with an idea."""
    
    entity_name: str  # Name of person or institution
    entity_type: str  # "PERSON" or "ORG"
    idea_snippet: str  # The idea attributed to them
    context: str  # Surrounding text for verification
    confidence: float = 1.0  # Confidence in extraction (0-1)
    validated: Optional[bool] = None  # AI validation result
    validation_confidence: Optional[float] = None  # AI confidence


@dataclass
class CitationAnalysis:
    """Complete citation analysis for a text."""
    
    entity_citations: List[EntityCitation]
    unique_entities: int
    unique_institutions: int
    total_attributions: int
    attribution_density: float  # Citations per 1000 words
    validation_accuracy: Optional[float] = None  # % of validated citations that are accurate
    
    @property
    def trust_score(self) -> float:
        """Calculate trust score from citations.
        
        Returns:
            Trust score between 0 and 1
        """
        if self.total_attributions == 0:
            return 0.3  # Low trust for no attributions
        
        # Base score from attribution density
        # Target: ~2-5 citations per 1000 words
        density_score = min(1.0, self.attribution_density / 5.0)
        
        # Bonus for entity diversity
        diversity_bonus = min(0.2, self.unique_entities * 0.02)
        
        # Bonus for institutional citations
        institution_bonus = min(0.1, self.unique_institutions * 0.05)
        
        base_score = 0.5 + (density_score * 0.3) + diversity_bonus + institution_bonus
        
        # Apply validation if available
        if self.validation_accuracy is not None:
            base_score = base_score * (0.5 + 0.5 * self.validation_accuracy)
        
        return min(1.0, max(0.0, base_score))


class EntityIdeaCitationParser:
    """Parses text to extract entity-idea attributions."""
    
    def __init__(self, use_spacy: bool = True):
        """Initialize citation parser.
        
        Args:
            use_spacy: Use spaCy for entity extraction (better quality)
        """
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        
        if self.use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model for entity extraction")
            except:
                logger.warning("Failed to load spaCy model. Falling back to regex.")
                self.use_spacy = False
                self.nlp = None
        else:
            self.nlp = None
    
    def parse(self, text: str) -> CitationAnalysis:
        """Parse text to extract entity-idea citations.
        
        Args:
            text: Text to analyze
            
        Returns:
            CitationAnalysis with extracted citations
        """
        word_count = len(text.split())
        
        if self.use_spacy:
            citations = self._extract_with_spacy(text)
        else:
            citations = self._extract_with_regex(text)
        
        # Calculate statistics
        unique_entities = len(set(c.entity_name for c in citations))
        unique_institutions = len(set(c.entity_name for c in citations if c.entity_type == "ORG"))
        total_attributions = len(citations)
        density = (total_attributions / word_count * 1000) if word_count > 0 else 0
        
        return CitationAnalysis(
            entity_citations=citations,
            unique_entities=unique_entities,
            unique_institutions=unique_institutions,
            total_attributions=total_attributions,
            attribution_density=density,
        )
    
    def _extract_with_spacy(self, text: str) -> List[EntityCitation]:
        """Extract citations using spaCy NER.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of EntityCitation objects
        """
        doc = self.nlp(text)
        citations = []
        
        # Find all entities (PERSON and ORG)
        for ent in doc.ents:
            if ent.label_ not in ["PERSON", "ORG"]:
                continue
            
            # Get context around the entity (±100 chars)
            start_idx = max(0, ent.start_char - 100)
            end_idx = min(len(text), ent.end_char + 100)
            context = text[start_idx:end_idx]
            
            # Extract the idea attributed to this entity
            idea = self._extract_idea_from_context(ent.text, context)
            
            if idea:
                citations.append(EntityCitation(
                    entity_name=ent.text,
                    entity_type=ent.label_,
                    idea_snippet=idea,
                    context=context,
                    confidence=0.9 if self.use_spacy else 0.6,
                ))
        
        return citations
    
    def _extract_with_regex(self, text: str) -> List[EntityCitation]:
        """Extract citations using regex patterns (fallback).
        
        Args:
            text: Text to analyze
            
        Returns:
            List of EntityCitation objects
        """
        citations = []
        
        # Common attribution patterns
        patterns = [
            # "According to Einstein, ..."
            r"(?:according to|as|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),?\s+(.{20,150})",
            
            # "Einstein said/argued/proposed that ..."
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:said|argued|proposed|showed|demonstrated|proved|found)\s+(?:that\s+)?(.{20,150})",
            
            # "Einstein's theory of ..."
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'s\s+(?:theory|concept|idea|work|research|study)\s+(?:of|on)\s+(.{20,100})",
            
            # "(Smith, 2024)" - academic citation
            r"\(([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*\d{4}\)",
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity = match.group(1).strip()
                idea = match.group(2).strip() if len(match.groups()) > 1 else "mentioned"
                
                # Get context
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                citations.append(EntityCitation(
                    entity_name=entity,
                    entity_type="PERSON",  # Regex can't distinguish
                    idea_snippet=idea[:150],  # Limit length
                    context=context,
                    confidence=0.6,  # Lower confidence for regex
                ))
        
        return citations
    
    def _extract_idea_from_context(self, entity: str, context: str) -> Optional[str]:
        """Extract the specific idea attributed to an entity.
        
        Args:
            entity: Entity name
            context: Context around entity mention
            
        Returns:
            Idea snippet or None
        """
        # Find sentences containing the entity
        sentences = re.split(r'[.!?]', context)
        
        for sentence in sentences:
            if entity in sentence:
                # Look for attribution patterns
                idea_patterns = [
                    r'(?:said|argued|proposed|showed|claimed)\s+(?:that\s+)?(.+)',
                    r"'s\s+(?:theory|concept|idea)\s+(?:of|on|that)\s+(.+)",
                    r'(?:according to|from)\s+' + re.escape(entity) + r',?\s+(.+)',
                ]
                
                for pattern in idea_patterns:
                    match = re.search(pattern, sentence, re.IGNORECASE)
                    if match:
                        return match.group(1).strip()[:150]
        
        # Return a snippet of the context if no specific pattern found
        return context[:100].strip() if context.strip() else None


class CitationValidator:
    """Validates entity-idea connections using AI.
    
    Uses LLM to verify that attributions are accurate.
    """
    
    def __init__(self, openai_api_key: str, model: str = "gpt-4o-mini"):
        """Initialize validator with OpenAI.
        
        Args:
            openai_api_key: OpenAI API key
            model: Model to use (gpt-4o-mini is cheap and good enough)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai required for validation. Install with: pip install openai")
        
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        
        logger.info(f"Initialized citation validator with {model}")
    
    def validate_citation(self, citation: EntityCitation) -> Tuple[bool, float, str]:
        """Validate a single entity-idea attribution.
        
        Args:
            citation: EntityCitation to validate
            
        Returns:
            (is_accurate, confidence, explanation)
        """
        prompt = f"""You are a fact-checker validating citations and attributions.

Entity: {citation.entity_name} ({citation.entity_type})
Attributed idea: {citation.idea_snippet}
Context: {citation.context}

Task: Is this attribution accurate? Did {citation.entity_name} actually propose/discuss this idea?

Respond in this format:
VERDICT: [ACCURATE / INACCURATE / UNCLEAR]
CONFIDENCE: [0.0-1.0]
EXPLANATION: [Brief explanation]"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for consistent fact-checking
                max_tokens=150,
            )
            
            result = response.choices[0].message.content
            
            # Parse response
            verdict = "UNCLEAR"
            confidence = 0.5
            explanation = result
            
            if "VERDICT:" in result:
                verdict_line = [line for line in result.split('\n') if 'VERDICT:' in line][0]
                verdict = verdict_line.split('VERDICT:')[1].strip().split()[0]
            
            if "CONFIDENCE:" in result:
                conf_line = [line for line in result.split('\n') if 'CONFIDENCE:' in line][0]
                try:
                    confidence = float(conf_line.split('CONFIDENCE:')[1].strip().split()[0])
                except:
                    confidence = 0.5
            
            is_accurate = verdict == "ACCURATE"
            
            return is_accurate, confidence, result
            
        except Exception as e:
            logger.error(f"Validation failed for {citation.entity_name}: {e}")
            return False, 0.0, f"Validation error: {e}"
    
    def validate_batch(
        self,
        citations: List[EntityCitation],
        max_validations: int = 10,
    ) -> List[EntityCitation]:
        """Validate multiple citations (respects API budget).
        
        Args:
            citations: List of citations to validate
            max_validations: Maximum number to validate (cost control)
            
        Returns:
            Citations with validation results filled in
        """
        # Validate top N citations (by confidence)
        sorted_citations = sorted(citations, key=lambda c: c.confidence, reverse=True)
        to_validate = sorted_citations[:max_validations]
        
        logger.info(f"Validating {len(to_validate)} citations (max: {max_validations})")
        
        for i, citation in enumerate(to_validate, 1):
            logger.info(f"  Validating {i}/{len(to_validate)}: {citation.entity_name}...")
            
            is_accurate, confidence, explanation = self.validate_citation(citation)
            
            citation.validated = is_accurate
            citation.validation_confidence = confidence
            
            logger.debug(f"    Result: {is_accurate} (confidence: {confidence:.2f})")
        
        return citations


def analyze_citations(
    text: str,
    use_spacy: bool = True,
    validate: bool = False,
    openai_api_key: Optional[str] = None,
    max_validations: int = 5,
) -> CitationAnalysis:
    """Complete citation analysis pipeline.
    
    Args:
        text: Text to analyze
        use_spacy: Use spaCy for entity extraction
        validate: Validate citations with AI
        openai_api_key: OpenAI key for validation
        max_validations: Max citations to validate
        
    Returns:
        CitationAnalysis with trust score
    """
    # Step 1: Extract entity-idea citations
    parser = EntityIdeaCitationParser(use_spacy=use_spacy)
    analysis = parser.parse(text)
    
    logger.info(f"Found {analysis.total_attributions} entity-idea attributions")
    logger.info(f"  Unique entities: {analysis.unique_entities}")
    logger.info(f"  Institutions: {analysis.unique_institutions}")
    logger.info(f"  Density: {analysis.attribution_density:.2f} per 1000 words")
    
    # Step 2: Validate citations (optional)
    if validate and openai_api_key:
        validator = CitationValidator(openai_api_key)
        validated_citations = validator.validate_batch(
            analysis.entity_citations,
            max_validations=max_validations,
        )
        
        # Update analysis with validation results
        analysis.entity_citations = validated_citations
        
        # Calculate validation accuracy
        validated_count = sum(1 for c in validated_citations if c.validated is not None)
        if validated_count > 0:
            accurate_count = sum(1 for c in validated_citations if c.validated is True)
            analysis.validation_accuracy = accurate_count / validated_count
            
            logger.info(f"Validation: {accurate_count}/{validated_count} accurate ({analysis.validation_accuracy:.1%})")
    
    return analysis


def quick_trust_score(text: str, use_spacy: bool = True) -> float:
    """Quick trust score without validation.
    
    Args:
        text: Text to analyze
        use_spacy: Use spaCy for better accuracy
        
    Returns:
        Trust score (0-1)
    """
    analysis = analyze_citations(text, use_spacy=use_spacy, validate=False)
    return analysis.trust_score


# Convenience functions for common patterns
def find_person_attributions(text: str) -> List[Tuple[str, str]]:
    """Find all person attributions in text.
    
    Args:
        text: Text to analyze
        
    Returns:
        List of (person_name, idea) tuples
    """
    parser = EntityIdeaCitationParser(use_spacy=SPACY_AVAILABLE)
    analysis = parser.parse(text)
    
    return [
        (c.entity_name, c.idea_snippet)
        for c in analysis.entity_citations
        if c.entity_type == "PERSON"
    ]


def find_institution_attributions(text: str) -> List[Tuple[str, str]]:
    """Find all institution attributions in text.
    
    Args:
        text: Text to analyze
        
    Returns:
        List of (institution_name, idea) tuples
    """
    parser = EntityIdeaCitationParser(use_spacy=SPACY_AVAILABLE)
    analysis = parser.parse(text)
    
    return [
        (c.entity_name, c.idea_snippet)
        for c in analysis.entity_citations
        if c.entity_type == "ORG"
    ]


def get_attribution_summary(text: str) -> Dict[str, Any]:
    """Get summary of all attributions in text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with attribution statistics
    """
    parser = EntityIdeaCitationParser(use_spacy=SPACY_AVAILABLE)
    analysis = parser.parse(text)
    
    # Group by entity
    entity_ideas = {}
    for citation in analysis.entity_citations:
        if citation.entity_name not in entity_ideas:
            entity_ideas[citation.entity_name] = []
        entity_ideas[citation.entity_name].append(citation.idea_snippet)
    
    return {
        'total_attributions': analysis.total_attributions,
        'unique_entities': analysis.unique_entities,
        'density': analysis.attribution_density,
        'entity_ideas': entity_ideas,
        'trust_score': analysis.trust_score,
    }


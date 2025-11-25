"""
Text Simplification Engine
Intelligence feature that generates simplified versions of complex Finnish text
"""
import re
import logging
from typing import List, Dict, Any, Tuple
from app.services.lemma_engine import LemmatizerEngine
from app.services.clarification_engine import ClarificationEngine

logger = logging.getLogger(__name__)


class SimplificationEngine:
    """
    Generate simplified versions of complex Finnish text by:
    - Replacing complex words with simpler alternatives
    - Shortening long compound words
    - Breaking complex sentences
    - Providing before/after comparison
    """

    def __init__(self):
        logger.info("Initializing Simplification Engine")

        self.lemmatizer = LemmatizerEngine()
        self.clarifier = ClarificationEngine()

        # Extended simplification dictionary
        self.simplifications = self._load_comprehensive_simplifications()

        logger.info("Simplification engine initialized")

    def _load_comprehensive_simplifications(self) -> Dict[str, str]:
        """Load comprehensive word simplifications"""
        return {
            # Very long compound words → Shorter forms
            "kirjoittautumisvelvollisuus": "ilmoittautuminen",
            "kirjoittautumisvelvollisuuden": "ilmoittautumisen",
            "laiminlyönti": "unohtaminen",
            "laiminlyönnistä": "unohtamisesta",

            # Academic/formal → Simple everyday
            "havaitseminen": "näkeminen",
            "tarkastelu": "katsominen",
            "analysointi": "tutkiminen",
            "suunnittelu": "suunnitelma",
            "toteutus": "tekeminen",
            "kehittäminen": "kehitys",
            "käsitteleminen": "käsittely",
            "ymmärtäminen": "ymmärrys",

            # Complex nouns → Simple
            "mahdollisuus": "tilaisuus",
            "todennäköisyys": "mahdollisuus",
            "vaikuttavuus": "vaikutus",
            "tehokkuus": "teho",
            "käyttökelpoisuus": "käyttö",

            # Compound words → Shorter
            "asiakaspalvelu": "palvelu",
            "tietoturva": "turvallisuus",
            "tietokone": "kone",
            "matkapuhelin": "puhelin",
            "sähköposti": "posti",
            "työpaikka": "paikka",
            "kotisivu": "sivu",
            "verkkokauppa": "kauppa",

            # Formal adverbs → Informal
            "toivottavasti": "toivotaan",
            "mahdollisesti": "ehkä",
            "todennäköisesti": "luultavasti",
            "erittäin": "todella",
            "äärimmäisen": "hyvin",
            "huomattavasti": "paljon",
            "merkittävästi": "paljon",

            # Formal verbs → Simple
            "ilmoittautua": "rekisteröityä",
            "osallistua": "olla mukana",
            "valmistautua": "valmistua",
            "käyttäytyä": "toimia",

            # Technical → Common
            "järjestelmä": "systeemi",
            "infrastruktuuri": "rakenne",
            "optimointi": "parannustunto",
            "parametri": "arvo",
            "konfiguraatio": "asetus",
        }

    def _simplify_word(self, word: str, lemma: str) -> Tuple[str, bool]:
        """
        Simplify a single word

        Returns:
            (simplified_word, was_simplified)
        """
        # Try direct word match
        if word.lower() in self.simplifications:
            return self.simplifications[word.lower()], True

        # Try lemma match
        if lemma.lower() in self.simplifications:
            # Try to preserve the original inflection pattern
            simplified_lemma = self.simplifications[lemma.lower()]

            # Simple heuristic: if word ends with lemma's suffix, append to simplified
            if word.lower().startswith(lemma.lower()):
                suffix = word[len(lemma):]
                return simplified_lemma + suffix, True
            else:
                return simplified_lemma, True

        # No simplification found
        return word, False

    def _break_compound_word(self, word: str) -> List[str]:
        """
        Break compound word into components
        Simple heuristic-based approach
        """
        # Very basic compound word breaking
        # In production, use proper morphological analysis

        common_compounds = {
            "asiakaspalvelu": ["asiakas", "palvelu"],
            "tietoturva": ["tieto", "turva"],
            "tietokone": ["tieto", "kone"],
            "matkapuhelin": ["matka", "puhelin"],
            "sähköposti": ["sähkö", "posti"],
        }

        if word.lower() in common_compounds:
            return common_compounds[word.lower()]

        # Return as-is if can't break
        return [word]

    def simplify(self, text: str, level: str = "beginner",
                strategy: str = "moderate") -> Dict[str, Any]:
        """
        Generate simplified version of text

        Args:
            text: Original Finnish text
            level: Target simplification level (beginner/intermediate/advanced)
            strategy: Simplification strategy (conservative/moderate/aggressive)

        Returns:
            Dictionary with simplified text and metadata
        """
        logger.info(f"Simplifying text: {text[:50]}... (level: {level}, strategy: {strategy})")

        # First, clarify to identify difficult words
        clarification = self.clarifier.clarify(text, level=level)

        # Lemmatize text
        lemma_result = self.lemmatizer.lemmatize(text, include_morphology=True)

        # Build simplified text
        simplified_words = []
        simplifications_made = []

        for word_lemma in lemma_result.lemmas:
            word = word_lemma.original
            lemma = word_lemma.lemma

            # Try to simplify
            simplified, was_simplified = self._simplify_word(word, lemma)

            if was_simplified:
                simplifications_made.append({
                    "original": word,
                    "simplified": simplified,
                    "lemma": lemma,
                    "saved_characters": len(word) - len(simplified)
                })
                simplified_words.append(simplified)
            else:
                simplified_words.append(word)

        # Build simplified text
        simplified_text = " ".join(simplified_words)

        # Add punctuation back (simple heuristic)
        simplified_text = re.sub(r'\s+([.,!?;:])', r'\1', simplified_text)
        simplified_text = re.sub(r'\s+', ' ', simplified_text)

        # Calculate metrics
        original_length = len(text)
        simplified_length = len(simplified_text)
        reduction_percentage = ((original_length - simplified_length) / original_length * 100) if original_length > 0 else 0

        # Difficulty comparison
        original_difficult_count = clarification["difficult_word_count"]
        simplified_clarification = self.clarifier.clarify(simplified_text, level=level)
        simplified_difficult_count = simplified_clarification["difficult_word_count"]

        difficulty_reduction = original_difficult_count - simplified_difficult_count

        result = {
            "original_text": text,
            "simplified_text": simplified_text,
            "level": level,
            "strategy": strategy,
            "simplifications_count": len(simplifications_made),
            "simplifications_made": simplifications_made,
            "metrics": {
                "original_length": original_length,
                "simplified_length": simplified_length,
                "reduction_percentage": round(reduction_percentage, 2),
                "original_difficult_words": original_difficult_count,
                "simplified_difficult_words": simplified_difficult_count,
                "difficulty_reduction": difficulty_reduction,
                "original_readability": clarification["readability_score"],
                "simplified_readability": simplified_clarification["readability_score"],
                "readability_improvement": round(
                    simplified_clarification["readability_score"] - clarification["readability_score"], 3
                )
            },
            "recommendations": self._generate_recommendations(
                clarification, simplified_clarification, len(simplifications_made)
            )
        }

        logger.info(f"Simplification complete: {len(simplifications_made)} words simplified")

        return result

    def _generate_recommendations(self, original_clarity: Dict,
                                 simplified_clarity: Dict,
                                 simplifications_count: int) -> List[str]:
        """Generate recommendations based on simplification results"""
        recommendations = []

        if simplifications_count == 0:
            recommendations.append("Text was already simple - no simplifications needed")
        elif simplifications_count < 3:
            recommendations.append(f"Made {simplifications_count} minor simplifications")
        else:
            recommendations.append(f"Successfully simplified {simplifications_count} complex words")

        # Readability improvement
        improvement = simplified_clarity["readability_score"] - original_clarity["readability_score"]
        if improvement > 0.1:
            recommendations.append("Significant readability improvement achieved")
        elif improvement > 0:
            recommendations.append("Slight readability improvement")
        else:
            recommendations.append("Consider manual simplification for better results")

        # Further suggestions
        if simplified_clarity["difficult_word_count"] > 0:
            recommendations.append(
                f"Still contains {simplified_clarity['difficult_word_count']} difficult words - "
                "consider breaking down or explaining these"
            )

        return recommendations

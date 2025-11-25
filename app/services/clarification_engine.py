"""
Text Clarification Engine
Intelligence feature that highlights difficult words and suggests simpler alternatives
"""
import re
import logging
from typing import List, Dict, Any
from app.services.lemma_engine import LemmatizerEngine
from app.services.complexity_engine import ComplexityEngine

logger = logging.getLogger(__name__)


class ClarificationEngine:
    """
    Clarify difficult Finnish text by:
    - Highlighting complex words
    - Suggesting simpler alternatives
    - Calculating readability scores
    - Providing difficulty ratings per word
    """

    def __init__(self):
        logger.info("Initializing Clarification Engine")

        self.lemmatizer = LemmatizerEngine()
        self.complexity_analyzer = ComplexityEngine()

        # Difficulty thresholds
        self.difficulty_criteria = {
            "word_length_hard": 10,  # Words > 10 chars are hard
            "word_length_medium": 7,  # Words 7-10 chars are medium
            "rare_word_threshold": 0.001,  # Frequency threshold
        }

        # Simple word alternatives
        self.simplifications = self._load_simplifications()

        # Common vs rare words
        self.common_words = self._load_common_words()

        logger.info("Clarification engine initialized")

    def _load_simplifications(self) -> Dict[str, str]:
        """Load simplification mappings"""
        return {
            # Complex academic/formal → Simple everyday
            "kirjoittautumisvelvollisuus": "ilmoittautuminen",
            "laiminlyönti": "unohtaminen",
            "ilmoittautuminen": "rekisteröinti",
            "havaitseminen": "näkeminen",
            "käyttäminen": "käyttö",
            "ymmärtäminen": "ymmärrys",
            "tarkastelu": "katsominen",
            "analysointi": "tutkiminen",
            "suunnittelu": "suunnitelma",
            "toteutus": "tekeminen",
            "kehittäminen": "kehitys",

            # Compound words → Simpler forms
            "asiakaspalvelu": "palvelu",
            "tietoturva": "turvallisuus",
            "tietokone": "kone",
            "matkapuhelin": "puhelin",
            "sähköposti": "posti",

            # Formal → Informal
            "toivottavasti": "toivotaan",
            "mahdollisesti": "ehkä",
            "todennäköisesti": "luultavasti",
            "erittäin": "todella",
            "äärimmäisen": "hyvin",
        }

    def _load_common_words(self) -> set:
        """Load set of common Finnish words"""
        return {
            # Top 100 most common Finnish words
            "ja", "on", "ei", "se", "että", "olla", "hän", "tämä", "joka",
            "kun", "niin", "mutta", "kaikki", "mitä", "vain", "sen", "jos",
            "ne", "nyt", "vielä", "sitten", "kuin", "myös", "minä", "sinä",
            "kissa", "koira", "talo", "auto", "ihminen", "päivä", "aika",
            "vuosi", "vuotta", "maa", "vesi", "kuu", "aurinko", "yö",
            "hyvä", "iso", "pieni", "uusi", "vanha", "mies", "nainen",
            "lapsi", "äiti", "isä", "ystävä", "koti", "kaupunki", "katu",
            "tie", "puu", "kukka", "lintu", "kala", "kieli", "kirja",
            "koulu", "työ", "raha", "ruoka", "kahvi", "leipä", "maito",
        }

    def _is_difficult_word(self, word: str, lemma: str) -> tuple[bool, str, int]:
        """
        Determine if word is difficult

        Returns:
            (is_difficult, reason, difficulty_score)
            difficulty_score: 1 (easy) to 3 (hard)
        """
        # Length-based difficulty
        if len(word) > self.difficulty_criteria["word_length_hard"]:
            return True, "long_word", 3

        if len(word) > self.difficulty_criteria["word_length_medium"]:
            # Check if it's a common word
            if lemma.lower() not in self.common_words:
                return True, "uncommon_word", 2

        # Check if it's a rare/uncommon word
        if lemma.lower() not in self.common_words and len(word) > 6:
            return True, "uncommon_word", 2

        return False, "common_word", 1

    def _suggest_alternative(self, word: str, lemma: str) -> str:
        """Suggest simpler alternative"""
        # Check direct simplifications
        if lemma.lower() in self.simplifications:
            return self.simplifications[lemma.lower()]

        # Check if word itself has simplification
        if word.lower() in self.simplifications:
            return self.simplifications[word.lower()]

        # No alternative found
        return None

    def clarify(self, text: str, level: str = "beginner") -> Dict[str, Any]:
        """
        Clarify text by highlighting difficult words

        Args:
            text: Finnish text to clarify
            level: Target audience level (beginner/intermediate/advanced)

        Returns:
            Dictionary with clarification information
        """
        logger.info(f"Clarifying text: {text[:50]}... (level: {level})")

        # Lemmatize text
        lemma_result = self.lemmatizer.lemmatize(text, include_morphology=True)

        # Analyze overall complexity
        complexity = self.complexity_analyzer.analyze(text, detailed=True)

        # Analyze each word
        difficult_words = []
        word_details = []

        for word_lemma in lemma_result.lemmas:
            word = word_lemma.original
            lemma = word_lemma.lemma
            pos = word_lemma.pos or "NOUN"

            # Check difficulty
            is_difficult, reason, difficulty_score = self._is_difficult_word(word, lemma)

            # Get alternative if difficult
            alternative = None
            if is_difficult:
                alternative = self._suggest_alternative(word, lemma)

            # Build word detail
            word_detail = {
                "word": word,
                "lemma": lemma,
                "pos": pos,
                "is_difficult": is_difficult,
                "difficulty_score": difficulty_score,
                "reason": reason,
                "alternative": alternative,
                "length": len(word)
            }

            word_details.append(word_detail)

            if is_difficult:
                difficult_words.append(word_detail)

        # Calculate readability metrics
        total_words = len(word_details)
        difficult_count = len(difficult_words)
        readability_score = 1.0 - (difficult_count / total_words) if total_words > 0 else 1.0

        # Readability rating
        if readability_score >= 0.85:
            readability_rating = "easy"
        elif readability_score >= 0.70:
            readability_rating = "moderate"
        elif readability_score >= 0.50:
            readability_rating = "challenging"
        else:
            readability_rating = "difficult"

        # Target audience appropriateness
        target_appropriate = self._check_target_appropriateness(
            difficult_count, total_words, level
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            difficult_words, readability_rating, level
        )

        result = {
            "text": text,
            "level": level,
            "word_count": total_words,
            "difficult_word_count": difficult_count,
            "readability_score": round(readability_score, 3),
            "readability_rating": readability_rating,
            "target_appropriate": target_appropriate,
            "difficult_words": difficult_words,
            "word_details": word_details,
            "recommendations": recommendations,
            "complexity_metrics": {
                "average_word_length": complexity.average_word_length,
                "max_word_length": complexity.max_word_length,
                "complexity_rating": complexity.complexity_rating
            }
        }

        logger.info(f"Clarification complete: {difficult_count}/{total_words} difficult words")

        return result

    def _check_target_appropriateness(self, difficult_count: int,
                                     total_words: int, level: str) -> bool:
        """Check if text is appropriate for target level"""
        if total_words == 0:
            return True

        difficulty_ratio = difficult_count / total_words

        # Thresholds per level
        thresholds = {
            "beginner": 0.15,  # Max 15% difficult words
            "intermediate": 0.35,  # Max 35% difficult words
            "advanced": 0.60,  # Max 60% difficult words
        }

        threshold = thresholds.get(level, 0.35)
        return difficulty_ratio <= threshold

    def _generate_recommendations(self, difficult_words: List[Dict],
                                 readability_rating: str, level: str) -> List[str]:
        """Generate recommendations for improving text clarity"""
        recommendations = []

        if readability_rating == "easy":
            recommendations.append("Text is clear and easy to understand")
        elif readability_rating == "moderate":
            recommendations.append("Text is moderately complex - some words may need clarification")
        else:
            recommendations.append("Text is challenging - consider simplifying difficult words")

        # Specific word recommendations
        if difficult_words:
            words_with_alternatives = [w for w in difficult_words if w.get("alternative")]
            if words_with_alternatives:
                recommendations.append(
                    f"Consider replacing {len(words_with_alternatives)} difficult words with simpler alternatives"
                )

        # Level-specific recommendations
        if level == "beginner":
            long_words = [w for w in difficult_words if w["length"] > 10]
            if long_words:
                recommendations.append(
                    f"Beginners may struggle with {len(long_words)} very long words - break them down or explain"
                )

        return recommendations

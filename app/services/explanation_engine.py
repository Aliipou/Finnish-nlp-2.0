"""
Linguistic Explanation Engine
High-level intelligence feature for educational purposes
Provides comprehensive morphological and syntactic explanations
"""
import re
import logging
from typing import List, Dict, Any
from app.services.lemma_engine import LemmatizerEngine
from app.services.complexity_engine import ComplexityEngine

logger = logging.getLogger(__name__)


class ExplanationEngine:
    """
    Provide detailed linguistic explanations of Finnish sentences
    Educational tool for language learners
    """

    def __init__(self):
        logger.info("Initializing Linguistic Explanation Engine")

        # Initialize analysis engines
        self.lemmatizer = LemmatizerEngine()
        self.complexity_analyzer = ComplexityEngine()

        # Frequency data (simplified - in production use real corpus frequencies)
        self.frequency_data = self._load_frequency_data()

        # Difficulty ratings
        self.difficulty_levels = {
            'beginner': {'word_length': 6, 'case_variety': 2},
            'intermediate': {'word_length': 9, 'case_variety': 4},
            'advanced': {'word_length': 12, 'case_variety': 6}
        }

        logger.info("Explanation engine initialized")

    def _load_frequency_data(self) -> Dict[str, str]:
        """Load word frequency ratings"""
        return {
            # Very common words
            'ja': 'very_common', 'on': 'very_common', 'ei': 'very_common',
            'se': 'very_common', 'että': 'very_common', 'olla': 'very_common',
            'kissa': 'common', 'koira': 'common', 'talo': 'common',
            'auto': 'common', 'päivä': 'common', 'yö': 'common',
            # Less common
            'puutarha': 'uncommon', 'kirjoittautumisvelvollisuus': 'rare',
        }

    def _simplify_text(self, text: str) -> str:
        """
        Create simplified version of text

        Args:
            text: Original text

        Returns:
            Simplified version
        """
        # Simple simplification: remove complex compound words, shorten sentences
        # In production, use more sophisticated simplification

        simplified = text

        # Replace compound words with simpler alternatives
        replacements = {
            'kirjoittautumisvelvollisuuden': 'ilmoittautumisen',
            'laiminlyönti': 'unohtaminen',
            'nopeasti': 'nope',
        }

        for complex_word, simple_word in replacements.items():
            simplified = simplified.replace(complex_word, simple_word)

        return simplified

    def _get_word_frequency(self, lemma: str) -> str:
        """Get frequency rating for word"""
        return self.frequency_data.get(lemma.lower(), 'common')

    def _determine_word_difficulty(self, word: str, lemma: str, morphology: Dict) -> str:
        """
        Determine difficulty level of word

        Args:
            word: Original word
            lemma: Lemmatized form
            morphology: Morphological features

        Returns:
            Difficulty level: beginner, intermediate, advanced
        """
        factors = []

        # Length factor
        if len(word) > 12:
            factors.append('advanced')
        elif len(word) > 8:
            factors.append('intermediate')
        else:
            factors.append('beginner')

        # Frequency factor
        freq = self._get_word_frequency(lemma)
        if freq == 'rare':
            factors.append('advanced')
        elif freq == 'uncommon':
            factors.append('intermediate')

        # Case complexity
        if morphology:
            case = morphology.get('case', 'Nominative')
            if case in ['Translative', 'Prolative', 'Instructive']:
                factors.append('advanced')
            elif case in ['Illative', 'Elative', 'Ablative']:
                factors.append('intermediate')

        # Determine overall level
        if 'advanced' in factors:
            return 'advanced'
        elif 'intermediate' in factors:
            return 'intermediate'
        else:
            return 'beginner'

    def _generate_learning_tip(self, word: str, lemma: str, morphology: Dict, pos: str) -> str:
        """
        Generate learning tip for word

        Args:
            word: Original word
            lemma: Lemmatized form
            morphology: Morphological features
            pos: Part of speech

        Returns:
            Learning tip string
        """
        tips = []

        # Possessive suffix tip
        if word.endswith('ni') and len(word) > 3:
            tips.append("Possessive suffix -ni means 'my'")

        # Plural tip
        if morphology and morphology.get('number') == 'Plural':
            tips.append("Plural form (multiple items)")

        # Case-specific tips
        if morphology:
            case = morphology.get('case', '')
            case_tips = {
                'Genitive': "Genitive case shows possession or relationship",
                'Partitive': "Partitive case for partial objects or uncountable things",
                'Inessive': "Inessive case (-ssa/-ssä) means 'in/inside'",
                'Elative': "Elative case (-sta/-stä) means 'from inside'",
                'Illative': "Illative case means 'into'",
                'Adessive': "Adessive case (-lla/-llä) means 'on/at'",
                'Ablative': "Ablative case (-lta/-ltä) means 'from on'",
                'Allative': "Allative case (-lle) means 'onto/to'",
                'Translative': "Translative case (-ksi) shows change/transformation",
            }
            if case in case_tips:
                tips.append(case_tips[case])

        # Verb tips
        if pos == 'VERB':
            tips.append("Finnish verbs conjugate by person and tense")

        return '; '.join(tips) if tips else "Common Finnish word"

    def _breakdown_word(self, word: str, lemma: str, morphology: Dict, pos: str) -> Dict[str, Any]:
        """
        Create detailed morphological breakdown

        Returns:
            Dictionary with breakdown information
        """
        breakdown = {
            'stem': lemma,
            'inflections': []
        }

        if morphology:
            # Case
            if morphology.get('case') and morphology['case'] != 'Nominative':
                breakdown['inflections'].append({
                    'type': 'case',
                    'value': morphology['case'],
                    'meaning': f"In {morphology['case'].lower()} case"
                })

            # Number
            if morphology.get('number') == 'Plural':
                breakdown['inflections'].append({
                    'type': 'number',
                    'value': 'Plural',
                    'meaning': 'Multiple items'
                })

            # Possessive
            if word.endswith(('ni', 'si', 'nsa', 'nsä', 'mme', 'nne')):
                possessive_map = {
                    'ni': 'my', 'si': 'your (sg)', 'nsa': 'his/her/its',
                    'nsä': 'his/her/its', 'mme': 'our', 'nne': 'your (pl)'
                }
                for suffix, meaning in possessive_map.items():
                    if word.endswith(suffix):
                        breakdown['inflections'].append({
                            'type': 'possessive',
                            'value': suffix,
                            'meaning': meaning
                        })
                        break

        return breakdown

    def explain(self, text: str, level: str = 'beginner') -> Dict[str, Any]:
        """
        Generate comprehensive linguistic explanation

        Args:
            text: Finnish sentence to explain
            level: Target difficulty level (beginner/intermediate/advanced)

        Returns:
            Dictionary with comprehensive explanation
        """
        logger.info(f"Generating explanation for: {text[:50]}... (level: {level})")

        # Lemmatize text
        lemma_result = self.lemmatizer.lemmatize(text, include_morphology=True)

        # Analyze complexity
        complexity_result = self.complexity_analyzer.analyze(text, detailed=True)

        # Generate simplified version
        simplified = self._simplify_text(text)

        # Word-by-word explanations
        word_explanations = []

        for word_lemma in lemma_result.lemmas:
            word = word_lemma.original
            lemma = word_lemma.lemma
            pos = word_lemma.pos or 'NOUN'
            morphology = word_lemma.morphology or {}

            # Breakdown
            breakdown = self._breakdown_word(word, lemma, morphology, pos)

            # Difficulty
            difficulty = self._determine_word_difficulty(word, lemma, morphology)

            # Frequency
            frequency = self._get_word_frequency(lemma)

            # Learning tip
            learning_tip = self._generate_learning_tip(word, lemma, morphology, pos)

            # Build meaning
            meaning_parts = [lemma]
            if morphology.get('case') and morphology['case'] != 'Nominative':
                meaning_parts.append(f"({morphology['case'].lower()} case)")
            if word.endswith(('ni', 'si', 'nsa', 'mme')):
                meaning_parts.append("+ possessive")

            word_explanations.append({
                'word': word,
                'lemma': lemma,
                'pos': pos,
                'meaning': ' '.join(meaning_parts),
                'breakdown': breakdown,
                'frequency': frequency,
                'difficulty': difficulty,
                'learning_tip': learning_tip,
                'morphology': morphology
            })

        # Syntax analysis
        words = re.findall(r'\b[\w]+\b', text)
        syntax_analysis = {
            'sentence_type': 'simple' if complexity_result.clause_count <= 1 else 'complex',
            'clause_count': complexity_result.clause_count,
            'word_count': complexity_result.word_count,
            'complexity_rating': complexity_result.complexity_rating,
            'main_verb': None,  # Would need more sophisticated parsing
            'subject': words[0] if words else None,  # Simple heuristic
        }

        # Learning focus points
        learning_focus = []
        for exp in word_explanations:
            if exp['difficulty'] in [level, 'advanced']:
                learning_focus.append(f"{exp['word']}: {exp['learning_tip']}")

        result = {
            'text': text,
            'simplified': simplified,
            'level': level,
            'word_explanations': word_explanations,
            'syntax_analysis': syntax_analysis,
            'overall_difficulty': complexity_result.complexity_rating,
            'learning_focus': learning_focus[:3],  # Top 3 focus points
            'statistics': {
                'total_words': complexity_result.word_count,
                'unique_cases': len(set(w['morphology'].get('case', 'Nominative') for w in word_explanations if w['morphology'])),
                'average_word_length': complexity_result.average_word_length
            }
        }

        logger.info(f"Explanation generated: {len(word_explanations)} words explained")

        return result

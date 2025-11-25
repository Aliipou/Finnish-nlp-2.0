"""
Morphological Entropy Engine
Calculate information-theoretic complexity of Finnish morphology
Novel capability - unique to this toolkit
"""
import re
import logging
from typing import Dict, List
from collections import Counter
import math
from app.models.schemas import ComplexityResponse

logger = logging.getLogger(__name__)


class MorphologicalEntropyEngine:
    """
    Calculate morphological entropy metrics for Finnish text
    Measures information-theoretic complexity based on:
    - Case distribution entropy
    - Suffix variety entropy
    - Word formation complexity
    - Stem variation patterns
    """

    def __init__(self):
        logger.info("Initializing Morphological Entropy Engine")

        # Finnish grammatical cases
        self.cases = [
            'nominative', 'genitive', 'partitive', 'inessive', 'elative',
            'illative', 'adessive', 'ablative', 'allative', 'essive',
            'translative', 'instructive', 'abessive', 'comitative', 'prolative'
        ]

        # Case suffix patterns
        self.case_patterns = {
            'nominative': [r'\b\w+\b(?![a-zäö])'],
            'genitive': [r'\b\w+n\b'],
            'partitive': [r'\b\w+(a|ä|ta|tä)\b'],
            'inessive': [r'\b\w+(ssa|ssä)\b'],
            'elative': [r'\b\w+(sta|stä)\b'],
            'illative': [r'\b\w+(seen|hin|hon|hön|an|än)\b'],
            'adessive': [r'\b\w+(lla|llä)\b'],
            'ablative': [r'\b\w+(lta|ltä)\b'],
            'allative': [r'\b\w+lle\b'],
            'essive': [r'\b\w+(na|nä)\b'],
            'translative': [r'\b\w+ksi\b'],
        }

        # Compound word patterns
        self.compound_patterns = [
            r'\b\w{5,}[aä]inen\b',  # -ainen compounds
            r'\b\w{6,}(ton|tön)\b',  # -ton/-tön compounds
            r'\b\w{8,}\b',  # Long words (likely compounds)
        ]

        logger.info("Entropy engine initialized")

    def _calculate_shannon_entropy(self, distribution: Dict[str, int]) -> float:
        """
        Calculate Shannon entropy: H(X) = -Σ P(x) log₂ P(x)

        Args:
            distribution: Dictionary of item -> count

        Returns:
            Shannon entropy value
        """
        if not distribution or sum(distribution.values()) == 0:
            return 0.0

        total = sum(distribution.values())
        entropy = 0.0

        for count in distribution.values():
            if count > 0:
                probability = count / total
                entropy -= probability * math.log2(probability)

        return entropy

    def _detect_cases(self, text: str) -> Dict[str, int]:
        """
        Detect grammatical cases in text

        Returns:
            Dictionary of case -> count
        """
        case_counts = {}
        text_lower = text.lower()

        for case, patterns in self.case_patterns.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                count += len(matches)
            case_counts[case] = count

        return case_counts

    def _extract_suffixes(self, text: str) -> List[str]:
        """
        Extract all inflectional suffixes from words

        Returns:
            List of suffixes
        """
        words = re.findall(r'\b[\w]+\b', text.lower())
        suffixes = []

        suffix_patterns = [
            r'(ssa|ssä|sta|stä|lla|llä|lta|ltä|lle|na|nä|ksi)$',  # Location/essive/translative
            r'(seen|hin|hon|hön|an|än)$',  # Illative
            r'(tta|ttä|ta|tä|a|ä)$',  # Partitive
            r'(ni|si|nsa|nsä|mme|tte)$',  # Possessive
            r'(ko|kö)$',  # Question particle
        ]

        for word in words:
            if len(word) > 3:
                for pattern in suffix_patterns:
                    match = re.search(pattern, word)
                    if match:
                        suffixes.append(match.group(1))

        return suffixes

    def _detect_compounds(self, text: str) -> int:
        """
        Detect compound words

        Returns:
            Count of compound words
        """
        compound_count = 0
        text_lower = text.lower()

        for pattern in self.compound_patterns:
            matches = re.findall(pattern, text_lower)
            compound_count += len(matches)

        return compound_count

    def _calculate_case_entropy(self, case_distribution: Dict[str, int]) -> float:
        """Calculate entropy of case distribution"""
        return self._calculate_shannon_entropy(case_distribution)

    def _calculate_suffix_entropy(self, text: str) -> float:
        """Calculate entropy of suffix distribution"""
        suffixes = self._extract_suffixes(text)
        suffix_counts = Counter(suffixes)
        return self._calculate_shannon_entropy(dict(suffix_counts))

    def _calculate_word_formation_entropy(self, text: str) -> float:
        """
        Calculate entropy based on word formation complexity
        Considers word length distribution and compound words
        """
        words = re.findall(r'\b[\w]+\b', text.lower())

        # Word length categories
        length_categories = {
            'short': 0,      # 1-4 chars
            'medium': 0,     # 5-8 chars
            'long': 0,       # 9-12 chars
            'very_long': 0   # 13+ chars (likely compounds)
        }

        for word in words:
            length = len(word)
            if length <= 4:
                length_categories['short'] += 1
            elif length <= 8:
                length_categories['medium'] += 1
            elif length <= 12:
                length_categories['long'] += 1
            else:
                length_categories['very_long'] += 1

        return self._calculate_shannon_entropy(length_categories)

    def _calculate_overall_entropy_score(
        self,
        case_entropy: float,
        suffix_entropy: float,
        word_formation_entropy: float
    ) -> float:
        """
        Calculate overall entropy score (0-100)

        Weighted combination:
        - Case entropy: 40%
        - Suffix entropy: 30%
        - Word formation entropy: 30%
        """
        # Normalize entropies (max case entropy ≈ 3.9 for 15 cases)
        case_norm = min(case_entropy / 3.9, 1.0)
        suffix_norm = min(suffix_entropy / 4.0, 1.0)
        word_formation_norm = min(word_formation_entropy / 2.0, 1.0)

        # Weighted average
        overall = (case_norm * 0.4 + suffix_norm * 0.3 + word_formation_norm * 0.3) * 100

        return round(overall, 2)

    def _determine_interpretation(self, entropy_score: float) -> str:
        """Interpret entropy score"""
        if entropy_score >= 75:
            return "Very High - Extremely complex morphology with diverse case usage"
        elif entropy_score >= 60:
            return "High - Complex morphology with varied inflections"
        elif entropy_score >= 40:
            return "Moderate - Moderate morphological complexity"
        elif entropy_score >= 25:
            return "Low - Simple morphology with limited case variety"
        else:
            return "Very Low - Minimal morphological complexity"

    def calculate_entropy(self, text: str, detailed: bool = True) -> Dict:
        """
        Calculate morphological entropy metrics

        Args:
            text: Finnish text to analyze
            detailed: Include detailed breakdown

        Returns:
            Dictionary with entropy metrics
        """
        logger.info(f"Calculating entropy for text: {text[:50]}...")

        # Detect cases
        case_distribution = self._detect_cases(text)

        # Calculate individual entropies
        case_entropy = self._calculate_case_entropy(case_distribution)
        suffix_entropy = self._calculate_suffix_entropy(text)
        word_formation_entropy = self._calculate_word_formation_entropy(text)

        # Calculate overall score
        overall_score = self._calculate_overall_entropy_score(
            case_entropy,
            suffix_entropy,
            word_formation_entropy
        )

        # Interpretation
        interpretation = self._determine_interpretation(overall_score)

        result = {
            'text': text,
            'overall_entropy_score': overall_score,
            'case_entropy': round(case_entropy, 3),
            'suffix_entropy': round(suffix_entropy, 3),
            'word_formation_entropy': round(word_formation_entropy, 3),
            'interpretation': interpretation
        }

        if detailed:
            # Add detailed breakdown
            unique_suffixes = len(set(self._extract_suffixes(text)))
            compound_count = self._detect_compounds(text)
            cases_used = sum(1 for count in case_distribution.values() if count > 0)

            # Calculate percentile (rough estimate based on score)
            percentile = min(int(overall_score * 0.85), 99)

            result['detailed_breakdown'] = {
                'case_distribution': case_distribution,
                'cases_used': cases_used,
                'unique_suffixes': unique_suffixes,
                'compound_words': compound_count,
                'entropy_percentile': percentile,
                'complexity_factors': {
                    'case_diversity': f"{cases_used}/11 cases used",
                    'suffix_variety': f"{unique_suffixes} unique suffixes",
                    'compound_complexity': f"{compound_count} compound words"
                }
            }

        logger.info(f"Entropy analysis complete: score={overall_score}, interpretation={interpretation}")

        return result

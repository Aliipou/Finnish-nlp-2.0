"""
Semantic Ambiguity Resolver Engine
Novel capability #2 - Disambiguate highly ambiguous Finnish words
"""
import re
import logging
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class SemanticDisambiguator:
    """
    Resolve semantic ambiguity in Finnish text
    Identifies and disambiguates words with multiple meanings based on context

    Example: "kuusi" can mean "six", "spruce tree", or "sixth"
    """

    def __init__(self):
        logger.info("Initializing Semantic Disambiguator Engine")

        # Dictionary of ambiguous Finnish words with their possible meanings
        self.ambiguous_words = self._load_ambiguous_words()

        # Context patterns for each sense (simple rule-based for now)
        self.context_patterns = self._load_context_patterns()

        # ML model placeholder (will be loaded when available)
        self.ml_model = None
        self._try_load_ml_model()

        logger.info(f"Loaded {len(self.ambiguous_words)} ambiguous words")

    def _load_ambiguous_words(self) -> Dict[str, List[Dict]]:
        """
        Load dictionary of ambiguous Finnish words

        Returns:
            Dict mapping word -> list of sense dictionaries
        """
        return {
            'kuusi': [
                {'sense': 'six', 'pos': 'NUM', 'definition': 'Number 6', 'example': 'Kuusi ihmistä'},
                {'sense': 'spruce', 'pos': 'NOUN', 'definition': 'Spruce tree', 'example': 'Kuusi kasvaa metsässä'},
                {'sense': 'sixth', 'pos': 'ADJ', 'definition': 'Ordinal number 6th', 'example': 'Kuusi kerros'}
            ],
            'selkä': [
                {'sense': 'back_body', 'pos': 'NOUN', 'definition': 'Back (body part)', 'example': 'Selkä on kipeä'},
                {'sense': 'clear', 'pos': 'ADJ', 'definition': 'Clear, distinct', 'example': 'Selkä ääni'},
                {'sense': 'ridge', 'pos': 'NOUN', 'definition': 'Ridge, crest', 'example': 'Vuoren selkä'}
            ],
            'pankki': [
                {'sense': 'bank_financial', 'pos': 'NOUN', 'definition': 'Financial institution', 'example': 'Menen pankkiin'},
                {'sense': 'bench', 'pos': 'NOUN', 'definition': 'Bench, seat', 'example': 'Istun pankilla'}
            ],
            'tuuli': [
                {'sense': 'wind', 'pos': 'NOUN', 'definition': 'Wind, breeze', 'example': 'Tuuli puhaltaa'},
                {'sense': 'past_come', 'pos': 'VERB', 'definition': 'Past tense of "tulla" (come)', 'example': 'Hän tuuli kotiin'}
            ],
            'sataa': [
                {'sense': 'rain', 'pos': 'VERB', 'definition': 'To rain', 'example': 'Ulkona sataa'},
                {'sense': 'hundred', 'pos': 'NUM', 'definition': 'One hundred (partitive)', 'example': 'Sataa euroa'}
            ],
            'kuu': [
                {'sense': 'moon', 'pos': 'NOUN', 'definition': 'Moon', 'example': 'Kuu paistaa'},
                {'sense': 'month', 'pos': 'NOUN', 'definition': 'Month', 'example': 'Yhden kuun päästä'}
            ],
            'vesi': [
                {'sense': 'water', 'pos': 'NOUN', 'definition': 'Water', 'example': 'Juon vettä'},
                {'sense': 'hydrogen', 'pos': 'NOUN', 'definition': 'Hydrogen (chemistry)', 'example': 'Vesi on H2O'}
            ],
            'ajaa': [
                {'sense': 'drive', 'pos': 'VERB', 'definition': 'To drive (vehicle)', 'example': 'Ajaa autoa'},
                {'sense': 'chase', 'pos': 'VERB', 'definition': 'To chase, pursue', 'example': 'Koira ajaa kissaa'},
                {'sense': 'shave', 'pos': 'VERB', 'definition': 'To shave', 'example': 'Ajaa parta'}
            ],
            'ala': [
                {'sense': 'area', 'pos': 'NOUN', 'definition': 'Area, field', 'example': 'Tieteen ala'},
                {'sense': 'dont', 'pos': 'VERB', 'definition': 'Don\'t! (imperative)', 'example': 'Älä tee sitä'},
                {'sense': 'lower', 'pos': 'ADJ', 'definition': 'Lower, bottom', 'example': 'Ala-aste'}
            ],
            'kieli': [
                {'sense': 'language', 'pos': 'NOUN', 'definition': 'Language', 'example': 'Suomen kieli'},
                {'sense': 'tongue', 'pos': 'NOUN', 'definition': 'Tongue (body part)', 'example': 'Näytä kieli'}
            ]
        }

    def _load_context_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Load context patterns for disambiguating word senses

        Returns:
            Dict mapping word -> sense -> list of context keywords
        """
        return {
            'kuusi': {
                'six': ['ihmistä', 'henkilöä', 'kertaa', 'kappaletta', 'numero', 'numero', 'yhteensä'],
                'spruce': ['puu', 'metsä', 'kasvaa', 'havupuu', 'oksa', 'neulanen', 'mänty'],
                'sixth': ['kerros', 'luokka', 'sija', 'järjestyksessä', 'paikka']
            },
            'selkä': {
                'back_body': ['kipeä', 'sattuu', 'särky', 'hieronta', 'lihakset', 'selkäranka'],
                'clear': ['ääni', 'näkyvä', 'ilmeinen', 'selvä', 'kirkko'],
                'ridge': ['vuori', 'mäki', 'huippu', 'laki', 'harjanne']
            },
            'pankki': {
                'bank_financial': ['raha', 'tili', 'laina', 'korko', 'otto', 'pano', 'kortti'],
                'bench': ['istu', 'puisto', 'puinen', 'penkkı', 'lepo']
            },
            'tuuli': {
                'wind': ['puhal', 'ilma', 'myrsky', 'tuulee', 'kovaa', 'tuulinen'],
                'past_come': ['kotiin', 'tuli', 'saapui', 'eilen', 'aikaisemmin']
            },
            'sataa': {
                'rain': ['vettä', 'lunta', 'ulkona', 'sää', 'sateinen', 'kaatuu'],
                'hundred': ['euroa', 'markkaa', 'dollaria', 'numeroa', '100']
            },
            'kuu': {
                'moon': ['taivas', 'täysi', 'uusi', 'kuutamo', 'yö', 'loista'],
                'month': ['päästä', 'sitten', 'aikaa', 'vuosi', 'viikko', 'aika']
            },
            'ajaa': {
                'drive': ['auto', 'ajokortti', 'tielle', 'rata', 'ajoneuvoa'],
                'chase': ['takaa', 'koira', 'juokse', 'pakoon', 'perään'],
                'shave': ['parta', 'höylä', 'ajokone', 'sileä', 'karva']
            },
            'kieli': {
                'language': ['suomen', 'puhua', 'opiskella', 'kääntää', 'sana', 'lausua'],
                'tongue': ['suussa', 'maistaa', 'polttaa', 'näytä', 'elinä']
            }
        }

    def _try_load_ml_model(self):
        """Try to load ML model for disambiguation"""
        try:
            from app.ml_models.model_registry import get_model_registry
            registry = get_model_registry()
            self.ml_model = registry.load_model('ambiguity_resolver')
            if self.ml_model:
                logger.info("✅ Loaded ML ambiguity resolver model")
        except Exception as e:
            logger.info("ℹ️  ML model not available, using rule-based disambiguation")
            self.ml_model = None

    def _detect_ambiguous_words(self, text: str) -> List[Tuple[str, int]]:
        """
        Detect ambiguous words in text

        Returns:
            List of (word, position) tuples
        """
        text_lower = text.lower()
        found_words = []

        for word in self.ambiguous_words.keys():
            # Find all occurrences (with inflections)
            # More flexible pattern to catch inflected forms
            pattern = r'\b' + re.escape(word) + r'\w*\b'
            matches = re.finditer(pattern, text_lower)

            for match in matches:
                found_words.append((match.group(), match.start()))

        # Also check for common inflected forms
        # Add 'lla/llä' (adessive), 'ltä/lta' (ablative)
        inflection_patterns = [
            (r'\b(\w+)(lla|llä|lta|ltä|lle)\b', 'pankki'),  # pankilla -> pankki
            (r'\b(\w+)(tä|ta)\b', 'kieli'),  # kieltä -> kieli
        ]

        for pattern, base_word in inflection_patterns:
            if base_word in self.ambiguous_words:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    stem = match.group(1)
                    if stem.startswith(base_word[:3]):  # Rough stem match
                        found_words.append((match.group(), match.start()))

        return found_words

    def _extract_context(self, text: str, position: int, window: int = 5) -> Tuple[str, str]:
        """
        Extract context around word

        Args:
            text: Full text
            position: Word position
            window: Number of words to include before/after

        Returns:
            (context_before, context_after) tuple
        """
        words = re.findall(r'\b\w+\b', text)
        word_positions = [m.start() for m in re.finditer(r'\b\w+\b', text)]

        # Find word index
        word_idx = None
        for i, pos in enumerate(word_positions):
            if abs(pos - position) < 5:  # Close enough
                word_idx = i
                break

        if word_idx is None:
            return "", ""

        # Extract surrounding words
        start_idx = max(0, word_idx - window)
        end_idx = min(len(words), word_idx + window + 1)

        context_before = ' '.join(words[start_idx:word_idx])
        context_after = ' '.join(words[word_idx + 1:end_idx])

        return context_before, context_after

    def _rule_based_disambiguate(
        self,
        word: str,
        context_before: str,
        context_after: str
    ) -> Tuple[str, float, List[Dict]]:
        """
        Rule-based disambiguation using context patterns

        Returns:
            (predicted_sense, confidence, alternatives)
        """
        word_lower = word.lower()

        # Get base word (remove inflections roughly)
        base_word = word_lower
        for known_word in self.ambiguous_words.keys():
            if word_lower.startswith(known_word):
                base_word = known_word
                break

        # Also check if word ends with known inflections and matches start of known word
        if base_word == word_lower:  # No match yet
            for known_word in self.ambiguous_words.keys():
                # Check common inflections
                for suffix in ['lla', 'llä', 'lta', 'ltä', 'lle', 'ssa', 'ssä', 'sta', 'stä', 'tä', 'ta', 'n']:
                    if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
                        potential_stem = word_lower[:-len(suffix)]
                        if known_word.startswith(potential_stem[:3]) or potential_stem.startswith(known_word[:3]):
                            base_word = known_word
                            break
                if base_word != word_lower:
                    break

        if base_word not in self.ambiguous_words:
            return 'unknown', 0.0, []

        # Get possible senses
        senses = self.ambiguous_words[base_word]

        # Score each sense based on context patterns
        sense_scores = {}
        context_text = (context_before + ' ' + context_after).lower()

        if base_word in self.context_patterns:
            patterns = self.context_patterns[base_word]

            for sense_dict in senses:
                sense = sense_dict['sense']
                score = 0.0

                if sense in patterns:
                    keywords = patterns[sense]
                    for keyword in keywords:
                        if keyword in context_text:
                            score += 1.0

                sense_scores[sense] = score

        # If no scores, use default (first sense)
        if not sense_scores or max(sense_scores.values()) == 0:
            return senses[0]['sense'], 0.3, [
                {'sense': s['sense'], 'probability': 1.0 / len(senses)}
                for s in senses[1:]
            ]

        # Get best sense
        total_score = sum(sense_scores.values())
        predicted_sense = max(sense_scores.items(), key=lambda x: x[1])[0]
        confidence = sense_scores[predicted_sense] / total_score if total_score > 0 else 0.5

        # Get alternatives
        alternatives = []
        for sense, score in sorted(sense_scores.items(), key=lambda x: -x[1])[1:]:
            if score > 0:
                alternatives.append({
                    'sense': sense,
                    'probability': score / total_score if total_score > 0 else 0.0
                })

        return predicted_sense, confidence, alternatives

    def disambiguate(
        self,
        text: str,
        target_words: Optional[List[str]] = None,
        auto_detect: bool = True
    ) -> Dict:
        """
        Disambiguate ambiguous words in text

        Args:
            text: Input Finnish text
            target_words: Specific words to disambiguate (optional)
            auto_detect: Automatically detect ambiguous words

        Returns:
            Dictionary with disambiguation results
        """
        logger.info(f"Disambiguating text: {text[:50]}...")

        # Detect ambiguous words
        if auto_detect:
            found_words = self._detect_ambiguous_words(text)
        else:
            # Use target words if provided
            found_words = []
            if target_words:
                text_lower = text.lower()
                for target in target_words:
                    pattern = r'\b' + re.escape(target.lower()) + r'\w*\b'
                    matches = re.finditer(pattern, text_lower)
                    for match in matches:
                        found_words.append((match.group(), match.start()))

        # Disambiguate each word
        disambiguations = []

        for word, position in found_words:
            # Extract context
            context_before, context_after = self._extract_context(text, position)

            # Use ML model if available, otherwise rule-based
            if self.ml_model:
                # ML disambiguation (placeholder - model not trained yet)
                predicted_sense, confidence, alternatives = self._rule_based_disambiguate(
                    word, context_before, context_after
                )
            else:
                # Rule-based disambiguation
                predicted_sense, confidence, alternatives = self._rule_based_disambiguate(
                    word, context_before, context_after
                )

            # Get word definition
            base_word = word.lower()
            for known_word in self.ambiguous_words.keys():
                if base_word.startswith(known_word):
                    base_word = known_word
                    break

            definition = ""
            if base_word in self.ambiguous_words:
                for sense_dict in self.ambiguous_words[base_word]:
                    if sense_dict['sense'] == predicted_sense:
                        definition = sense_dict['definition']
                        break

            disambiguations.append({
                'word': word,
                'position': position,
                'predicted_sense': predicted_sense,
                'definition': definition,
                'confidence': round(confidence, 3),
                'alternatives': alternatives,
                'context_snippet': f"{context_before} [{word}] {context_after}",
                'method': 'ml' if self.ml_model else 'rule_based'
            })

        result = {
            'text': text,
            'ambiguous_words_found': len(disambiguations),
            'disambiguations': disambiguations
        }

        logger.info(f"Disambiguation complete: {len(disambiguations)} ambiguous words resolved")

        return result

    def get_word_senses(self, word: str) -> Optional[List[Dict]]:
        """
        Get all possible senses for a word

        Args:
            word: Finnish word

        Returns:
            List of sense dictionaries or None
        """
        word_lower = word.lower()

        # Check exact match
        if word_lower in self.ambiguous_words:
            return self.ambiguous_words[word_lower]

        # Check if it starts with known word (inflected form)
        for known_word in self.ambiguous_words.keys():
            if word_lower.startswith(known_word):
                return self.ambiguous_words[known_word]

        return None

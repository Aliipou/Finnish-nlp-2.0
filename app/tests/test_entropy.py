"""
Tests for Morphological Entropy Engine
Novel capability testing
"""
import pytest
from app.services.entropy_engine import MorphologicalEntropyEngine


@pytest.fixture
def entropy_engine():
    """Fixture for entropy engine"""
    return MorphologicalEntropyEngine()


def test_initialization(entropy_engine):
    """Test engine initializes correctly"""
    assert entropy_engine is not None
    assert len(entropy_engine.cases) == 15  # 15 Finnish cases
    assert 'nominative' in entropy_engine.cases
    assert 'genitive' in entropy_engine.cases


def test_simple_text_entropy(entropy_engine):
    """Test entropy calculation for simple text"""
    text = "Kissa istui."
    result = entropy_engine.calculate_entropy(text, detailed=False)

    assert 'overall_entropy_score' in result
    assert 'case_entropy' in result
    assert 'suffix_entropy' in result
    assert 'word_formation_entropy' in result
    assert 'interpretation' in result
    assert result['overall_entropy_score'] >= 0
    assert result['overall_entropy_score'] <= 100


def test_complex_text_entropy(entropy_engine):
    """Test entropy for morphologically complex text"""
    text = "Kissani söi hiiren puutarhassani nopeasti talvella."
    result = entropy_engine.calculate_entropy(text, detailed=True)

    assert result['overall_entropy_score'] > 0
    assert 'detailed_breakdown' in result
    assert 'case_distribution' in result['detailed_breakdown']
    assert 'cases_used' in result['detailed_breakdown']
    assert 'unique_suffixes' in result['detailed_breakdown']


def test_entropy_with_multiple_cases(entropy_engine):
    """Test text using multiple grammatical cases"""
    # Text with nominative, genitive, inessive, elative
    text = "Kissa juoksi talosta taloon puutarhassa."
    result = entropy_engine.calculate_entropy(text, detailed=True)

    cases_used = result['detailed_breakdown']['cases_used']
    assert cases_used >= 2  # Should detect multiple cases


def test_empty_text(entropy_engine):
    """Test entropy with minimal text"""
    text = "a"
    result = entropy_engine.calculate_entropy(text, detailed=False)

    # Should return low entropy for single character
    assert result['overall_entropy_score'] < 20


def test_shannon_entropy_calculation(entropy_engine):
    """Test Shannon entropy calculation"""
    # Uniform distribution should give high entropy
    uniform_dist = {'a': 10, 'b': 10, 'c': 10, 'd': 10}
    entropy = entropy_engine._calculate_shannon_entropy(uniform_dist)
    assert entropy > 1.9  # log2(4) = 2, should be close

    # Skewed distribution should give low entropy
    skewed_dist = {'a': 90, 'b': 5, 'c': 3, 'd': 2}
    entropy_skewed = entropy_engine._calculate_shannon_entropy(skewed_dist)
    assert entropy_skewed < entropy  # Less entropy than uniform


def test_case_detection(entropy_engine):
    """Test grammatical case detection"""
    text = "Kissassa, kissasta, kissalle"  # Inessive, elative, allative
    case_dist = entropy_engine._detect_cases(text)

    assert case_dist['inessive'] > 0
    assert case_dist['elative'] > 0
    assert case_dist['allative'] > 0


def test_suffix_extraction(entropy_engine):
    """Test suffix extraction"""
    text = "Kissani, koirani, taloni"  # Possessive suffix -ni
    suffixes = entropy_engine._extract_suffixes(text)

    assert len(suffixes) > 0
    assert 'ni' in suffixes


def test_compound_word_detection(entropy_engine):
    """Test compound word detection"""
    text = "Kirjoittautumisvelvollisuuden laiminlyönti"
    compound_count = entropy_engine._detect_compounds(text)

    assert compound_count > 0  # Should detect long compound words


def test_interpretation_levels(entropy_engine):
    """Test interpretation at different score levels"""
    # Very simple text (low entropy)
    simple_text = "Auto on."
    simple_result = entropy_engine.calculate_entropy(simple_text)

    # Complex text (higher entropy)
    complex_text = "Kirjoittautumisvelvollisuuden laiminlyönnistä aiheutuvat sanktiot ovat merkittäviä oikeudellisia seuraamuksia."
    complex_result = entropy_engine.calculate_entropy(complex_text)

    # Complex should have higher score
    assert complex_result['overall_entropy_score'] > simple_result['overall_entropy_score']


def test_detailed_breakdown_fields(entropy_engine):
    """Test that detailed breakdown contains all expected fields"""
    text = "Kissani söi hiiren puutarhassani nopeasti."
    result = entropy_engine.calculate_entropy(text, detailed=True)

    breakdown = result['detailed_breakdown']
    assert 'case_distribution' in breakdown
    assert 'cases_used' in breakdown
    assert 'unique_suffixes' in breakdown
    assert 'compound_words' in breakdown
    assert 'entropy_percentile' in breakdown
    assert 'complexity_factors' in breakdown


def test_entropy_score_range(entropy_engine):
    """Test that entropy scores are always in valid range"""
    test_texts = [
        "A",
        "Kissa istui.",
        "Kissani söi hiiren puutarhassani nopeasti.",
        "Kirjoittautumisvelvollisuuden laiminlyönti on vakava rikkomus."
    ]

    for text in test_texts:
        result = entropy_engine.calculate_entropy(text)
        score = result['overall_entropy_score']

        assert score >= 0, f"Score below 0 for text: {text}"
        assert score <= 100, f"Score above 100 for text: {text}"


def test_case_entropy_properties(entropy_engine):
    """Test case entropy calculation properties"""
    # Text with no case variation (all nominative)
    uniform_text = "Kissa koira auto talo"
    uniform_result = entropy_engine.calculate_entropy(uniform_text)

    # Text with high case variation
    varied_text = "Kissassa kissasta kissalle kissana kissaksi"
    varied_result = entropy_engine.calculate_entropy(varied_text)

    # Varied should have higher case entropy
    assert varied_result['case_entropy'] >= uniform_result['case_entropy']

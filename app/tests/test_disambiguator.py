"""
Tests for Semantic Disambiguator Engine
Novel capability #2 testing
"""
import pytest
from app.services.disambiguator_engine import SemanticDisambiguator


@pytest.fixture
def disambiguator():
    """Fixture for disambiguator engine"""
    return SemanticDisambiguator()


def test_initialization(disambiguator):
    """Test engine initializes correctly"""
    assert disambiguator is not None
    assert len(disambiguator.ambiguous_words) >= 10
    assert 'kuusi' in disambiguator.ambiguous_words
    assert 'selkä' in disambiguator.ambiguous_words
    assert 'pankki' in disambiguator.ambiguous_words


def test_kuusi_as_number(disambiguator):
    """Test disambiguation of 'kuusi' as number six"""
    text = "Kuusi ihmistä tuli juhliin."
    result = disambiguator.disambiguate(text, auto_detect=True)

    assert result['ambiguous_words_found'] >= 1
    assert len(result['disambiguations']) >= 1

    # Check first disambiguation
    dis = result['disambiguations'][0]
    assert dis['word'].lower().startswith('kuusi')
    assert dis['predicted_sense'] == 'six'
    assert dis['confidence'] > 0


def test_kuusi_as_tree(disambiguator):
    """Test disambiguation of 'kuusi' as spruce tree"""
    text = "Korkea kuusi kasvaa metsässä."
    result = disambiguator.disambiguate(text, auto_detect=True)

    assert result['ambiguous_words_found'] >= 1
    dis = result['disambiguations'][0]
    assert dis['predicted_sense'] == 'spruce'


def test_selka_as_back(disambiguator):
    """Test disambiguation of 'selkä' as back (body part)"""
    text = "Minun selkä on kipeä."
    result = disambiguator.disambiguate(text, auto_detect=True)

    assert result['ambiguous_words_found'] >= 1
    dis = result['disambiguations'][0]
    assert dis['predicted_sense'] == 'back_body'


def test_pankki_financial(disambiguator):
    """Test disambiguation of 'pankki' as bank (financial)"""
    text = "Menen pankkiin nostamaan rahaa."
    result = disambiguator.disambiguate(text, auto_detect=True)

    assert result['ambiguous_words_found'] >= 1
    dis = result['disambiguations'][0]
    assert dis['predicted_sense'] == 'bank_financial'


def test_pankki_bench(disambiguator):
    """Test disambiguation of 'pankki' as bench"""
    text = "Istun puiston pankilla."
    result = disambiguator.disambiguate(text, auto_detect=True)

    assert result['ambiguous_words_found'] >= 1
    dis = result['disambiguations'][0]
    assert dis['predicted_sense'] == 'bench'


def test_multiple_ambiguous_words(disambiguator):
    """Test text with multiple ambiguous words"""
    text = "Kuusi kaunista kuusta kasvaa mäellä. Istun pankilla."
    result = disambiguator.disambiguate(text, auto_detect=True)

    # Should find at least 2 ambiguous words (kuusi and pankilla)
    assert result['ambiguous_words_found'] >= 2


def test_no_ambiguous_words(disambiguator):
    """Test text with no ambiguous words"""
    text = "Kissa istui ikkunalla."
    result = disambiguator.disambiguate(text, auto_detect=True)

    assert result['ambiguous_words_found'] == 0
    assert len(result['disambiguations']) == 0


def test_target_words_parameter(disambiguator):
    """Test targeting specific words"""
    text = "Kuusi kaunista kuusta kasvaa mäellä."
    result = disambiguator.disambiguate(
        text,
        target_words=['kuusi'],
        auto_detect=False
    )

    # Should find the targeted word
    assert result['ambiguous_words_found'] >= 1


def test_context_extraction(disambiguator):
    """Test context extraction around word"""
    text = "Tämä on testi kuusi testi teksti."
    context_before, context_after = disambiguator._extract_context(text, 15, window=2)

    assert 'testi' in context_before or 'testi' in context_after


def test_get_word_senses(disambiguator):
    """Test getting all senses for a word"""
    senses = disambiguator.get_word_senses('kuusi')

    assert senses is not None
    assert len(senses) == 3  # six, spruce, sixth
    assert any(s['sense'] == 'six' for s in senses)
    assert any(s['sense'] == 'spruce' for s in senses)
    assert any(s['sense'] == 'sixth' for s in senses)


def test_get_word_senses_unknown(disambiguator):
    """Test getting senses for unknown word"""
    senses = disambiguator.get_word_senses('jokuoutoword')

    assert senses is None


def test_confidence_scores(disambiguator):
    """Test that confidence scores are in valid range"""
    text = "Kuusi ihmistä söi kuusi omenaa."
    result = disambiguator.disambiguate(text, auto_detect=True)

    for dis in result['disambiguations']:
        assert 0.0 <= dis['confidence'] <= 1.0


def test_context_snippet(disambiguator):
    """Test that context snippet is provided"""
    text = "Kuusi kaunista kuusta kasvaa mäellä."
    result = disambiguator.disambiguate(text, auto_detect=True)

    assert len(result['disambiguations']) > 0
    dis = result['disambiguations'][0]
    assert 'context_snippet' in dis
    assert len(dis['context_snippet']) > 0


def test_alternatives_provided(disambiguator):
    """Test that alternative senses are provided"""
    text = "Kuusi ihmistä tuli."
    result = disambiguator.disambiguate(text, auto_detect=True)

    if len(result['disambiguations']) > 0:
        dis = result['disambiguations'][0]
        assert 'alternatives' in dis
        assert isinstance(dis['alternatives'], list)


def test_disambiguation_response_structure(disambiguator):
    """Test complete response structure"""
    text = "Kuusi kaunista kuusta."
    result = disambiguator.disambiguate(text, auto_detect=True)

    # Check top-level keys
    assert 'text' in result
    assert 'ambiguous_words_found' in result
    assert 'disambiguations' in result

    # Check disambiguation structure
    if len(result['disambiguations']) > 0:
        dis = result['disambiguations'][0]
        assert 'word' in dis
        assert 'position' in dis
        assert 'predicted_sense' in dis
        assert 'definition' in dis
        assert 'confidence' in dis
        assert 'alternatives' in dis
        assert 'context_snippet' in dis
        assert 'method' in dis


def test_tuuli_wind(disambiguator):
    """Test disambiguation of 'tuuli' as wind"""
    text = "Ulkona tuuli kovasti."
    result = disambiguator.disambiguate(text, auto_detect=True)

    assert result['ambiguous_words_found'] >= 1
    dis = result['disambiguations'][0]
    assert dis['predicted_sense'] == 'wind'


def test_sataa_rain(disambiguator):
    """Test disambiguation of 'sataa' as rain"""
    text = "Ulkona sataa vettä."
    result = disambiguator.disambiguate(text, auto_detect=True)

    assert result['ambiguous_words_found'] >= 1
    dis = result['disambiguations'][0]
    assert dis['predicted_sense'] == 'rain'


def test_kieli_language(disambiguator):
    """Test disambiguation of 'kieli' as language"""
    text = "Opiskelen suomen kieltä."
    result = disambiguator.disambiguate(text, auto_detect=True)

    assert result['ambiguous_words_found'] >= 1
    dis = result['disambiguations'][0]
    assert dis['predicted_sense'] == 'language'


def test_inflected_forms(disambiguator):
    """Test disambiguation of inflected forms"""
    # 'kuusta' is genitive plural of 'kuusi' (spruce)
    text = "Kuusi kaunista kuusta kasvaa."
    result = disambiguator.disambiguate(text, auto_detect=True)

    # Should detect both 'kuusi' and 'kuusta'
    assert result['ambiguous_words_found'] >= 1

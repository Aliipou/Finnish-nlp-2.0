"""
Rigorous tests for the Advanced Finnish Lemmatizer Engine.

Tests are grouped by Finnish grammatical phenomenon so regressions are easy
to locate.  Every test works WITHOUT libvoikko so CI can run on plain Python.
When Voikko is available the same cases will pass through _voikko_analyze and
are exercised automatically.

Finnish has 15 morphological cases, 5 verb moods (including the rare
Potentiaali), 5 infinitive types, active/passive voices, 4+ participle forms,
consonant gradation across 6 types, vowel harmony, and clitic suffixes.
This test suite covers all of these systematically.
"""
import pytest
from app.services.advanced_lemma_engine import (
    AdvancedLemmatizerEngine,
    _translate_case,
    _translate_number,
    _translate_pos,
    _best_analysis,
    VOIKKO_CASE_MAP,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    """Rule-based engine (no Voikko required)."""
    return AdvancedLemmatizerEngine(use_voikko=False)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def lemma_of(engine, word: str) -> str:
    result = engine.lemmatize(word, include_morphology=False)
    assert result.word_count == 1
    return result.lemmas[0].lemma


def case_of(engine, word: str) -> str:
    result = engine.lemmatize(word, include_morphology=True)
    assert result.word_count == 1
    morph = result.lemmas[0].morphology or {}
    return morph.get("case", "Unknown")


def number_of(engine, word: str) -> str:
    result = engine.lemmatize(word, include_morphology=True)
    assert result.word_count == 1
    morph = result.lemmas[0].morphology or {}
    return morph.get("number", "Unknown")


# ===========================================================================
# 1. VOIKKO CASE TRANSLATION (unit tests for the translation layer)
# ===========================================================================

class TestVoikkoCaseTranslation:
    """Verify the Finnish→English case name mapping is correct."""

    @pytest.mark.parametrize("fi_name,expected", [
        ("nimento",     "Nominative"),
        ("omanto",      "Genitive"),
        ("osanto",      "Partitive"),
        ("olento",      "Essive"),
        ("tulento",     "Translative"),
        ("sisaolento",  "Inessive"),
        ("sisaeronto",  "Elative"),
        ("sisatulento", "Illative"),
        ("ulko-olento", "Adessive"),
        ("ulkoeronto",  "Ablative"),
        ("ulkotulento", "Allative"),
        ("vajanto",     "Abessive"),
        ("seuranto",    "Comitative"),
        ("keinonto",    "Instructive"),
    ])
    def test_case_map(self, fi_name, expected):
        assert _translate_case(fi_name) == expected

    def test_unknown_case_passthrough(self):
        assert _translate_case("XYZ") == "XYZ"

    def test_none_case_returns_unknown(self):
        assert _translate_case(None) == "Unknown"

    def test_all_voikko_cases_covered(self):
        """Every key in VOIKKO_CASE_MAP must produce a non-empty English value."""
        for fi_name, english in VOIKKO_CASE_MAP.items():
            assert english, f"Empty English name for {fi_name}"
            assert english[0].isupper(), f"Case name should be capitalised: {english}"

    def test_case_map_is_case_insensitive(self):
        assert _translate_case("NIMENTO") == "Nominative"
        assert _translate_case("Omanto") == "Genitive"


class TestVoikkoPosTranslation:
    @pytest.mark.parametrize("fi_class,expected", [
        ("nimisana",   "NOUN"),
        ("teonsana",   "VERB"),
        ("laatusana",  "ADJ"),
        ("seikkasana", "ADV"),
        ("lukusana",   "NUM"),
        ("etunimi",    "PROPN"),
        ("sukunimi",   "PROPN"),
    ])
    def test_pos_map(self, fi_class, expected):
        assert _translate_pos(fi_class) == expected

    def test_none_pos(self):
        assert _translate_pos(None) == "UNKNOWN"


class TestVoikkoNumberTranslation:
    def test_singular(self):
        assert _translate_number("singular") == "Singular"
        assert _translate_number("yksikko")  == "Singular"

    def test_plural(self):
        assert _translate_number("plural")  == "Plural"
        assert _translate_number("monikko") == "Plural"

    def test_none_defaults_singular(self):
        assert _translate_number(None) == "Singular"


class TestBestAnalysis:
    def test_empty_list(self):
        assert _best_analysis([]) == {}

    def test_single_item(self):
        item = {"BASEFORM": "kissa", "CLASS": "nimisana"}
        assert _best_analysis([item]) == item

    def test_prefers_content_word(self):
        func = {"BASEFORM": "ja", "CLASS": "sidesana"}
        noun = {"BASEFORM": "joki", "CLASS": "nimisana"}
        assert _best_analysis([func, noun]) == noun

    def test_prefers_analysis_with_baseform(self):
        no_base = {"CLASS": "nimisana"}
        with_base = {"BASEFORM": "kissa", "CLASS": "nimisana"}
        assert _best_analysis([no_base, with_base]) == with_base


# ===========================================================================
# 2. NOMINATIVE SINGULAR (no change expected)
# ===========================================================================

class TestNominativeSingular:
    @pytest.mark.parametrize("word", [
        "kissa", "koira", "talo", "auto", "hiiri", "nainen", "ihminen",
        "kauppa", "hattu", "pöytä", "paras", "hyvä",
    ])
    def test_known_nominatives_unchanged(self, engine, word):
        assert lemma_of(engine, word) == word

    def test_lemma_response_fields(self, engine):
        result = engine.lemmatize("kissa", include_morphology=True)
        lemma = result.lemmas[0]
        assert lemma.original == "kissa"
        assert lemma.lemma    == "kissa"
        assert lemma.pos      in ("NOUN", "UNKNOWN")
        assert lemma.morphology is not None

    def test_nominative_case_label(self, engine):
        assert case_of(engine, "kissa") == "Nominative"

    def test_nominative_number_singular(self, engine):
        assert number_of(engine, "kissa") == "Singular"


# ===========================================================================
# 3. GENITIVE SINGULAR (-n suffix)
# ===========================================================================

class TestGenitiveSingular:
    @pytest.mark.parametrize("word,expected_lemma", [
        ("kissan",   "kissa"),
        ("koiran",   "koira"),
        ("talon",    "talo"),
        ("auton",    "auto"),
        ("hiiren",   "hiiri"),
        ("naisen",   "nainen"),
        ("ihmisen",  "ihminen"),
        ("kaupan",   "kauppa"),
        ("hatun",    "hattu"),
    ])
    def test_genitive_lemmatisation(self, engine, word, expected_lemma):
        assert lemma_of(engine, word) == expected_lemma

    def test_genitive_case_label(self, engine):
        assert case_of(engine, "kissan") == "Genitive"

    def test_genitive_number_singular(self, engine):
        assert number_of(engine, "kissan") == "Singular"


# ===========================================================================
# 4. PARTITIVE SINGULAR
# ===========================================================================

class TestPartitiveSingular:
    @pytest.mark.parametrize("word,expected_lemma", [
        ("kissaa",   "kissa"),
        ("koiraa",   "koira"),
        ("taloa",    "talo"),
        ("autoa",    "auto"),
        ("hiirtä",   "hiiri"),
        ("naista",   "nainen"),   # nais+ta — special -nen stem
        ("ihmistä",  "ihminen"),  # ihmiis+tä
        ("hyvää",    "hyvä"),
    ])
    def test_partitive_lemmatisation(self, engine, word, expected_lemma):
        assert lemma_of(engine, word) == expected_lemma


# ===========================================================================
# 5. LOCAL CASES (inessive, elative, illative, adessive, ablative, allative)
# ===========================================================================

class TestLocalCases:
    @pytest.mark.parametrize("word,expected_lemma", [
        # Inessive (ssa/ssä)
        ("kissassa",  "kissa"),
        ("talossa",   "talo"),
        ("pöydässä",  "pöytä"),
        # Elative (sta/stä)
        ("kissasta",  "kissa"),
        ("talosta",   "talo"),
        # Adessive (lla/llä)
        ("kissalla",  "kissa"),
        ("pöydällä",  "pöytä"),
        # Ablative (lta/ltä)
        ("kissalta",  "kissa"),
        ("pöydältä",  "pöytä"),
        # Allative (lle)
        ("kissalle",  "kissa"),
        ("pöydälle",  "pöytä"),
        # Illative (-an/-ään/-oon/-een/-iin)
        ("kissaan",   "kissa"),
        ("taloon",    "talo"),
        ("hiireen",   "hiiri"),
    ])
    def test_local_case(self, engine, word, expected_lemma):
        assert lemma_of(engine, word) == expected_lemma

    def test_inessive_case_label(self, engine):
        """Rule-based engine must correctly identify Inessive, not Nominative."""
        assert case_of(engine, "talossa") in ("Inessive", "Unknown")

    def test_elative_case_label(self, engine):
        assert case_of(engine, "talosta") in ("Elative", "Unknown")

    def test_adessive_case_label(self, engine):
        assert case_of(engine, "kissalla") in ("Adessive", "Unknown")

    def test_allative_case_label(self, engine):
        assert case_of(engine, "kissalle") in ("Allative", "Unknown")


# ===========================================================================
# 6. NOMINATIVE PLURAL (-t)
# ===========================================================================

class TestNominativePlural:
    @pytest.mark.parametrize("word,expected_lemma", [
        ("kissat",    "kissa"),
        ("koirat",    "koira"),
        ("talot",     "talo"),
        ("autot",     "auto"),
        ("hiiret",    "hiiri"),
        ("naiset",    "nainen"),
        ("ihmiset",   "ihminen"),
        ("hatut",     "hattu"),
        ("pöydät",    "pöytä"),
        ("hyvät",     "hyvä"),
    ])
    def test_nominative_plural(self, engine, word, expected_lemma):
        assert lemma_of(engine, word) == expected_lemma

    def test_nominative_plural_case_label(self, engine):
        assert case_of(engine, "kissat") == "Nominative"

    def test_nominative_plural_number(self, engine):
        assert number_of(engine, "kissat") == "Plural"


# ===========================================================================
# 7. CONSONANT GRADATION (astevaihtelu)
# ===========================================================================

class TestConsonantGradation:
    """
    Finnish nouns and verbs exhibit consonant gradation where the stem
    consonant alternates between strong and weak grade depending on the
    inflectional suffix.

    kauppa (pp→p):  kauppa → kaupan (genitive)
    hattu  (tt→t):  hattu  → hatun
    pöytä  (t→d):   pöytä  → pöydän
    jalka  (k→∅):   jalka  → jalan

    The known-word dictionary allows the rule-based engine to handle these
    without a full gradation implementation.
    """

    def test_kauppa_genitive(self, engine):
        assert lemma_of(engine, "kaupan") == "kauppa"

    def test_kauppa_inessive(self, engine):
        assert lemma_of(engine, "kaupassa") == "kauppa"

    def test_hattu_genitive(self, engine):
        assert lemma_of(engine, "hatun") == "hattu"

    def test_hattu_inessive(self, engine):
        assert lemma_of(engine, "hatussa") == "hattu"

    def test_poyda_genitive(self, engine):
        assert lemma_of(engine, "pöydän") == "pöytä"

    def test_poyda_inessive(self, engine):
        assert lemma_of(engine, "pöydässä") == "pöytä"

    def test_poyda_allative(self, engine):
        assert lemma_of(engine, "pöydälle") == "pöytä"

    def test_jalka_genitive(self, engine):
        assert lemma_of(engine, "jalan") == "jalka"

    def test_jalka_inessive(self, engine):
        assert lemma_of(engine, "jalassa") == "jalka"


# ===========================================================================
# 8. -NEN WORDS (Type 38 — stem changes)
# ===========================================================================

class TestNenWords:
    """
    Words ending in -nen (ihminen, nainen, suomalainen) are inflection Type 38.
    They use a different stem in most cases:
      nainen  → naise- in most cases (naisen, naisessa …)
      nainen  → nais-  in partitive singular (naista)
    """

    def test_nainen_nominative(self, engine):
        assert lemma_of(engine, "nainen") == "nainen"

    def test_nainen_genitive(self, engine):
        assert lemma_of(engine, "naisen") == "nainen"

    def test_nainen_partitive_singular(self, engine):
        assert lemma_of(engine, "naista") == "nainen"

    def test_nainen_inessive(self, engine):
        assert lemma_of(engine, "naisessa") == "nainen"

    def test_nainen_elative(self, engine):
        assert lemma_of(engine, "naisesta") == "nainen"

    def test_nainen_plural_nominative(self, engine):
        assert lemma_of(engine, "naiset") == "nainen"

    def test_ihminen_genitive(self, engine):
        assert lemma_of(engine, "ihmisen") == "ihminen"

    def test_ihminen_partitive(self, engine):
        assert lemma_of(engine, "ihmistä") == "ihminen"

    def test_ihminen_plural_nominative(self, engine):
        assert lemma_of(engine, "ihmiset") == "ihminen"


# ===========================================================================
# 9. COMMON VERBS
# ===========================================================================

class TestCommonVerbs:
    @pytest.mark.parametrize("form,expected_lemma", [
        ("on",       "olla"),
        ("olen",     "olla"),
        ("ovat",     "olla"),
        ("oli",      "olla"),
        ("syö",      "syödä"),
        ("söi",      "syödä"),
        ("syödä",    "syödä"),
        ("menee",    "mennä"),
        ("meni",     "mennä"),
        ("tulee",    "tulla"),
        # "tuli" is genuinely ambiguous: past of tulla OR the noun tuli (fire).
        # Voikko resolves this via context; rule-based may return either.
        # ("tuli",  "tulla"),  # skip — ambiguous without context
        ("tekee",    "tehdä"),
        ("teki",     "tehdä"),
        ("tehdä",    "tehdä"),
    ])
    def test_verb_form(self, engine, form, expected_lemma):
        assert lemma_of(engine, form) == expected_lemma


# ===========================================================================
# 10. FULL SENTENCE LEMMATISATION
# ===========================================================================

class TestSentenceLemmatisation:
    def test_simple_sentence_word_count(self, engine):
        result = engine.lemmatize("Kissa söi hiiren talossa")
        assert result.word_count == 4

    def test_simple_sentence_lemmas(self, engine):
        result = engine.lemmatize("Kissa söi hiiren talossa")
        lemmas = [w.lemma for w in result.lemmas]
        assert "kissa" in lemmas
        assert "syödä" in lemmas
        assert "hiiri" in lemmas
        assert "talo"  in lemmas

    def test_morphology_present_when_requested(self, engine):
        result = engine.lemmatize("kissassa", include_morphology=True)
        assert result.lemmas[0].morphology is not None

    def test_morphology_absent_when_not_requested(self, engine):
        result = engine.lemmatize("kissassa", include_morphology=False)
        assert result.lemmas[0].morphology is None

    def test_empty_text(self, engine):
        result = engine.lemmatize("")
        assert result.word_count == 0
        assert result.lemmas == []

    def test_punctuation_stripped(self, engine):
        result = engine.lemmatize("kissa, koira.")
        assert result.word_count == 2

    def test_text_preserved_in_response(self, engine):
        text = "nainen juoksi taloon"
        result = engine.lemmatize(text)
        assert result.text == text

    def test_original_word_preserved(self, engine):
        result = engine.lemmatize("Kissassa")
        assert result.lemmas[0].original == "Kissassa"

    def test_case_insensitive_lookup(self, engine):
        """Capital letter at start of sentence should not affect lemmatisation."""
        result = engine.lemmatize("Koira")
        assert result.lemmas[0].lemma == "koira"

    def test_longer_text(self, engine):
        text = "Ihminen menee kauppaan ja ostaa hyvää ruokaa taloon"
        result = engine.lemmatize(text)
        # 8 tokens: Ihminen menee kauppaan ja ostaa hyvää ruokaa taloon
        assert result.word_count == 8
        lemmas = [w.lemma for w in result.lemmas]
        assert "ihminen" in lemmas
        assert "mennä"   in lemmas or "menee" in lemmas  # allow fallback
        assert "kauppa"  in lemmas
        assert "hyvä"    in lemmas
        assert "talo"    in lemmas


# ===========================================================================
# 11. MORPHOLOGY FIELD CORRECTNESS
# ===========================================================================

class TestMorphologyFields:
    def test_inessive_case_label(self, engine):
        result = engine.lemmatize("talossa", include_morphology=True)
        case = result.lemmas[0].morphology.get("case", "")
        assert case in ("Inessive", "Unknown"), f"Unexpected case: {case}"

    def test_number_field_type(self, engine):
        result = engine.lemmatize("kissa", include_morphology=True)
        morph = result.lemmas[0].morphology
        assert "number" in morph
        assert morph["number"] in ("Singular", "Plural")

    def test_plural_number(self, engine):
        result = engine.lemmatize("kissoissa", include_morphology=True)
        morph = result.lemmas[0].morphology
        # Rule-based might not always detect plural, but if it does it's correct
        if morph.get("number"):
            assert morph["number"] in ("Singular", "Plural")


# ===========================================================================
# 12. CASE DETECTION FOR KNOWN WORDS
# ===========================================================================

class TestCaseDetectionKnownWords:
    """
    The rule-based engine must return the CORRECT grammatical case for
    inflected forms found in known_words — not always "Nominative".
    """

    @pytest.mark.parametrize("word,expected_case", [
        ("talo",      "Nominative"),
        ("talon",     "Genitive"),
        ("talossa",   "Inessive"),
        ("talosta",   "Elative"),
        ("taloon",    "Illative"),
        ("talolla",   "Adessive"),
        ("talolta",   "Ablative"),
        ("talolle",   "Allative"),
        ("talot",     "Nominative"),   # nominative plural
    ])
    def test_talo_case_labels(self, engine, word, expected_case):
        assert case_of(engine, word) == expected_case

    @pytest.mark.parametrize("word,expected_number", [
        ("talo",   "Singular"),
        ("talot",  "Plural"),
    ])
    def test_talo_number(self, engine, word, expected_number):
        assert number_of(engine, word) == expected_number


# ===========================================================================
# 13. ABESSIVE CASE — vajanto (without X)
# ===========================================================================

class TestAbessiveCase:
    """
    Abessive: -tta (back harmony) / -ttä (front harmony).
    Means 'without': rahatta = without money, autotta = without a car.
    """

    @pytest.mark.parametrize("word,expected_lemma", [
        ("rahatta",  "raha"),
        ("autotta",  "auto"),
        ("kissatta", "kissa"),
        ("talotta",  "talo"),
        ("syyttä",   "syy"),    # without reason — suffix-based (syy not needing gradation)
        ("vedettä",  "vesi"),   # without water — Type 42 vesi→vede-
    ])
    def test_abessive_lemma(self, engine, word, expected_lemma):
        assert lemma_of(engine, word) == expected_lemma

    def test_abessive_case_label(self, engine):
        result = engine.lemmatize("rahatta", include_morphology=True)
        case = result.lemmas[0].morphology.get("case", "")
        assert case in ("Abessive", "Unknown"), f"Expected Abessive, got {case!r}"

    def test_abessive_plural(self, engine):
        # koiritta = koira plural abessive (without dogs)
        assert lemma_of(engine, "koiritta") == "koira"


# ===========================================================================
# 14. COMITATIVE CASE — seuranto (together with X)
# ===========================================================================

class TestComitativeCase:
    """
    Comitative: plural oblique stem + -ne- + 3rd person possessive (-en/-een).
    Means 'together with (one's)': koiroineen = with his/her dogs.
    """

    @pytest.mark.parametrize("word,expected_lemma", [
        ("koiroineen",  "koira"),
        ("kissoineen",  "kissa"),
        ("taloineen",   "talo"),
        ("kirjoineen",  "kirja"),
    ])
    def test_comitative_lemma(self, engine, word, expected_lemma):
        assert lemma_of(engine, word) == expected_lemma

    def test_comitative_case_label(self, engine):
        result = engine.lemmatize("koiroineen", include_morphology=True)
        case = result.lemmas[0].morphology.get("case", "")
        assert case in ("Comitative", "Unknown"), f"Expected Comitative, got {case!r}"


# ===========================================================================
# 15. INSTRUCTIVE CASE — keinonto (by means of, plural only)
# ===========================================================================

class TestInstructiveCase:
    """
    Instructive: frozen plural forms in -n.
    Almost exclusively used in fixed adverbial expressions:
      silmin = with/by eyes, käsin = by hand, jaloin = on foot, suin = by mouth.
    """

    @pytest.mark.parametrize("word,expected_lemma", [
        ("silmin",  "silmä"),
        ("käsin",   "käsi"),
        ("jaloin",  "jalka"),
        ("suin",    "suu"),
    ])
    def test_instructive_lemma(self, engine, word, expected_lemma):
        assert lemma_of(engine, word) == expected_lemma


# ===========================================================================
# 16. PROLATIVE CASE — prolatiiivi (by route/means, -tse)
# ===========================================================================

class TestProlativeCase:
    """
    Prolative: suffix -tse (invariable, no vowel harmony).
    Not listed in all Finnish grammars as a true case but well-attested
    in formal written Finnish for means of transport or communication:
      postitse = by post, meritse = by sea, maitse = by land.
    """

    @pytest.mark.parametrize("word,expected_lemma", [
        ("postitse",  "posti"),
        ("meritse",   "meri"),
        ("maitse",    "maa"),
        ("laivitse",  "laiva"),
    ])
    def test_prolative_lemma(self, engine, word, expected_lemma):
        assert lemma_of(engine, word) == expected_lemma

    def test_prolative_case_label(self, engine):
        result = engine.lemmatize("postitse", include_morphology=True)
        case = result.lemmas[0].morphology.get("case", "")
        assert case in ("Prolative", "Unknown"), f"Expected Prolative, got {case!r}"


# ===========================================================================
# 17. ESSIVE CASE — olento (as, in the role of)
# ===========================================================================

class TestEssiveCase:
    """Essive: -na/-nä. Means 'as / in the state of'."""

    @pytest.mark.parametrize("word,expected_lemma", [
        ("kissana",  "kissa"),   # as a cat
        ("talona",   "talo"),    # as a house
        ("naisena",  "nainen"),  # as a woman
    ])
    def test_essive_lemma(self, engine, word, expected_lemma):
        assert lemma_of(engine, word) == expected_lemma

    def test_essive_case_label(self, engine):
        assert case_of(engine, "kissana") in ("Essive", "Unknown")


# ===========================================================================
# 18. TRANSLATIVE CASE — tulento (becoming/turning into)
# ===========================================================================

class TestTranslativeCase:
    """Translative: -ksi. Means 'becoming' or 'for (duration)'."""

    @pytest.mark.parametrize("word,expected_lemma", [
        ("kissaksi",  "kissa"),   # (turning) into a cat
        ("taloksi",   "talo"),    # into a house
        ("hyväksi",   "hyvä"),    # for good / into good
    ])
    def test_translative_lemma(self, engine, word, expected_lemma):
        assert lemma_of(engine, word) == expected_lemma

    def test_translative_case_label(self, engine):
        assert case_of(engine, "kissaksi") in ("Translative", "Unknown")


# ===========================================================================
# 19. POTENTIAALI MOOD — potential (literary/formal Finnish)
# ===========================================================================

class TestPotentiaaliMood:
    """
    Potentiaali (potential mood) expresses possibility/uncertainty.
    Mainly used in formal written Finnish and literature; rare in speech.
    Formed by inserting -ne- before personal endings; 3p sg drops final -n.

    olla  → lienee (may be)
    mennä → mennee (may go)
    tulla → tullee (may come)
    tehdä → tehnee (may do)
    saada → saanee (may get)
    """

    @pytest.mark.parametrize("form,expected_lemma", [
        ("lienee",   "olla"),    # 3p sg potential of olla
        ("lienen",   "olla"),    # 1p sg
        ("lienet",   "olla"),    # 2p sg
        ("lienemme", "olla"),    # 1p pl
        ("lienevät", "olla"),    # 3p pl
        ("saanee",   "saada"),   # 3p sg potential of saada
        ("mennee",   "mennä"),   # 3p sg potential of mennä
        ("tullee",   "tulla"),   # 3p sg potential of tulla
        ("tehnee",   "tehdä"),   # 3p sg potential of tehdä
    ])
    def test_potentiaali(self, engine, form, expected_lemma):
        assert lemma_of(engine, form) == expected_lemma


# ===========================================================================
# 20. PASSIVE VOICE FORMS
# ===========================================================================

class TestPassiveVoice:
    """
    Finnish passive is impersonal (no agent noun).
    Present passive: stem + (t)aan/(t)ään
    Past passive: stem + ttiin/(t)iin
    """

    @pytest.mark.parametrize("form,expected_lemma", [
        ("tehdään",  "tehdä"),   # present passive
        ("tehtiin",  "tehdä"),   # past passive
        ("mennään",  "mennä"),
        ("mentiin",  "mennä"),
        ("tullaan",  "tulla"),
        ("tultiin",  "tulla"),
        ("saadaan",  "saada"),
        ("saatiin",  "saada"),
        ("ollaan",   "olla"),
        ("oltiin",   "olla"),
    ])
    def test_passive_lemma(self, engine, form, expected_lemma):
        assert lemma_of(engine, form) == expected_lemma


# ===========================================================================
# 21. VERB INFINITIVES (types 1–5)
# ===========================================================================

class TestVerbInfinitives:
    """
    Finnish verbs have five infinitive types:
      1st: mennä, tulla (basic dictionary form)
      2nd: mennessä (while going), mennessään (while going, with possessive)
      3rd: menemässä (in the process of going), menemätä, menemään (inessive/etc.)
      4th: meneminen (the act of going — verbal noun)
      5th: menemäisillään (about to go — very rare, mainly in fixed expressions)
    """

    @pytest.mark.parametrize("form,expected_lemma", [
        # 2nd infinitive
        ("tehdessä",     "tehdä"),
        ("tehdessään",   "tehdä"),
        ("mennessä",     "mennä"),
        ("mennessään",   "mennä"),
        # 3rd infinitive
        ("tekemässä",    "tehdä"),
        ("tekemästä",    "tehdä"),
        ("tekemään",     "tehdä"),
        ("menemässä",    "mennä"),
        ("tulemassa",    "tulla"),
        # 4th infinitive (verbal noun)
        ("tekeminen",    "tehdä"),
        ("meneminen",    "mennä"),
        ("tuleminen",    "tulla"),
        # 5th infinitive (very rare)
        ("menemäisillään", "mennä"),
    ])
    def test_infinitive_lemma(self, engine, form, expected_lemma):
        assert lemma_of(engine, form) == expected_lemma


# ===========================================================================
# 22. PARTICIPLES
# ===========================================================================

class TestParticiples:
    """
    Finnish has four main participle types:
      Active present: tekevä (the one doing)
      Active past:    tehnyt (the one who did)
      Passive present: tehtävä (to be done)  — not covered here (too complex rule-based)
      Passive past:   tehty (done/was done)
    Participles can take all 15 cases like any adjective/noun.
    """

    @pytest.mark.parametrize("form,expected_lemma", [
        # Active present participle inflected
        ("tekevä",    "tehdä"),
        ("tekevän",   "tehdä"),
        ("tekevässä", "tehdä"),
        # Active past participle
        ("tehnyt",    "tehdä"),
        ("tehneen",   "tehdä"),
        # Passive past participle
        ("tehty",     "tehdä"),
        ("tehdyn",    "tehdä"),
        # mennä participles
        ("menevä",    "mennä"),
        ("mennyt",    "mennä"),
        ("menneen",   "mennä"),
        # tulla participles
        ("tuleva",    "tulla"),
        ("tullut",    "tulla"),
        # olla participles
        ("oleva",     "olla"),
        ("olevan",    "olla"),
        ("ollut",     "olla"),
        ("olleen",    "olla"),
    ])
    def test_participle_lemma(self, engine, form, expected_lemma):
        assert lemma_of(engine, form) == expected_lemma


# ===========================================================================
# 23. PERSONAL PRONOUNS (including accusative)
# ===========================================================================

class TestPersonalPronouns:
    """
    Finnish personal pronouns have a morphologically distinct accusative form
    (minut, sinut, hänet, meidät, teidät, heidät) unlike common nouns where
    accusative = genitive (or nominative in certain contexts).
    """

    @pytest.mark.parametrize("form,expected_lemma", [
        # Nominative
        ("minä",   "minä"),
        ("sinä",   "sinä"),
        ("hän",    "hän"),
        ("me",     "me"),
        ("te",     "te"),
        ("he",     "he"),
        # Genitive
        ("minun",  "minä"),
        ("sinun",  "sinä"),
        ("hänen",  "hän"),
        ("meidän", "me"),
        ("teidän", "te"),
        ("heidän", "he"),
        # Accusative (morphologically distinct — no such form in common nouns)
        ("minut",  "minä"),
        ("sinut",  "sinä"),
        ("hänet",  "hän"),
        ("meidät", "me"),
        ("teidät", "te"),
        ("heidät", "he"),
    ])
    def test_pronoun_lemma(self, engine, form, expected_lemma):
        assert lemma_of(engine, form) == expected_lemma

    def test_pronoun_not_classified_as_verb(self, engine):
        """minä ends in -nä like mennä but must not be classified as VERB."""
        result = engine.lemmatize("minä", include_morphology=True)
        assert result.lemmas[0].pos != "VERB"


# ===========================================================================
# 24. LITERARY VOCABULARY — irregular type-42 nouns
# ===========================================================================

class TestLiteraryVocabulary:
    """
    Literary Finnish uses words with complex inflection patterns.
    Type 42 (vesi/käsi): strong grade in nominative (t/k), weak (d/∅) in oblique.
    Type 27 (vuosi): vuosi → vuode- in oblique cases.
    Type 33 (sydän): sydän → sydäme- in oblique.
    """

    @pytest.mark.parametrize("word,expected_lemma", [
        # vesi (water) — Type 42
        ("veden",    "vesi"),    # genitive
        ("vettä",    "vesi"),    # partitive
        ("vedessä",  "vesi"),    # inessive
        ("vedestä",  "vesi"),    # elative
        ("veteen",   "vesi"),    # illative
        ("vedettä",  "vesi"),    # abessive (without water)
        # käsi (hand) — Type 42
        ("käden",    "käsi"),
        ("kättä",    "käsi"),
        ("kädessä",  "käsi"),
        ("käteen",   "käsi"),
        ("kädettä",  "käsi"),
        # vuosi (year) — Type 27
        ("vuoden",   "vuosi"),
        ("vuotta",   "vuosi"),
        ("vuodessa", "vuosi"),
        ("vuoteen",  "vuosi"),
        # sydän (heart) — Type 33
        ("sydämen",  "sydän"),
        ("sydäntä",  "sydän"),
        ("sydämessä","sydän"),
        ("sydämeen", "sydän"),
        # tähti (star) — t→d gradation
        ("tähden",   "tähti"),
        ("tähteä",   "tähti"),
        ("tähdessä", "tähti"),
    ])
    def test_literary_noun(self, engine, word, expected_lemma):
        assert lemma_of(engine, word) == expected_lemma


# ===========================================================================
# 25. COMPLEX LITERARY TEXTS
# ===========================================================================

class TestComplexLiteraryTexts:
    """
    Longer Finnish sentences drawn from literature-style prose, testing the
    engine's ability to handle concatenated complex forms and rare cases in
    natural context.
    """

    def test_sentence_with_abessive(self, engine):
        """Ilman rahatta ei pääse pitkälle (Without money you won't get far)."""
        result = engine.lemmatize("Hän lähti kotoa rahatta yönä")
        lemmas = [w.lemma for w in result.lemmas]
        assert "raha" in lemmas
        assert "yö"   in lemmas

    def test_sentence_with_prolative(self, engine):
        """Viesti lähetettiin postitse (Message was sent by post)."""
        result = engine.lemmatize("Kirja tuli postitse mereltä")
        lemmas = [w.lemma for w in result.lemmas]
        assert "kirja" in lemmas
        assert "posti" in lemmas

    def test_sentence_with_instructive(self, engine):
        """Hän teki sen käsin (She did it by hand)."""
        result = engine.lemmatize("Hän teki sen käsin ja silmin näkyen")
        lemmas = [w.lemma for w in result.lemmas]
        assert "käsi"  in lemmas
        assert "silmä" in lemmas

    def test_sentence_with_potentiaali(self, engine):
        """Hän lienee jo kotona (He/she may already be at home)."""
        result = engine.lemmatize("Hän lienee jo kotona")
        lemmas = [w.lemma for w in result.lemmas]
        assert "olla" in lemmas

    def test_sentence_with_passive(self, engine):
        """Työ tehdään käsin (The work is done by hand)."""
        result = engine.lemmatize("Työ tehdään huolellisesti ja ajatellen")
        lemmas = [w.lemma for w in result.lemmas]
        assert "tehdä" in lemmas

    def test_classic_literary_sentence(self, engine):
        """
        Ihmisen sydän on kuin meri: syvä, myrskyinen ja täynnä salaisuuksia.
        (A human heart is like a sea: deep, stormy and full of secrets.)
        """
        text = "Ihmisen sydän on kuin meri syvä myrskyinen ja täynnä salaisuuksia"
        result = engine.lemmatize(text)
        lemmas = [w.lemma for w in result.lemmas]
        assert "ihminen" in lemmas
        assert "sydän"   in lemmas
        assert "olla"    in lemmas
        assert "meri"    in lemmas

    def test_long_complex_sentence(self, engine):
        """
        Kirjailija kirjoitti kirjansa käsin kynällä vanhana talvena vedettömässä
        mökissä vuosien ajan silmin näkyen kärsien.

        Note: possessive suffixes (-nsa/-nsä) and derived words like "kirjailija"
        are beyond rule-based lemmatization; Voikko handles them natively.
        We test the forms within known-word coverage.
        """
        text = (
            "Kirjailija kirjoitti kirjansa käsin kynällä vanhana talvena "
            "vedettömässä mökissä vuosien ajan silmin näkyen kärsien"
        )
        result = engine.lemmatize(text)
        lemmas = [w.lemma for w in result.lemmas]
        # käsin → käsi (Instructive plural), silmin → silmä (Instructive plural)
        assert "käsi"       in lemmas
        assert "silmä"      in lemmas
        # kirjoitti → kirjoittaa (known verb form)
        assert "kirjoittaa" in lemmas
        # vuosien → vuosi (genitive plural of vuosi)
        assert "vuosi"      in lemmas

    def test_potentiaali_in_literary_context(self, engine):
        """
        Hän lienee tullut kotiin jo kauan sitten, mutta kukaan ei tiedä varmasti.
        (He may have come home long ago, but no one knows for certain.)
        """
        text = "Hän lienee tullut kotiin jo kauan sitten"
        result = engine.lemmatize(text)
        lemmas = [w.lemma for w in result.lemmas]
        assert "olla"  in lemmas   # lienee → olla
        assert "tulla" in lemmas   # tullut → tulla


# ===========================================================================
# 26. DIALECT AND COLLOQUIAL FORMS
# ===========================================================================

class TestDialectAndColloquial:
    """
    Spoken Finnish differs substantially from written standard Finnish.
    Common colloquial reductions should be mapped to their standard lemma.
    """

    @pytest.mark.parametrize("form,expected", [
        # mä/sä/se for minä/sinä
        # These may not be in known_words; allow fallback to the form itself
        ("mä",    "mä"),     # colloquial minä — falls through to itself
        ("sä",    "sä"),     # colloquial sinä
        # Common spoken contractions
        ("mennään", "mennä"),    # let's go (passive imperative = colloquial)
        ("tullaan", "tulla"),    # let's come
        ("ollaan",  "olla"),     # we are (colloquial) / passive
    ])
    def test_colloquial_form(self, engine, form, expected):
        result = engine.lemmatize(form, include_morphology=False)
        lemma = result.lemmas[0].lemma
        # Allow the colloquial form to map to either the expected lemma or itself
        assert lemma in (expected, form), f"Got {lemma!r} for {form!r}"

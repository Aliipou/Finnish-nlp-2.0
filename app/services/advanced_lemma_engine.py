"""
Advanced Finnish Lemmatizer Engine with Voikko Integration
Real morphological analysis for Finnish language
"""
import logging
import re
from typing import List, Dict, Optional, Tuple
from app.models.schemas import LemmatizationResponse, WordLemma

logger = logging.getLogger(__name__)

# Voikko SIJAMUOTO (Finnish case name) -> English mapping
# Bug fix: Voikko returns Finnish-language case names; previous code stored
# them untranslated, causing incorrect morphology output like "nimento"
# instead of "Nominative".
VOIKKO_CASE_MAP: Dict[str, str] = {
    "nimento":     "Nominative",
    "omanto":      "Genitive",
    "osanto":      "Partitive",
    "olento":      "Essive",
    "tulento":     "Translative",
    "sisaolento":  "Inessive",
    "sisaeronto":  "Elative",
    "sisatulento": "Illative",
    "ulko-olento": "Adessive",
    "ulkoeronto":  "Ablative",
    "ulkotulento": "Allative",
    "vajanto":     "Abessive",
    "seuranto":    "Comitative",
    "keinonto":    "Instructive",
}

VOIKKO_POS_MAP: Dict[str, str] = {
    "nimisana":              "NOUN",
    "laatusana":             "ADJ",
    "nimisana_laatusana":    "ADJ",
    "teonsana":              "VERB",
    "seikkasana":            "ADV",
    "asemosana":             "PRON",
    "suhdesana":             "ADP",
    "huudahdussana":         "INTJ",
    "sidesana":              "CONJ",
    "kieltosana":            "PART",
    "lukusana":              "NUM",
    "etunimi":               "PROPN",
    "sukunimi":              "PROPN",
    "paikannimi":            "PROPN",
}

VOIKKO_NUMBER_MAP: Dict[str, str] = {
    "singular": "Singular",
    "plural":   "Plural",
    "yksikko":  "Singular",
    "monikko":  "Plural",
}


def _translate_case(voikko_case: Optional[str]) -> str:
    if not voikko_case:
        return "Unknown"
    return VOIKKO_CASE_MAP.get(voikko_case.lower(), voikko_case)


def _translate_pos(voikko_class: Optional[str]) -> str:
    if not voikko_class:
        return "UNKNOWN"
    return VOIKKO_POS_MAP.get(voikko_class.lower(), voikko_class.upper())


def _translate_number(voikko_number: Optional[str]) -> str:
    if not voikko_number:
        return "Singular"
    return VOIKKO_NUMBER_MAP.get(voikko_number.lower(), voikko_number.capitalize())


def _best_analysis(analyses: list) -> dict:
    """
    Select the best analysis when Voikko returns multiple candidates.

    Voikko may return several parses for ambiguous forms. We prefer:
    1. Analyses that have a BASEFORM.
    2. Content-word classes (noun, verb, adjective) over function words.
    3. Otherwise the first result.
    """
    if not analyses:
        return {}
    if len(analyses) == 1:
        return analyses[0]

    content_classes = {"nimisana", "teonsana", "laatusana", "nimisana_laatusana"}
    with_base = [a for a in analyses if a.get("BASEFORM")]
    if not with_base:
        return analyses[0]
    content = [a for a in with_base if a.get("CLASS", "").lower() in content_classes]
    return content[0] if content else with_base[0]


class AdvancedLemmatizerEngine:
    """
    Production-grade Finnish lemmatizer using libvoikko.

    When libvoikko is available it provides accurate morphological analysis
    covering all 15 grammatical cases, consonant gradation (astevaihtelu),
    vowel harmony, compound words, verb conjugations, and clitics.

    Falls back to a rule-based engine when Voikko is unavailable.  The
    fallback covers the 14 most common cases plus Potentiaali mood, and
    handles literary/rare forms through an extended known-word dictionary.
    """

    def __init__(self, use_voikko: bool = True):
        logger.info("Initialising Advanced Finnish Lemmatizer Engine")
        self.voikko = None
        self.use_voikko = use_voikko

        if use_voikko:
            try:
                import libvoikko
                self.voikko = libvoikko.Voikko("fi")
                logger.info("Voikko library loaded successfully")
            except ImportError:
                logger.warning(
                    "libvoikko not available — falling back to rule-based engine. "
                    "Install: pip install libvoikko  (requires system libvoikko-dev)"
                )
            except Exception as e:
                logger.warning("Voikko initialisation failed (%s) — using fallback", e)

        self._init_fallback_rules()
        logger.info("Lemmatizer ready (voikko=%s)", bool(self.voikko))

    # ------------------------------------------------------------------
    # Fallback rule-based tables
    # ------------------------------------------------------------------

    def _init_fallback_rules(self) -> None:
        # Ordered suffix list — longer/more-specific patterns come first.
        # Plural patterns precede singular ones to avoid false-positive
        # partial matches on plural stems.
        self.case_patterns: List[tuple] = [
            # Plural cases
            ("Abessive",    "Plural",   ["oitta", "öittä", "itta", "ittä"]),
            ("Comitative",  "Plural",   ["neen", "nesi", "neni", "nemme", "nenne"]),
            ("Partitive",   "Plural",   ["oja", "öjä", "ita", "itä", "ia", "iä"]),
            ("Genitive",    "Plural",   ["iden", "itten", "ojen", "öjen", "ien", "jen"]),
            ("Inessive",    "Plural",   ["issa", "issä"]),
            ("Elative",     "Plural",   ["ista", "istä"]),
            ("Illative",    "Plural",   ["isiin", "ihin", "oihin", "öihin"]),
            ("Adessive",    "Plural",   ["oilla", "öillä", "illa", "illä"]),
            ("Ablative",    "Plural",   ["oilta", "öiltä", "ilta", "iltä"]),
            ("Allative",    "Plural",   ["oille", "öille", "ille"]),
            ("Essive",      "Plural",   ["oina", "öinä", "ina", "inä"]),
            ("Translative", "Plural",   ["oiksi", "öiksi", "iksi"]),
            ("Instructive", "Plural",   ["in"]),
            # Prolative — postitse, meritse, maitse (by means of transport/route)
            # Rare; not recognised by all Finnish grammars as a true case but
            # well-attested in formal/written Finnish.
            ("Prolative",   "Singular", ["tse", "itse"]),
            # Singular cases
            # Illative: long-vowel+n forms (kissaan, taloon, hiireen) must
            # precede genitive -n to avoid swallowing single-n forms.
            ("Illative",    "Singular", ["seen", "siin", "hin", "hon", "hön", "hun",
                                          "hyn", "aan", "ään", "oon", "een", "iin",
                                          "uun", "yyn"]),
            # Abessive before Partitive so "tta"/"ttä" binds before "ta"/"tä"
            ("Abessive",    "Singular", ["tta", "ttä"]),
            ("Inessive",    "Singular", ["ssa", "ssä"]),
            ("Elative",     "Singular", ["sta", "stä"]),
            ("Adessive",    "Singular", ["lla", "llä"]),
            ("Ablative",    "Singular", ["lta", "ltä"]),
            ("Allative",    "Singular", ["lle"]),
            ("Translative", "Singular", ["ksi"]),
            ("Essive",      "Singular", ["na", "nä"]),
            ("Partitive",   "Singular", ["aa", "ää", "ta", "tä", "a", "ä"]),
            ("Genitive",    "Singular", ["n"]),
        ]

        # Verb suffixes used in is_verb detection
        self._verb_endings = (
            "da", "dä", "ta", "tä", "la", "lä", "ra", "rä", "na", "nä",
        )

        # Known pronouns — excluded from verb-ending heuristic
        # (minä ends with "nä" like mennä but is a pronoun, not a verb)
        self._pronoun_lemmas = {
            "minä", "sinä", "hän", "me", "te", "he",
            "se", "ne", "tämä", "tuo", "joka", "mikä", "kuka",
        }

        # Known words: lemma → [inflected forms].
        # Entries cover the 14 grammatical cases in singular and plural plus
        # abessive, comitative, instructive, and rare/literary forms.
        self.known_words: Dict[str, List[str]] = {
            # ── Core nouns ──────────────────────────────────────────────
            "talo": [
                "talo", "talon", "taloa", "talossa", "talosta", "taloon",
                "talolla", "talolta", "talolle", "talona", "taloksi", "talotta",
                "talot", "talojen", "taloja", "taloissa", "taloista", "taloihin",
                "taloilla", "taloilta", "taloille", "taloina", "taloiksi",
                "taloitta", "taloineen",
            ],
            "koira": [
                "koira", "koiran", "koiraa", "koirassa", "koirasta", "koiraan",
                "koiralla", "koiralta", "koiralle", "koirana", "koiraksi",
                "koiratta", "koirat", "koirien", "koiria", "koirissa", "koirista",
                "koiriin", "koirilla", "koirilta", "koirille", "koirina",
                "koiriksi", "koiritta", "koiroineen",
            ],
            "auto": [
                "auto", "auton", "autoa", "autossa", "autosta", "autoon",
                "autolla", "autolta", "autolle", "autona", "autoksi", "autotta",
                "autot", "autojen", "autoja", "autoissa", "autoista", "autoihin",
                "autoilla", "autoilta", "autoille", "autoina", "autoiksi",
                "autoitta", "autoineen",
            ],
            "kissa": [
                "kissa", "kissan", "kissaa", "kissassa", "kissasta", "kissaan",
                "kissalla", "kissalta", "kissalle", "kissana", "kissaksi",
                "kissatta", "kissat", "kissojen", "kissoja", "kissoissa",
                "kissoista", "kissoihin", "kissoilla", "kissoilta", "kissoille",
                "kissoina", "kissoiksi", "kissoitta", "kissoineen",
            ],
            "hiiri": [
                "hiiri", "hiiren", "hiirtä", "hiiressä", "hiirestä", "hiireen",
                "hiirellä", "hiireltä", "hiirelle", "hiirenä", "hiireksi",
                "hiirrettä", "hiiret", "hiirten", "hiiriä", "hiirissä",
                "hiiristä", "hiiriin", "hiirillä", "hiiriltä", "hiirille",
                "hiirinä", "hiiriksi", "hiirittä", "hiireineen",
            ],
            "nainen": [
                "nainen", "naisen", "naista", "naisessa", "naisesta", "naiseen",
                "naisella", "naiselta", "naiselle", "naisena", "naiseksi",
                "naisetta", "naiset", "naisten", "naisia", "naisissa", "naisista",
                "naisiin", "naisilla", "naisilta", "naisille", "naisina",
                "naisiksi", "naisitta", "naisineen",
            ],
            "ihminen": [
                "ihminen", "ihmisen", "ihmistä", "ihmisessä", "ihmisestä",
                "ihmiseen", "ihmisellä", "ihmiseltä", "ihmiselle", "ihmisenä",
                "ihmiseksi", "ihmisettä", "ihmiset", "ihmisten", "ihmisiä",
                "ihmisissä", "ihmisistä", "ihmisiin", "ihmisillä", "ihmisiltä",
                "ihmisille", "ihmisinä", "ihmisiksi", "ihmisineen",
            ],
            "kauppa": [
                "kauppa", "kaupan", "kauppaa", "kaupassa", "kaupasta",
                "kauppaan", "kaupalla", "kaupalta", "kaupalle", "kauppana",
                "kaupaksi", "kaupatta", "kaupat", "kauppojen", "kauppoja",
                "kaupoissa", "kaupoista", "kauppoihin", "kaupoilla", "kaupoilta",
                "kaupoille", "kauppoina", "kaupoiksi", "kaupoitta", "kauppoineen",
            ],
            "hattu": [
                "hattu", "hatun", "hattua", "hatussa", "hatusta", "hattuun",
                "hatulla", "hatulta", "hatulle", "hattuna", "hatuksi",
                "hatutta", "hatut", "hattujen", "hattuja", "hatuissa", "hatuista",
                "hattuihin", "hatuilla", "hatuilta", "hatuille", "hattuina",
                "hatuksi", "hatuitta", "hattuineen",
            ],
            "pöytä": [
                "pöytä", "pöydän", "pöytää", "pöydässä", "pöydästä", "pöytään",
                "pöydällä", "pöydältä", "pöydälle", "pöytänä", "pöydäksi",
                "pöydättä", "pöydät", "pöytien", "pöytiä", "pöydissä",
                "pöydistä", "pöytiin", "pöydillä", "pöydiltä", "pöydille",
                "pöytinä", "pöydiksi", "pöyditta", "pöytineen",
            ],
            "paras": [
                "paras", "parhaan", "parasta", "parhaassa", "parhaasta",
                "parhaaseen", "parhaalla", "parhaalta", "parhaalle", "parhaana",
                "parhaaksi", "parhaat", "parhaiden", "parhaita", "parhaissa",
                "parhaista", "parhaisiin", "parhailla", "parhailta", "parhaille",
                "parhaina", "parhaiksi",
            ],
            "hyvä": [
                "hyvä", "hyvän", "hyvää", "hyvässä", "hyvästä", "hyvään",
                "hyvällä", "hyvältä", "hyvälle", "hyvänä", "hyväksi",
                "hyvättä", "hyvät", "hyvien", "hyviä", "hyvissä", "hyvistä",
                "hyviin", "hyvillä", "hyviltä", "hyville", "hyvinä", "hyviksi",
                "hyvitta", "hyvineen",
            ],
            # ── Literary nouns ──────────────────────────────────────────
            # kirja (book) — regular a-stem
            "kirja": [
                "kirja", "kirjan", "kirjaa", "kirjassa", "kirjasta", "kirjaan",
                "kirjalla", "kirjalta", "kirjalle", "kirjana", "kirjaksi",
                "kirjatta", "kirjat", "kirjojen", "kirjoja", "kirjoissa",
                "kirjoista", "kirjoihin", "kirjoilla", "kirjoilta", "kirjoille",
                "kirjoina", "kirjoiksi", "kirjoitta", "kirjoineen",
            ],
            # maa (land/country/earth) — long-vowel stem
            "maa": [
                "maa", "maan", "maata", "maassa", "maasta", "maahan",
                "maalla", "maalta", "maalle", "maana", "maaksi", "maatta",
                "maat", "maiden", "maita", "maissa", "maista", "maihin",
                "mailla", "mailta", "maille", "maina", "maiksi", "maitta",
                "maineen",
                # Prolative: maitse = by land / overland
                "maitse",
            ],
            # yö (night) — front vowel ö
            "yö": [
                "yö", "yön", "yötä", "yössä", "yöstä", "yöhön", "yöllä",
                "yöltä", "yölle", "yönä", "yöksi", "yöttä", "yöt", "yöiden",
                "yöitä", "yöissä", "yöistä", "yöihin", "yöillä", "yöiltä",
                "yöille", "yöinä", "yöiksi", "yöitta", "yöineen",
            ],
            # silmä (eye) — ä-stem
            "silmä": [
                "silmä", "silmän", "silmää", "silmässä", "silmästä", "silmään",
                "silmällä", "silmältä", "silmälle", "silmänä", "silmäksi",
                "silmättä", "silmät", "silmien", "silmiä", "silmissä", "silmistä",
                "silmiin", "silmillä", "silmiltä", "silmille", "silminä",
                "silmiksi", "silmitta", "silmineen",
                # Instructive plural: silmin = by eye / with eyes
                "silmin",
            ],
            # sydän (heart) — Type 33, stem sydäme-
            "sydän": [
                "sydän", "sydämen", "sydäntä", "sydämessä", "sydämestä",
                "sydämeen", "sydämellä", "sydämeltä", "sydämelle", "sydämenä",
                "sydämeksi", "sydämettä", "sydämet", "sydänten", "sydämiä",
                "sydämissä", "sydämistä", "sydämiin", "sydämillä", "sydämiltä",
                "sydämille", "sydäminä", "sydämiksi", "sydämitta", "sydämineen",
            ],
            # vesi (water) — Type 42, t→d gradation
            "vesi": [
                "vesi", "veden", "vettä", "vedessä", "vedestä", "veteen",
                "vedellä", "vedeltä", "vedelle", "vetenä", "vedeksi", "vedettä",
                "vedet", "vesien", "vesiä", "vesissä", "vesistä", "vesiin",
                "vesillä", "vesiltä", "vesille", "vesinä", "vesiksi", "vesitta",
                "vesineen",
            ],
            # käsi (hand) — Type 42 like vesi
            "käsi": [
                "käsi", "käden", "kättä", "kädessä", "kädestä", "käteen",
                "kädellä", "kädeltä", "kädelle", "kätenä", "kädeksi", "kädettä",
                "kädet", "käsien", "käsiä", "käsissä", "käsistä", "käsiin",
                "käsillä", "käsiltä", "käsille", "käsinä", "käsiksi", "käsitta",
                "käsineen",
                # Instructive plural: käsin = by hand / manually
                "käsin",
            ],
            # vuosi (year) — Type 27, irregular
            "vuosi": [
                "vuosi", "vuoden", "vuotta", "vuodessa", "vuodesta", "vuoteen",
                "vuodella", "vuodelta", "vuodelle", "vuotena", "vuodeksi",
                "vuodetta", "vuodet", "vuosien", "vuosia", "vuosissa", "vuosista",
                "vuosiin", "vuosilla", "vuosilta", "vuosille", "vuosina",
                "vuosiksi", "vuositta", "vuosineen",
            ],
            # mieli (mind/will) — i-stem
            "mieli": [
                "mieli", "mielen", "mieltä", "mielessä", "mielestä", "mieleen",
                "mielellä", "mieleltä", "mielelle", "mielen", "mieleksi",
                "mielettä", "mielet", "mielten", "mieliä", "mielissä", "mielistä",
                "mieliin", "mielillä", "mieliltä", "mielille", "mielinä",
                "mieliksi", "mielitta", "mielineen",
            ],
            # tähtä (star) — ä-stem
            "tähti": [
                "tähti", "tähden", "tähteä", "tähdessä", "tähdestä", "tähteen",
                "tähdellä", "tähdeltä", "tähdelle", "tähtenä", "tähdeksi",
                "tähdettä", "tähdet", "tähtien", "tähtiä", "tähdissä", "tähdistä",
                "tähtiin", "tähdillä", "tähdiltä", "tähdille", "tähtinä",
                "tähdiksi", "tähtitta", "tähtineen",
            ],
            # jalka (foot/leg) — Type 44, k→∅ gradation
            "jalka": [
                "jalka", "jalan", "jalkaa", "jalassa", "jalasta", "jalkaan",
                "jalalla", "jalalta", "jalalle", "jalkana", "jalaksi",
                "jalkatta", "jalat", "jalkojen", "jalkoja", "jaloissa",
                "jaloista", "jalkoihin", "jaloilla", "jaloilta", "jaloille",
                "jalkoina", "jaloiksi", "jaloitta", "jalkoineen",
                # Instructive plural: jaloin = on foot
                "jaloin",
            ],
            # suu (mouth) — long-vowel stem
            "suu": [
                "suu", "suun", "suuta", "suussa", "suusta", "suuhun",
                "suulla", "suulta", "suulle", "suuna", "suuksi", "suutta",
                "suut", "suiden", "suita", "suissa", "suista", "suihin",
                "suilla", "suilta", "suille", "suina", "suiksi", "suitta",
                "suineen",
                # Instructive plural: suin = by mouth, completely (suin surmin = utterly)
                "suin",
            ],
            # syy (reason/cause) — yy-stem
            "syy": [
                "syy", "syyn", "syytä", "syyssä", "syystä", "syyhyn",
                "syyllä", "syyllä", "syyllä", "syynä", "syyksi", "syyttä",
                "syyt", "syiden", "syitä", "syissä", "syistä", "syihin",
                "syillä", "syiltä", "syille", "syinä", "syiksi", "syittä",
                "syineen",
            ],
            # raha (money)
            "raha": [
                "raha", "rahan", "rahaa", "rahassa", "rahasta", "rahaan",
                "rahalla", "rahalta", "rahalle", "rahana", "rahaksi", "rahatta",
                "rahat", "rahojen", "rahoja", "rahoissa", "rahoista", "rahoihin",
                "rahoilla", "rahoilta", "rahoille", "rahoina", "rahoiksi",
                "rahoitta", "rahoineen",
            ],
            # ── High-frequency nouns (all 15 cases × 2 numbers + abe + com) ──

            # mies (man) — Type 33, strong/weak stem mies→miehe-
            "mies": [
                "mies", "miehen", "miestä", "miehessä", "miehestä", "mieheen",
                "miehellä", "mieheltä", "miehelle", "miehenä", "mieheksi", "miehettä",
                "miehet", "miesten", "miehiä", "miehissä", "miehistä", "miehiin",
                "miehillä", "miehiltä", "miehille", "miehinä", "miehiksi", "miehittä",
                "miehineen",
            ],
            # lapsi (child) — Type 7, lapsi→lapse-
            "lapsi": [
                "lapsi", "lapsen", "lasta", "lapsessa", "lapsesta", "lapseen",
                "lapsella", "lapselta", "lapselle", "lapsena", "lapseksi", "lapsetta",
                "lapset", "lasten", "lapsia", "lapsissa", "lapsista", "lapsiin",
                "lapsilla", "lapsilta", "lapsille", "lapsina", "lapsiksi", "lapsitta",
                "lapsineen",
            ],
            # äiti (mother) — Type 3, tt→d gradation: äiti→äidin
            "äiti": [
                "äiti", "äidin", "äitiä", "äidissä", "äidistä", "äitiin",
                "äidillä", "äidiltä", "äidille", "äitinä", "äidiksi", "äidittä",
                "äidit", "äitien", "äitejä", "äideissä", "äideistä", "äiteihin",
                "äideillä", "äideiltä", "äideille", "äiteinä", "äideiksi", "äideittä",
                "äiteineen",
            ],
            # isä (father) — regular Type 1 ä-stem
            "isä": [
                "isä", "isän", "isää", "isässä", "isästä", "isään",
                "isällä", "isältä", "isälle", "isänä", "isäksi", "isättä",
                "isät", "isien", "isiä", "isissä", "isistä", "isihin",
                "isillä", "isiltä", "isille", "isinä", "isiksi", "isittä",
                "isineen",
            ],
            # ystävä (friend) — regular Type 1 ä-stem
            "ystävä": [
                "ystävä", "ystävän", "ystävää", "ystävässä", "ystävästä", "ystävään",
                "ystävällä", "ystävältä", "ystävälle", "ystävänä", "ystäväksi", "ystävättä",
                "ystävät", "ystävien", "ystäviä", "ystävissä", "ystävistä", "ystäviin",
                "ystävillä", "ystäviltä", "ystäville", "ystävinä", "ystäviksi", "ystävittä",
                "ystävineen",
            ],
            # perhe (family) — Type 6, e→ee stem: perhe→perheen
            "perhe": [
                "perhe", "perheen", "perhettä", "perheessä", "perheestä", "perheeseen",
                "perheellä", "perheeltä", "perheelle", "perheenä", "perheeksi", "perheettä",
                "perheet", "perheiden", "perheitä", "perheissä", "perheistä", "perheihin",
                "perheillä", "perheilta", "perheille", "perheinä", "perheiksi", "perheitta",
                "perheineeen",
            ],
            # päivä (day) — regular Type 1 ä-stem
            "päivä": [
                "päivä", "päivän", "päivää", "päivässä", "päivästä", "päivään",
                "päivällä", "päivältä", "päivälle", "päivänä", "päiväksi", "päivättä",
                "päivät", "päivien", "päiviä", "päivissä", "päivistä", "päiviin",
                "päivillä", "päiviltä", "päiville", "päivinä", "päiviksi", "päivittä",
                "päivineen",
            ],
            # elämä (life) — regular Type 1 ä-stem
            "elämä": [
                "elämä", "elämän", "elämää", "elämässä", "elämästä", "elämään",
                "elämällä", "elämältä", "elämälle", "elämänä", "elämäksi", "elämättä",
                "elämät", "elämien", "elämiä", "elämissä", "elämistä", "elämiin",
                "elämill ä", "elämiltä", "elämille", "eläminä", "elämiksi", "elämittä",
                "elämineen",
            ],
            # aika (time) — Type 9, kk→k gradation: aika→ajan
            "aika": [
                "aika", "ajan", "aikaa", "ajassa", "ajasta", "aikaan",
                "ajalla", "ajalta", "ajalle", "aikana", "ajaksi", "aikatta",
                "ajat", "aikojen", "aikoja", "ajoissa", "ajoista", "aikoihin",
                "ajoilla", "ajoilta", "ajoille", "aikoina", "ajoiksi", "ajoitta",
                "aikoineen",
            ],
            # puu (tree) — Type 3, long uu-stem
            "puu": [
                "puu", "puun", "puuta", "puussa", "puusta", "puuhun",
                "puulla", "puulta", "puulle", "puuna", "puuksi", "puutta",
                "puut", "puiden", "puita", "puissa", "puista", "puihin",
                "puilla", "puilta", "puille", "puina", "puiksi", "puitta",
                "puineen",
            ],
            # pää (head) — Type 3, long ää-stem
            "pää": [
                "pää", "pään", "päätä", "päässä", "päästä", "päähän",
                "päällä", "päältä", "päälle", "päänä", "pääksi", "päättä",
                "päät", "päiden", "päitä", "päissä", "päistä", "päihin",
                "päillä", "päiltä", "päille", "päinä", "päiksi", "päittä",
                "päineen",
            ],
            # tuli (fire) — Type 5, tuli→tule-
            "tuli": [
                "tuli", "tulen", "tulta", "tulessa", "tulesta", "tuleen",
                "tulella", "tulelta", "tulelle", "tulena", "tuleksi", "tuletta",
                "tulet", "tulien", "tulia", "tulissa", "tulista", "tuliin",
                "tulilla", "tulilta", "tulille", "tulina", "tuliksi", "tulitta",
                "tulineen",
            ],
            # ovi (door) — Type 5, ovi→ove-
            "ovi": [
                "ovi", "oven", "ovea", "ovessa", "ovesta", "oveen",
                "ovella", "ovelta", "ovelle", "ovena", "oveksi", "ovetta",
                "ovet", "ovien", "ovia", "ovissa", "ovista", "oviin",
                "ovilla", "ovilta", "oville", "ovina", "oviksi", "ovitta",
                "ovineen",
            ],
            # kivi (stone) — Type 5, kivi→kive-
            "kivi": [
                "kivi", "kiven", "kiveä", "kivessä", "kivestä", "kiveen",
                "kivellä", "kiveltä", "kivelle", "kivenä", "kiveksi", "kivettä",
                "kivet", "kivien", "kiviä", "kivissä", "kivistä", "kiviin",
                "kivillä", "kiviltä", "kiville", "kivinä", "kiviksi", "kivittä",
                "kivineen",
            ],
            # huone (room) — Type 6, huone→huonee-
            "huone": [
                "huone", "huoneen", "huonetta", "huoneessa", "huoneesta", "huoneeseen",
                "huoneella", "huoneelta", "huoneelle", "huoneena", "huoneeksi", "huoneetta",
                "huoneet", "huoneiden", "huoneita", "huoneissa", "huoneista", "huoneisiin",
                "huoneilla", "huoneilta", "huoneille", "huoneina", "huoneiksi", "huoneitta",
                "huoneineen",
            ],
            # onni (happiness/luck) — Type 5, onni→onne-
            "onni": [
                "onni", "onnen", "onnea", "onnessa", "onnesta", "onneen",
                "onnella", "onnelta", "onnelle", "onnena", "onneksi", "onnetta",
                "onnet", "onnien", "onnia", "onnissa", "onnista", "onniin",
                "onnilla", "onnilta", "onnille", "onnina", "onniksi", "onnitta",
                "onnineen",
            ],
            # rakkaus (love) — Type 40, -us→-uude-
            "rakkaus": [
                "rakkaus", "rakkauden", "rakkautta", "rakkaudessa", "rakkaudesta",
                "rakkauteen", "rakkaudella", "rakkaudelta", "rakkaudelle",
                "rakkautena", "rakkaudeksi", "rakkaudetta",
                # plural (rare for this abstract noun but grammatically valid)
                "rakkaudet", "rakkauksien", "rakkauksia",
            ],
            # vapaus (freedom) — Type 40, -us→-uude-
            "vapaus": [
                "vapaus", "vapauden", "vapautta", "vapaudessa", "vapaudesta",
                "vapauteen", "vapaudella", "vapaudelta", "vapaudelle",
                "vapautena", "vapaudeksi", "vapaudetta",
                "vapaudet", "vapauksien", "vapauksia",
            ],
            # tieto (knowledge/information) — Type 2, t→d gradation: tieto→tiedon
            "tieto": [
                "tieto", "tiedon", "tietoa", "tiedossa", "tiedosta", "tietoon",
                "tiedolla", "tiedolta", "tiedolle", "tietona", "tiedoksi", "tiedotta",
                "tiedot", "tietojen", "tietoja", "tiedoissa", "tiedoista", "tietoihin",
                "tiedoilla", "tiedoilta", "tiedoille", "tietoina", "tiedoiksi", "tiedoitta",
                "tietoineen",
            ],
            # lintu (bird) — NT→NN gradation: lintu→linnun
            "lintu": [
                "lintu", "linnun", "lintua", "linnussa", "linnusta", "lintuun",
                "linnulla", "linnulta", "linnulle", "lintuna", "linnuksi", "lintutta",
                "linnut", "lintujen", "lintuja", "linnuissa", "linnuista", "lintuihin",
                "linnuilla", "linnuilta", "linnuille", "lintuina", "linnuiksi", "lintuitta",
                "lintuineen",
            ],
            # kukka (flower) — pp→p gradation: kukka→kukan
            "kukka": [
                "kukka", "kukan", "kukkaa", "kukassa", "kukasta", "kukkaan",
                "kukalla", "kukalta", "kukalle", "kukkana", "kukaksi", "kukatta",
                "kukat", "kukkien", "kukkia", "kukissa", "kukista", "kukkiin",
                "kukilla", "kukilta", "kukille", "kukkina", "kukiksi", "kukitta",
                "kukkeineen",
            ],
            # tyttö (girl) — pp→t gradation: tyttö→tytön
            "tyttö": [
                "tyttö", "tytön", "tyttöä", "tytössä", "tytöstä", "tyttöön",
                "tytöllä", "tytöltä", "tytölle", "tyttönä", "tytöksi", "tytöttä",
                "tytöt", "tyttöjen", "tyttöjä", "tytöissä", "tytöistä", "tyttöihin",
                "tytöillä", "tytöiltä", "tytöille", "tyttöinä", "tytöiksi", "tyttöitta",
                "tyttöineen",
            ],
            # poika (boy) — k→j gradation: poika→pojan
            "poika": [
                "poika", "pojan", "poikaa", "pojassa", "pojasta", "poikaan",
                "pojalla", "pojalta", "pojalle", "poikana", "pojaksi", "poikatta",
                "pojat", "poikien", "poikia", "pojissa", "pojista", "poikiin",
                "pojilla", "pojilta", "pojille", "poikina", "pojiksi", "poikitta",
                "poikeineen",
            ],
            # ── High-frequency adjectives ────────────────────────────────
            # vanha (old) — regular Type 1 a-adjective
            "vanha": [
                "vanha", "vanhan", "vanhaa", "vanhassa", "vanhasta", "vanhaan",
                "vanhalla", "vanhalta", "vanhalle", "vanhana", "vanhaksi", "vanhatta",
                "vanhat", "vanhojen", "vanhoja", "vanhoissa", "vanhoista", "vanhoihin",
                "vanhoilla", "vanhoilta", "vanhoille", "vanhoina", "vanhoiksi", "vanhoitta",
                "vanhoineen",
            ],
            # suuri (large/great) — Type 27, suuri→suure-
            "suuri": [
                "suuri", "suuren", "suurta", "suuressa", "suuresta", "suureen",
                "suurella", "suurelta", "suurelle", "suurena", "suureksi", "suuretta",
                "suuret", "suurten", "suuria", "suurissa", "suurista", "suuriin",
                "suurilla", "suurilta", "suurille", "suurina", "suuriksi", "suuritta",
                "suurineen",
            ],
            # uusi (new) — Type 27, uusi→uude- (s→d alternation)
            "uusi": [
                "uusi", "uuden", "uutta", "uudessa", "uudesta", "uuteen",
                "uudella", "uudelta", "uudelle", "uutena", "uudeksi", "uudetta",
                "uudet", "uusien", "uusia", "uusissa", "uusista", "uusiin",
                "uusilla", "uusilta", "uusille", "uusina", "uusiksi", "uusitta",
                "uusineen",
            ],
            # pieni (small) — Type 26, pieni→piene-
            "pieni": [
                "pieni", "pienen", "pientä", "pienessä", "pienestä", "pieneen",
                "pienellä", "pieneltä", "pienelle", "pienenä", "pieneksi", "pienettä",
                "pienet", "pienien", "pieniä", "pienissä", "pienistä", "pieniin",
                "pienillä", "pieniltä", "pienille", "pieninä", "pieniksi", "pienittä",
                "pieneineen",
            ],
            # pitkä (long/tall) — regular Type 1 ä-adjective
            "pitkä": [
                "pitkä", "pitkän", "pitkää", "pitkässä", "pitkästä", "pitkään",
                "pitkällä", "pitkältä", "pitkälle", "pitkänä", "pitkäksi", "pitkättä",
                "pitkät", "pitkien", "pitkiä", "pitkissä", "pitkistä", "pitkiin",
                "pitkillä", "pitkiltä", "pitkille", "pitkinä", "pitkiksi", "pitkittä",
                "pitkineen",
            ],
            # lyhyt (short) — Type 41, lyhyt→lyhye-
            "lyhyt": [
                "lyhyt", "lyhyen", "lyhyttä", "lyhyessä", "lyhyestä", "lyhyeen",
                "lyhyellä", "lyhyeltä", "lyhyelle", "lyhyenä", "lyhyeksi", "lyhyettä",
                "lyhyet", "lyhyiden", "lyhyitä", "lyhyissä", "lyhyistä", "lyhyihin",
                "lyhyillä", "lyhyiltä", "lyhyille", "lyhyinä", "lyhyiksi", "lyhyittä",
                "lyhyineen",
            ],
            # kaunis (beautiful) — Type 41, kaunis→kauniis-
            "kaunis": [
                "kaunis", "kauniin", "kaunista", "kauniissa", "kauniista", "kauniiseen",
                "kauniilla", "kauniilta", "kauniille", "kauniina", "kauniiksi", "kaunitta",
                "kauniit", "kauniiden", "kauniita", "kauniissa", "kauniista", "kauniisiin",
                "kauniilla", "kauniilta", "kauniille", "kauniina", "kauniiksi", "kaunitta",
                "kauniineen",
            ],
            # vahva (strong) — regular Type 1 a-adjective
            "vahva": [
                "vahva", "vahvan", "vahvaa", "vahvassa", "vahvasta", "vahvaan",
                "vahvalla", "vahvalta", "vahvalle", "vahvana", "vahvaksi", "vahvatta",
                "vahvat", "vahvojen", "vahvoja", "vahvoissa", "vahvoista", "vahvoihin",
                "vahvoilla", "vahvoilta", "vahvoille", "vahvoina", "vahvoiksi", "vahvoitta",
                "vahvoineen",
            ],
            # ── Verbs — see extended paradigms below ─────────────────────
            "syödä": [
                "syö", "syön", "syöt", "syömme", "syötte", "syövät",
                "söin", "söit", "söi", "söimme", "söitte", "söivät", "syödä",
                # passive
                "syödään", "syötiin", "syöty",
                # 2nd infinitive
                "syödessä", "syödessään",
                # 3rd infinitive
                "syömässä", "syömästä", "syömään", "syömällä", "syömättä",
                # participles
                "syövä", "syövän", "syönyt", "syöneen",
            ],
            "sanoa": [
                "sanon", "sanot", "sanoo", "sanomme", "sanotte", "sanovat",
                "sanoin", "sanoit", "sanoi", "sanoimme", "sanoitte", "sanoivat",
                "sanoa", "sano", "sanoisin",
            ],
            "antaa": [
                "annan", "annat", "antaa", "annamme", "annatte", "antavat",
                "annoin", "annoit", "antoi", "annoimme", "annoitte", "antoivat",
                "antaa", "anna", "antaisin",
            ],
            "lukea": [
                "luen", "luet", "lukee", "luemme", "luette", "lukevat",
                "luin", "luit", "luki", "luimme", "luitte", "lukivat",
                "lukea", "lue",
            ],
            "kirjoittaa": [
                "kirjoitan", "kirjoitat", "kirjoittaa", "kirjoitamme",
                "kirjoitatte", "kirjoittavat", "kirjoitin", "kirjoitit",
                "kirjoitti", "kirjoitimme", "kirjoititte", "kirjoittivat",
                "kirjoittaa", "kirjoita",
            ],
            "nähdä": [
                "näen", "näet", "näkee", "näemme", "näette", "näkevät",
                "näin", "näit", "näki", "näimme", "näitte", "näkivät",
                "nähdä", "näe",
            ],
            "kuulla": [
                "kuulen", "kuulet", "kuulee", "kuulemme", "kuulette", "kuulevat",
                "kuulin", "kuulit", "kuuli", "kuulimme", "kuulitte", "kuulivat",
                "kuulla", "kuule",
            ],
            "tietää": [
                "tiedän", "tiedät", "tietää", "tiedämme", "tiedätte", "tietävät",
                "tiesin", "tiesit", "tiesi", "tiesimme", "tiesitte", "tiesivät",
                "tietää", "tiedä",
            ],
            "haluta": [
                "haluan", "haluat", "haluaa", "haluamme", "haluatte", "haluavat",
                "halusin", "halusit", "halusi", "halusimme", "halusitte",
                "halusivat", "haluta",
            ],
            "voida": [
                "voin", "voit", "voi", "voimme", "voitte", "voivat",
                "voisin", "voisit", "voisi", "voisimme", "voisitte", "voisivat",
                "voida",
            ],
            "pitää": [
                "pidän", "pidät", "pitää", "pidämme", "pidätte", "pitävät",
                "pidin", "pidit", "piti", "pidimme", "piditte", "pitivät",
                "pitää", "pidä",
            ],
            "ajatella": [
                "ajattelen", "ajattelet", "ajattelee", "ajattelemme",
                "ajattelette", "ajattelevat", "ajattelin", "ajattelit",
                "ajatteli", "ajattelimme", "ajattelitte", "ajattelivat",
                "ajatella", "ajattele",
            ],
            # ── More common verbs ─────────────────────────────────────────
            "rakastaa": [
                "rakastan", "rakastat", "rakastaa", "rakastamme", "rakastatte",
                "rakastavat", "rakastin", "rakastit", "rakasti", "rakastimme",
                "rakastitte", "rakastivat", "rakastaa", "rakasta",
                "rakastetaan", "rakastettiin",
                "rakastaessa", "rakastettava", "rakastettu",
            ],
            "auttaa": [
                "autan", "autat", "auttaa", "autamme", "autatte", "auttavat",
                "autoin", "autoit", "auttoi", "autoimme", "autoitte", "auttoivat",
                "auttaa", "auta",
                "autetaan", "autettiin",
            ],
            "oppia": [
                "opin", "opit", "oppii", "opimme", "opitte", "oppivat",
                "opin", "opit", "oppi", "opimme", "opitte", "oppivat",
                "oppia", "opi",
                "opitaan", "opittiin",
                "oppiessa", "oppinut", "opittu",
            ],
            "löytää": [
                "löydän", "löydät", "löytää", "löydämme", "löydätte", "löytävät",
                "löysin", "löysit", "löysi", "löysimme", "löysitte", "löysivät",
                "löytää", "löydä",
                "löydetään", "löydettiin",
                "löytäessä", "löytänyt", "löydetty",
            ],
            "ottaa": [
                "otan", "otat", "ottaa", "otamme", "otatte", "ottavat",
                "otin", "otit", "otti", "otimme", "otitte", "ottivat",
                "ottaa", "ota",
                "otetaan", "otettiin",
                "ottaessa", "ottanut", "otettu",
            ],
            "antaa": [
                "annan", "annat", "antaa", "annamme", "annatte", "antavat",
                "annoin", "annoit", "antoi", "annoimme", "annoitte", "antoivat",
                "antaa", "anna",
                "annetaan", "annettiin",
                "antaessa", "antanut", "annettu",
            ],
            "tarvita": [
                "tarvitsen", "tarvitset", "tarvitsee", "tarvitsemme",
                "tarvitsette", "tarvitsevat", "tarvitsin", "tarvitsit",
                "tarvitsi", "tarvitsimme", "tarvitsitte", "tarvitsivat",
                "tarvita", "tarvitse",
                "tarvitaan", "tarvittiin",
            ],
            "ajaa": [
                "ajan", "ajat", "ajaa", "ajamme", "ajatte", "ajavat",
                "ajoin", "ajoit", "ajoi", "ajoimme", "ajoitte", "ajoivat",
                "ajaa", "aja",
                "ajetaan", "ajettiin",
                "ajaessa", "ajanut", "ajettu",
            ],
            "katsoa": [
                "katson", "katsot", "katsoo", "katsomme", "katsotte", "katsovat",
                "katsoin", "katsoit", "katsoi", "katsoimme", "katsoitte", "katsoivat",
                "katsoa", "katso",
                "katsotaan", "katsottiin",
                "katsoessa", "katsonut", "katsottu",
            ],
            "tuntea": [
                "tunnen", "tunnet", "tuntee", "tunnemme", "tunnette", "tuntevat",
                "tunsin", "tunsit", "tunsi", "tunsimme", "tunsitte", "tunsivat",
                "tuntea", "tunne",
                "tunnetaan", "tunnettiin",
                "tuntessa", "tuntenut", "tunnettu",
            ],
            "elää": [
                "elän", "elät", "elää", "elämme", "elätte", "elävät",
                "elin", "elit", "eli", "elimme", "elitte", "elivät",
                "elää", "elä",
                "eletään", "elettiin",
                "eläessä", "elänyt", "eletty",
            ],
            "puhua": [
                "puhun", "puhut", "puhuu", "puhumme", "puhutte", "puhuvat",
                "puhuin", "puhuit", "puhui", "puhuimme", "puhuitte", "puhuivat",
                "puhua", "puhu",
                "puhutaan", "puhuttiin",
                "puhuessa", "puhunut", "puhuttu",
            ],
            "kysyä": [
                "kysyn", "kysyt", "kysyy", "kysymme", "kysytte", "kysyvät",
                "kysyin", "kysyit", "kysyi", "kysyimme", "kysyitte", "kysyivät",
                "kysyä", "kysy",
                "kysytään", "kysyttiin",
                "kysynyt", "kysytty",
            ],
            "vastata": [
                "vastaan", "vastaat", "vastaa", "vastaamme", "vastaatte", "vastaavat",
                "vastasin", "vastasit", "vastasi", "vastasimme", "vastasitte", "vastasivat",
                "vastata", "vastaa",
                "vastataan", "vastattiin",
            ],
            "juosta": [
                "juoksen", "juokset", "juoksee", "juoksemme", "juoksette", "juoksevat",
                "juoksin", "juoksit", "juoksi", "juoksimme", "juoksitte", "juoksivat",
                "juosta", "juokse",
                "juostaan", "juostiin",
                "juosten", "juossut",
            ],
            "tanssia": [
                "tanssin", "tanssit", "tanssii", "tanssimme", "tanssitte", "tanssivat",
                "tanssin", "tanssit", "tanssi", "tanssimme", "tanssitte", "tanssivat",
                "tanssia", "tanssi",
                "tanssitaan", "tanssittiin",
            ],
            "laulaa": [
                "laulan", "laulat", "laulaa", "laulamme", "laulatte", "laulavat",
                "lauloin", "lauloit", "lauloi", "lauloimme", "lauloitte", "lauloivat",
                "laulaa", "laula",
                "lauletaan", "laulettiin",
                "laulanut", "laulettu",
            ],
            # ── Personal pronouns (including accusative -t forms) ────────
            # Finnish pronouns have a morphologically distinct accusative
            # (minut, sinut, hänet…) unlike nouns where accusative = genitive.
            "minä": [
                "minä", "minun", "minua", "minussa", "minusta", "minuun",
                "minulla", "minulta", "minulle", "minuna", "minuksi", "minut",
            ],
            "sinä": [
                "sinä", "sinun", "sinua", "sinussa", "sinusta", "sinuun",
                "sinulla", "sinulta", "sinulle", "sinuna", "sinuksi", "sinut",
            ],
            "hän": [
                "hän", "hänen", "häntä", "hänessä", "hänestä", "häneen",
                "hänellä", "häneltä", "hänelle", "hänenä", "häneksi", "hänet",
            ],
            "me": [
                "me", "meidän", "meitä", "meissä", "meistä", "meihin",
                "meillä", "meiltä", "meille", "meinä", "meiksi", "meidät",
            ],
            "te": [
                "te", "teidän", "teitä", "teissä", "teistä", "teihin",
                "teillä", "teiltä", "teille", "teinä", "teiksi", "teidät",
            ],
            "he": [
                "he", "heidän", "heitä", "heissä", "heistä", "heihin",
                "heillä", "heiltä", "heille", "heinä", "heiksi", "heidät",
            ],
            # ── Negation verb ei (kieltosana) ────────────────────────────
            "ei": [
                "ei", "en", "et", "emme", "ette", "eivät",
                "en ole", "ei ole",  # common negation phrases stored as tokens
            ],
            # ── Extended verb paradigms ──────────────────────────────────
            # tehdä — includes passive, all infinitives, participles
            # (Voikko handles these natively; rule-based covers common forms)
            "tehdä": [
                # indicative
                "teen", "teet", "tekee", "teemme", "teette", "tekevät",
                "tein", "teit", "teki", "teimme", "teitte", "tekivät",
                "tehdä", "tee", "tekisin", "tekisi",
                # potentiaali
                "tehnee", "tehnen", "tehnet", "tehnemme", "tehnette", "tehnevät",
                # passive (tehdä — present: tehdään, past: tehtiin)
                "tehdään", "tehtiin",
                # 2nd infinitive: tehdessä (while doing)
                "tehdessä", "tehdessään",
                # 3rd infinitive: tekemässä (in the process of), tekemästä, tekemään
                "tekemässä", "tekemästä", "tekemään", "tekemällä", "tekemättä",
                # 4th infinitive verbal noun: tekeminen
                "tekeminen", "tekemisen", "tekemistä",
                # active present participle: tekevä (the one who does)
                "tekevä", "tekevän", "tekevää", "tekevässä",
                # active past participle: tehnyt (the one who did), tehneenä
                "tehnyt", "tehneen", "tehnyttä", "tehneinä",
                # passive past participle: tehty (done)
                "tehty", "tehdyn", "tehdyssä",
            ],
            "mennä": [
                "menen", "menet", "menee", "menemme", "menette", "menevät",
                "meni", "menin", "menit", "menimme", "menitte", "menivät",
                "mennä", "mene", "menisin", "menisi",
                # potentiaali
                "mennee", "mennen", "mennet", "mennemme", "mennette", "mennevät",
                # passive
                "mennään", "mentiin", "menty",
                # 2nd infinitive
                "mennessä", "mennessään",
                # 3rd infinitive
                "menemässä", "menemästä", "menemään", "menemällä", "menemättä",
                # 4th infinitive
                "meneminen", "menemisen", "menemistä",
                # participles
                "menevä", "menevän", "mennyt", "menneen",
                # 5th infinitive (very rare: about to go)
                "menemäisillään", "menemäisilläni",
            ],
            "tulla": [
                "tulen", "tulet", "tulee", "tulemme", "tulette", "tulevat",
                "tuli", "tulin", "tulit", "tulimme", "tulitte", "tulivat",
                "tulla", "tule", "tulisin", "tulisi",
                # potentiaali
                "tullee", "tullen", "tullet", "tullemme", "tullette", "tullevat",
                # passive
                "tullaan", "tultiin", "tultu",
                # 2nd infinitive
                "tullessa", "tullessaan",
                # 3rd infinitive
                "tulemassa", "tulemasta", "tulemaan", "tulemalla", "tulematta",
                # 4th infinitive
                "tuleminen", "tulemisen", "tulemista",
                # participles
                "tuleva", "tulevan", "tullut", "tulleen",
            ],
            "olla": [
                "on", "olen", "olet", "olemme", "olette", "ovat",
                "oli", "olin", "olit", "olimme", "olitte", "olivat",
                "olla", "ole", "olisi", "olisit", "olisimme", "olisitte",
                "olisivat", "ollut", "olleet", "oltiin", "oltaisiin",
                # potentiaali
                "lienee", "lienen", "lienet", "lienemme", "lienette", "lienevät",
                "lienisi", "lienisit", "lienisimme", "lienisitte", "lienisivät",
                "liekkö", "liekö",
                # passive
                "ollaan", "oltiin",
                # 2nd infinitive
                "ollessa", "ollessaan",
                # 3rd infinitive
                "olemassa", "olemasta", "olemaan", "olemalla", "olematta",
                # active present participle: oleva
                "oleva", "olevan", "olevaa", "olevassa",
                # active past participle: ollut
                "ollut", "olleen",
            ],
            "saada": [
                "saan", "saat", "saa", "saamme", "saatte", "saavat",
                "sain", "sait", "sai", "saimme", "saitte", "saivat",
                "saada", "saa",
                # potentiaali
                "saanee", "saanen", "saanet", "saanemme", "saanette", "saanevat",
                # passive
                "saadaan", "saatiin", "saatu",
                # 2nd infinitive
                "saadessa", "saadessaan",
                # 3rd infinitive
                "saamassa", "saamasta", "saamaan", "saamalla", "saamatta",
                # participles
                "saava", "saavan", "saanut", "saaneen",
            ],
            # ── Prolative examples ───────────────────────────────────────
            # Prolative (-tse suffix) — by route/means: postitse, meritse, maitse
            "posti": [
                "posti", "postin", "postia", "postissa", "postista", "postiin",
                "postilla", "postilta", "postille", "postina", "postiksi",
                "postitse",
            ],
            "meri": [
                "meri", "meren", "merta", "meressä", "merestä", "mereen",
                "merellä", "mereltä", "merelle", "merenä", "mereksi",
                "meritse",
            ],
            "laiva": [
                "laiva", "laivan", "laivaa", "laivassa", "laivasta", "laivaan",
                "laivalla", "laivalta", "laivalle", "laivana", "laivaksi",
                "laivatta", "laivat", "laivojen", "laivoja",
                "laivitse",
            ],
        }

    # ------------------------------------------------------------------
    # Voikko analysis
    # ------------------------------------------------------------------

    def _voikko_analyze(self, word: str) -> Optional[Dict]:
        if not self.voikko:
            return None
        try:
            analyses = self.voikko.analyze(word)
        except Exception as e:
            logger.debug("Voikko error for %r: %s", word, e)
            return None
        if not analyses:
            return None

        a = _best_analysis(analyses)
        return {
            "lemma":    a.get("BASEFORM", word),
            "pos":      _translate_pos(a.get("CLASS") or a.get("class")),
            "case":     _translate_case(a.get("SIJAMUOTO") or a.get("sijamuoto")),
            "number":   _translate_number(a.get("NUMBER") or a.get("number")),
            "mood":     a.get("MOOD"),
            "tense":    a.get("TENSE"),
            "person":   a.get("PERSON"),
            "negative": a.get("NEGATIVE"),
        }

    # ------------------------------------------------------------------
    # Rule-based fallback
    # ------------------------------------------------------------------

    def _is_verb_lemma(self, lemma: str) -> bool:
        # Personal pronouns like "minä" end in "-nä" but are not verbs
        if lemma in self._pronoun_lemmas:
            return False
        return lemma.endswith(self._verb_endings)

    def _detect_case_and_number(self, word: str, lemma: str) -> Tuple[str, str]:
        """
        Infer grammatical case and number from an inflected noun/adjective form.

        Strategy:
          1. Exact match with lemma → Nominative Singular.
          2. Ends with -t (and len > 2) → Nominative Plural (Finnish -t plural).
          3. Scan ordered case_patterns for the longest matching suffix.
          4. Fall back to Nominative Singular.
        """
        w = word.lower()
        l = lemma.lower()
        if w == l:
            return "Nominative", "Singular"
        # Nominative plural: standard -t suffix for nouns and adjectives
        if w.endswith("t") and len(w) > 2:
            return "Nominative", "Plural"
        # Suffix-based detection — longer suffixes checked first inside each tuple
        for case, number, suffixes in self.case_patterns:
            for suf in sorted(suffixes, key=len, reverse=True):
                if w.endswith(suf) and len(w) > len(suf):
                    return case, number
        return "Nominative", "Singular"

    def _rule_based_lemmatize(self, word: str) -> str:
        w = word.lower()

        # -nen words (Type 38): ihminen, nainen — stem changes to -ise- / -is-
        if (w.endswith("sta") or w.endswith("stä")) and len(w) > 4:
            stem = w[:-2]
            if stem.endswith("s"):
                return stem[:-1] + "nen"
        for _c, _n, suffixes in self.case_patterns:
            for suf in suffixes:
                if w.endswith(suf) and len(w) > len(suf) + 2:
                    stem = w[:-len(suf)]
                    if stem.endswith("ise"):
                        return stem[:-3] + "nen"
                    if stem.endswith("se"):
                        return stem[:-2] + "nen"

        # Nominative plural -t
        if w.endswith("t") and len(w) > 3:
            base = w[:-1]
            if base.endswith(("aa", "ää", "ee", "ii", "oo", "öö", "uu", "yy")):
                base = base[:-1]
            return base

        # Regular case suffix stripping
        for case, number, suffixes in self.case_patterns:
            for suf in sorted(suffixes, key=len, reverse=True):
                if w.endswith(suf) and len(w) > len(suf) + 2:
                    stem = w[:-len(suf)]
                    # Illative VVn: restore the stem-final vowel that was doubled
                    if case == "Illative" and number == "Singular" and len(suf) == 3:
                        # e.g. "kissaan"→stem "kiss", suf "aan"→restore "a"
                        if suf in ("aan", "ään", "oon", "een", "iin", "uun", "yyn"):
                            return stem + suf[0]
                    if number == "Plural":
                        if stem.endswith("i") and len(stem) > 2 and stem[-2] not in "aeiouyäö":
                            stem = stem[:-1]
                        if stem.endswith("j"):
                            stem = stem[:-1]
                        if stem.endswith("o") and not stem.endswith("oo"):
                            stem = stem[:-1] + "a"
                        elif stem.endswith("ö") and not stem.endswith("öö"):
                            stem = stem[:-1] + "ä"
                    if stem.endswith(("aa", "ää", "ee", "ii", "oo", "öö", "uu", "yy")):
                        stem = stem[:-1]
                    return stem

        return w

    def _rule_based_analyze(self, word: str) -> Dict:
        w = word.lower()
        for lemma, forms in self.known_words.items():
            if w in forms:
                is_verb = self._is_verb_lemma(lemma)
                pos = "VERB" if is_verb else "NOUN"
                if is_verb:
                    # Verbs don't carry nominal case
                    return {"lemma": lemma, "pos": pos,
                            "case": "Unknown", "number": "Singular"}
                case, number = self._detect_case_and_number(w, lemma)
                return {"lemma": lemma, "pos": pos,
                        "case": case, "number": number}
        lemma = self._rule_based_lemmatize(w)
        pos = "ADV" if w.endswith("sti") else (
              "VERB" if w.endswith(("da", "dä", "ta", "tä", "la", "lä", "ra", "rä")) else "NOUN")
        # Try case detection for the suffix-lemmatized word
        case, number = self._detect_case_and_number(w, lemma)
        if case == "Nominative" and w != lemma:
            case = "Unknown"
        return {"lemma": lemma, "pos": pos, "case": case, "number": number}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _lemmatize_word(self, word: str, include_morphology: bool) -> WordLemma:
        result = self._voikko_analyze(word)
        if result:
            morph = None
            if include_morphology:
                morph = {"case": result["case"], "number": result["number"]}
                for f in ("mood", "tense", "person", "negative"):
                    if result.get(f):
                        morph[f] = result[f]
            return WordLemma(original=word, lemma=result["lemma"], pos=result["pos"],
                             morphology=morph)

        analysis = self._rule_based_analyze(word)
        morph = ({"case": analysis["case"], "number": analysis["number"]}
                 if include_morphology else None)
        return WordLemma(original=word, lemma=analysis["lemma"], pos=analysis["pos"],
                         morphology=morph)

    def lemmatize(self, text: str, include_morphology: bool = True) -> LemmatizationResponse:
        """
        Lemmatize Finnish *text*.

        When libvoikko is installed each token receives full morphological
        detail (all 15 cases, consonant gradation, verb conjugation, clitics).
        Without voikko the rule-based fallback covers common inflections and
        rare literary forms including Potentiaali mood and Abessive/Comitative
        cases.
        """
        logger.info("Lemmatizing %d chars (voikko=%s)", len(text), bool(self.voikko))
        tokens = re.findall(r"[\w\-]+", text, re.UNICODE)
        lemmas = [self._lemmatize_word(t, include_morphology) for t in tokens if t.strip()]
        return LemmatizationResponse(text=text, lemmas=lemmas, word_count=len(lemmas))

    def __del__(self) -> None:
        if self.voikko:
            try:
                self.voikko.terminate()
            except Exception:
                pass

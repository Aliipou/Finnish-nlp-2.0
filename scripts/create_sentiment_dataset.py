"""
Create Finnish Sentiment Analysis Dataset
600 samples: 200 positive, 200 negative, 200 neutral
"""
import csv
import random
from typing import List, Dict

# Positive sentiment templates
POSITIVE_TEMPLATES = [
    ("Olen {adverb} tyytyväinen {target}.", "positive", 0.85),
    ("Tämä on {adj} kokemus!", "positive", 0.90),
    ("Rakastan {target}. {Reason}.", "positive", 0.95),
    ("{Adj} {thing}! Suosittelen lämpimästi.", "positive", 0.88),
    ("Olen {adverb} iloinen {target}.", "positive", 0.87),
    ("Paras {thing} ikinä! {Reason}.", "positive", 0.92),
    ("{Thing} ylitti {thing2}.", "positive", 0.89),
    ("En voi kuin {verb} {target}.", "positive", 0.86),
    ("Erinomainen {thing}. {Reason}.", "positive", 0.90),
    ("{Adj}, {adj2} ja {adj3} - täydellinen yhdistelmä!", "positive", 0.91),
    ("Tämä muutti {thing}. Kiitos!", "positive", 0.88),
    ("Olen vaikuttunut {target}.", "positive", 0.84),
    ("Joka kerta kun {verb}, olen {adj}.", "positive", 0.87),
    ("{Adj} laatu ja {adj2} palvelu!", "positive", 0.90),
    ("Suosittelen ehdottomasti {person}.", "positive", 0.85),
]

# Negative sentiment templates
NEGATIVE_TEMPLATES = [
    ("Olen {adverb} pettynyt {target}.", "negative", 0.15),
    ("Tämä on {adj} kokemus.", "negative", 0.12),
    ("Inhoan {target}. {Reason}.", "negative", 0.08),
    ("{Adj} {thing}. En suosittele.", "negative", 0.14),
    ("Olen {adverb} tyytymätön {target}.", "negative", 0.13),
    ("Huonoin {thing} ikinä. {Reason}.", "negative", 0.10),
    ("{Thing} ei vastannut {thing2}.", "negative", 0.16),
    ("En voi kuin {verb} {target}.", "negative", 0.14),
    ("Kauhea {thing}. {Reason}.", "negative", 0.11),
    ("{Adj}, {adj2} ja {adj3} - täydellinen katastrofi.", "negative", 0.09),
    ("Tämä pilasi {thing}. Ikävää.", "negative", 0.12),
    ("Olen järkyttynyt {target}.", "negative", 0.10),
    ("Joka kerta kun {verb}, olen {adj}.", "negative", 0.13),
    ("{Adj} laatu ja {adj2} palvelu.", "negative", 0.14),
    ("En suosittele {person}.", "negative", 0.15),
]

# Neutral sentiment templates
NEUTRAL_TEMPLATES = [
    ("Palvelu oli {adj}.", "neutral", 0.50),
    ("Tuote vastasi {thing}.", "neutral", 0.52),
    ("{Thing} on {adj}.", "neutral", 0.51),
    ("Kokemus oli {adj}.", "neutral", 0.50),
    ("Hinta on {adj} {target}.", "neutral", 0.49),
    ("{Thing} toimii {adverb}.", "neutral", 0.52),
    ("En osaa sanoa {thing}.", "neutral", 0.50),
    ("Tämä riippuu {target}.", "neutral", 0.51),
    ("{Adj} vaihtoehto {person}.", "neutral", 0.50),
    ("Normaali {thing} ilman {thing2}.", "neutral", 0.51),
    ("{Thing} on keskivertoa.", "neutral", 0.50),
    ("Asiakaspalvelu vastasi {thing}.", "neutral", 0.52),
    ("Odotettavissa oleva {thing}.", "neutral", 0.51),
    ("Ei {adj} eikä {adj2}.", "neutral", 0.50),
    ("Standardin mukainen {thing}.", "neutral", 0.51),
]

# Vocabularies for positive sentiment
POSITIVE_VOCAB = {
    "adverb": ["erittäin", "todella", "äärimmäisen", "täysin", "hyvin"],
    "target": ["tähän tuotteeseen", "palveluun", "ostokseen", "kokemukseen", "tulokseen"],
    "adj": ["loistava", "upea", "erinomainen", "fantastinen", "mahtava", "ihana",
            "täydellinen", "huikea", "uskomaton", "vaikuttava", "iloinen", "onnellinen"],
    "Reason": ["Ylitti odotukset", "Toimii täydellisesti", "Paras valinta", "Kannattaa ehdottomasti"],
    "Adj": ["Loistava", "Upea", "Erinomainen", "Fantastinen", "Mahtava"],
    "thing": ["tuote", "palvelu", "kokemus", "ratkaisu", "valinta", "investointi"],
    "Thing": ["Tuote", "Palvelu", "Kokemus", "Ratkaisu", "Valinta"],
    "thing2": ["odotukseni", "toiveeni", "vaatimukseni", "standardini"],
    "verb": ["kehua", "suositella", "ihailla", "arvostaa", "ylistää"],
    "adj2": ["nopea", "tehokas", "ystävällinen", "ammattitaitoinen", "laadukas"],
    "adj3": ["edullinen", "kestävä", "luotettava", "helppokäyttöinen", "toimiva"],
    "person": ["kaikille", "jokaiselle", "ystävilleni", "kaikille tuttuilleni"],
}

# Vocabularies for negative sentiment
NEGATIVE_VOCAB = {
    "adverb": ["erittäin", "todella", "äärimmäisen", "täysin", "hyvin"],
    "target": ["tähän tuotteeseen", "palveluun", "ostokseen", "kokemukseen", "tulokseen"],
    "adj": ["huono", "kamala", "kauhea", "hirveä", "pettymys", "surkea",
            "kehno", "riittämätön", "pettynyt", "tyytymätön", "surullinen"],
    "Reason": ["Ei vastannut odotuksia", "Toimi huonosti", "Huono valinta", "En suosittele"],
    "Adj": ["Huono", "Kamala", "Kauhea", "Hirveä", "Surkea"],
    "thing": ["tuote", "palvelu", "kokemus", "ratkaisu", "valinta", "investointi"],
    "Thing": ["Tuote", "Palvelu", "Kokemus", "Ratkaisu", "Valinta"],
    "thing2": ["odotuksiani", "toiveitani", "vaatimuksiani", "standardejani"],
    "verb": ["kritisoida", "valittaa", "pahoitella", "harmitella"],
    "adj2": ["hidas", "tehoton", "epäystävällinen", "epäpätevä", "huonolaatuinen"],
    "adj3": ["kallis", "heikko", "epäluotettava", "vaikea käyttää", "rikki"],
    "person": ["kenellekään", "muille", "ystävilleni", "tuttavilleni"],
}

# Vocabularies for neutral sentiment
NEUTRAL_VOCAB = {
    "adj": ["tavallinen", "keskinkertainen", "normaali", "tyypillinen", "standardin mukainen",
            "odotettavissa oleva", "kohtuullinen", "riittävä", "perus"],
    "thing": ["kuvausta", "hintaa", "laatua", "palvelua", "lopputulosta"],
    "Thing": ["Tuote", "Palvelu", "Hinta", "Laatu", "Toimitus"],
    "adverb": ["normaalisti", "kohtuullisesti", "riittävästi", "tyydyttävästi"],
    "target": ["vertailussa", "kilpailijoihin", "hintaan nähden", "kategoriassaan"],
    "Adj": ["Tavallinen", "Normaali", "Perus", "Kohtuullinen"],
    "person": ["useimmille", "joillekin", "tietyille käyttäjille", "monille"],
    "thing2": ["yllätyksiä", "erikoisuuksia", "lisäominaisuuksia"],
}

def generate_sample(template: str, sentiment: str, score: float,
                   vocab: dict, sample_id: int) -> Dict:
    """Generate a single sentiment sample"""
    text = template

    # Replace all placeholders
    for key, values in vocab.items():
        if "{" + key + "}" in text:
            text = text.replace("{" + key + "}", random.choice(values))

    # Add slight variation to score
    score_varied = round(score + random.uniform(-0.05, 0.05), 2)
    score_varied = max(0.0, min(1.0, score_varied))  # Clamp to [0, 1]

    return {
        "id": sample_id,
        "text": text,
        "sentiment": sentiment,
        "score": score_varied,
        "domain": random.choice(["product_review", "service_review", "social_media", "feedback"])
    }

def create_sentiment_dataset(num_samples: int = 600) -> List[Dict]:
    """Create balanced sentiment dataset"""
    samples = []
    sample_id = 1

    samples_per_class = num_samples // 3  # 200 each

    print(f"Generating {num_samples} sentiment samples:")
    print(f"  - Positive: {samples_per_class}")
    print(f"  - Negative: {samples_per_class}")
    print(f"  - Neutral: {samples_per_class}")

    # Generate positive samples
    for _ in range(samples_per_class):
        template, sentiment, score = random.choice(POSITIVE_TEMPLATES)
        sample = generate_sample(template, sentiment, score, POSITIVE_VOCAB, sample_id)
        samples.append(sample)
        sample_id += 1

    # Generate negative samples
    for _ in range(samples_per_class):
        template, sentiment, score = random.choice(NEGATIVE_TEMPLATES)
        sample = generate_sample(template, sentiment, score, NEGATIVE_VOCAB, sample_id)
        samples.append(sample)
        sample_id += 1

    # Generate neutral samples
    for _ in range(samples_per_class):
        template, sentiment, score = random.choice(NEUTRAL_TEMPLATES)
        sample = generate_sample(template, sentiment, score, NEUTRAL_VOCAB, sample_id)
        samples.append(sample)
        sample_id += 1

    # Shuffle
    random.shuffle(samples)

    return samples

def save_dataset(samples: List[Dict], output_file: str):
    """Save sentiment dataset to CSV"""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'text', 'sentiment', 'score', 'domain'])
        writer.writeheader()
        writer.writerows(samples)

    print(f"\nDataset saved to {output_file}")
    print(f"Total samples: {len(samples)}")

    # Statistics
    positive = sum(1 for s in samples if s['sentiment'] == 'positive')
    negative = sum(1 for s in samples if s['sentiment'] == 'negative')
    neutral = sum(1 for s in samples if s['sentiment'] == 'neutral')

    print(f"  - Positive: {positive} ({positive/len(samples)*100:.1f}%)")
    print(f"  - Negative: {negative} ({negative/len(samples)*100:.1f}%)")
    print(f"  - Neutral: {neutral} ({neutral/len(samples)*100:.1f}%)")

if __name__ == "__main__":
    random.seed(42)

    print("=" * 60)
    print("Finnish Sentiment Dataset Creation")
    print("=" * 60)

    # Create dataset directory if it doesn't exist
    import os
    os.makedirs("data/datasets/finnish_sentiment_corpus", exist_ok=True)

    samples = create_sentiment_dataset(num_samples=600)
    save_dataset(samples, "data/datasets/finnish_sentiment_corpus/finnish_sentiment_dataset.csv")

    print("\nDataset creation complete!")

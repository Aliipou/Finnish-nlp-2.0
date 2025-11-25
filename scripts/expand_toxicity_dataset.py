"""
Expand Finnish Toxicity Dataset from 50 to 1200 samples
Uses template-based generation with variations
"""
import csv
import random
from typing import List, Dict, Tuple

# Template patterns for non-toxic content
NON_TOXIC_TEMPLATES = [
    # Greetings and positive messages
    ("Hyvää {time}! Toivottavasti sinulla on {adj} päivä.", 0, 0.01, "none", "social_media"),
    ("Kiitos {amount} {thing}. Arvostan sitä {adverb}.", 0, 0.02, "none", "social_media"),
    ("{Thing} on todella {adj}. Opin paljon.", 0, 0.03, "none", "forum"),
    ("Olen samaa mieltä {person}. Hyvä pointti!", 0, 0.02, "none", "forum"),
    ("{Adj} {time} tänään. Nauti {thing}!", 0, 0.01, "none", "social_media"),

    # Reviews and feedback
    ("Erinomainen {thing}. Kiitos!", 0, 0.01, "none", "review"),
    ("Nopea {thing} ja laadukas {item}.", 0, 0.01, "none", "review"),
    ("{Adj} kokemus alusta loppuun.", 0, 0.01, "none", "review"),
    ("Olen tyytyväinen {thing}.", 0, 0.02, "none", "review"),
    ("Hyvä hinta-laatusuhde {item}.", 0, 0.01, "none", "review"),
    ("Suosittelen tätä {item}.", 0, 0.01, "none", "review"),
    ("Toimii juuri kuten {thing}.", 0, 0.01, "none", "review"),

    # Helpful comments
    ("Kiitos kun jaoit tämän. {Adj}!", 0, 0.02, "none", "forum"),
    ("Tämä auttoi minua {amount}. Kiitos!", 0, 0.02, "none", "forum"),
    ("{Adj} idea! Kokeilen varmasti.", 0, 0.01, "none", "social_media"),
    ("{Adj} työtä! Jatka samaan malliin.", 0, 0.01, "none", "social_media"),

    # Neutral feedback
    ("Olen {adverb} pettynyt tähän {thing}.", 0, 0.15, "none", "review"),
    ("Tämä ei vastannut {thing}.", 0, 0.12, "none", "review"),
    ("{Adj} kokemus valitettavasti.", 0, 0.18, "none", "review"),
    ("En suosittelisi {person}.", 0, 0.22, "none", "review"),
    ("{Thing} oli {adj} ja {adj2}.", 0, 0.25, "none", "review"),
]

# Template patterns for toxic content - profanity (mild)
TOXIC_MILD_PROFANITY = [
    ("Hitto että {verb}.", 1, 0.38, "profanity", "social_media"),
    ("Perkele tämä on {adj}.", 1, 0.45, "profanity", "forum"),
    ("Helvetti {word} miksi tämä ei {verb}!", 1, 0.42, "profanity", "forum"),
    ("Helvetti että {verb} {adverb}!", 1, 0.42, "profanity", "social_media"),
    ("Perkele kun {verb}!", 1, 0.44, "profanity", "social_media"),
    ("Saatana mikä {noun}.", 1, 0.46, "profanity", "social_media"),
    ("Hitto että oli {adj} mutta {verb}.", 1, 0.35, "profanity", "forum"),
]

# Template patterns for toxic content - profanity (severe)
TOXIC_SEVERE_PROFANITY = [
    ("Vittu mikä {noun} {thing}.", 1, 0.85, "profanity", "review"),
    ("Helvetti saatana miksi {thing} ei {verb}!", 1, 0.68, "profanity", "forum"),
    ("Jumalauta mikä {noun}.", 1, 0.62, "profanity", "forum"),
    ("Vittu että {verb} {adverb}.", 1, 0.81, "profanity", "forum"),
    ("Saatanan {adj} {noun}.", 1, 0.72, "profanity", "forum"),
    ("Paska {noun}.", 1, 0.68, "profanity", "forum"),
    ("Tämä on täyttä {noun}.", 1, 0.70, "profanity", "forum"),
]

# Template patterns for toxic content - insults
TOXIC_INSULTS = [
    ("Saatanan {insult}.", 1, 0.82, "insult", "social_media"),
    ("Ole hiljaa senkin {insult}.", 1, 0.71, "insult", "social_media"),
    ("{Insult} {person}.", 1, 0.66, "insult", "review"),
    ("{Adj} {thing}.", 1, 0.48, "insult", "forum"),
    ("{Insult}.", 1, 0.58, "insult", "social_media"),
    ("Helvetin {insult}.", 1, 0.84, "insult", "social_media"),
    ("Senkin {insult}.", 1, 0.74, "insult", "social_media"),
]

# Template patterns for toxic content - hate speech
TOXIC_HATE = [
    ("Vihaan {target}.", 1, 0.78, "hate_speech", "social_media"),
    ("Inhoan {target}.", 1, 0.52, "hate_speech", "review"),
    ("{Target} ovat {adj}.", 1, 0.72, "hate_speech", "forum"),
]

# Vocabulary for templates
VOCABULARIES = {
    "time": ["päivää", "huomenta", "iltaa", "viikonloppua"],
    "adj": ["mahtava", "hieno", "loistava", "upea", "erinomainen", "hyvä", "mukava",
            "vaikea", "haasteellinen", "monimutkainen", "huono", "paha", "kamala",
            "tyhmä", "idiootti", "ääliömäinen"],
    "adj2": ["epäystävällinen", "hidas", "nopea", "ystävällinen", "ammattitaitoinen"],
    "amount": ["paljon", "erittäin", "todella", "suuresti"],
    "thing": ["avustasi", "palveluun", "tuotteeseen", "odotuksiani", "auringosta",
              "luvattu", "asiakaspalvelu", "toimitus"],
    "adverb": ["suuresti", "paljon", "todella", "erittäin"],
    "Thing": ["Tämä", "Artikkeli", "Kirjoitus", "Palvelu", "Tuote"],
    "person": ["kanssasi", "kenellekään", "asiakkaalle", "käyttäjälle"],
    "Adj": ["Kaunis", "Hieno", "Mahtava", "Loistava", "Erinomainen", "Paras",
            "Täydellinen", "Hienoa", "Huono", "Kamala"],
    "item": ["tuote", "tuotetta", "palvelu", "kokemus", "ostokseen"],
    "verb": ["toimi", "onnistui", "meni pieleen", "onnistuu", "toimii", "ärsyttää"],
    "word": ["saatana", "perkele"],
    "noun": ["paska", "sekoilu", "onni", "juttu", "idea", "suunnitelma", "palvelu"],
    "Insult": ["Idiootti", "Ääliö", "Typerys", "Kusipää", "Mulkku"],
    "insult": ["tyhmä", "idiootti", "ääliö", "typerys"],
    "target": ["sinua", "tätä paikkaa", "tätä", "heitä"],
    "Target": ["He", "Nämä ihmiset", "Tämä ryhmä"],
}

def generate_sample(template: str, severity: int, toxicity_score: float,
                   category: str, source: str, sample_id: int) -> Dict:
    """Generate a single sample from template"""
    text = template

    # Replace all placeholders
    for key, values in VOCABULARIES.items():
        if "{" + key + "}" in text:
            text = text.replace("{" + key + "}", random.choice(values))

    return {
        "id": sample_id,
        "text": text,
        "is_toxic": 1 if toxicity_score > 0.30 else 0,
        "severity": severity,
        "toxicity_score": round(toxicity_score + random.uniform(-0.05, 0.05), 2),
        "category": category,
        "source": source
    }

def expand_dataset(target_samples: int = 1200) -> List[Dict]:
    """Expand dataset to target number of samples"""
    samples = []
    sample_id = 1

    # Calculate distribution (60% non-toxic, 40% toxic)
    num_non_toxic = int(target_samples * 0.60)  # 720
    num_toxic = target_samples - num_non_toxic  # 480

    # Further split toxic categories
    num_mild_profanity = int(num_toxic * 0.30)  # 144
    num_severe_profanity = int(num_toxic * 0.35)  # 168
    num_insults = int(num_toxic * 0.25)  # 120
    num_hate = num_toxic - num_mild_profanity - num_severe_profanity - num_insults  # 48

    print(f"Generating {target_samples} samples:")
    print(f"  - Non-toxic: {num_non_toxic}")
    print(f"  - Mild profanity: {num_mild_profanity}")
    print(f"  - Severe profanity: {num_severe_profanity}")
    print(f"  - Insults: {num_insults}")
    print(f"  - Hate speech: {num_hate}")

    # Generate non-toxic samples
    for _ in range(num_non_toxic):
        template, severity, score, category, source = random.choice(NON_TOXIC_TEMPLATES)
        sample = generate_sample(template, severity, score, category, source, sample_id)
        samples.append(sample)
        sample_id += 1

    # Generate mild profanity
    for _ in range(num_mild_profanity):
        template, severity, score, category, source = random.choice(TOXIC_MILD_PROFANITY)
        sample = generate_sample(template, severity, score, category, source, sample_id)
        samples.append(sample)
        sample_id += 1

    # Generate severe profanity
    for _ in range(num_severe_profanity):
        template, severity, score, category, source = random.choice(TOXIC_SEVERE_PROFANITY)
        sample = generate_sample(template, severity, score, category, source, sample_id)
        samples.append(sample)
        sample_id += 1

    # Generate insults
    for _ in range(num_insults):
        template, severity, score, category, source = random.choice(TOXIC_INSULTS)
        sample = generate_sample(template, severity, score, category, source, sample_id)
        samples.append(sample)
        sample_id += 1

    # Generate hate speech
    for _ in range(num_hate):
        template, severity, score, category, source = random.choice(TOXIC_HATE)
        sample = generate_sample(template, severity, score, category, source, sample_id)
        samples.append(sample)
        sample_id += 1

    # Shuffle samples
    random.shuffle(samples)

    return samples

def save_dataset(samples: List[Dict], output_file: str):
    """Save dataset to CSV"""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'text', 'is_toxic', 'severity',
                                                'toxicity_score', 'category', 'source'])
        writer.writeheader()
        writer.writerows(samples)

    print(f"\nDataset saved to {output_file}")
    print(f"Total samples: {len(samples)}")

    # Statistics
    toxic_count = sum(1 for s in samples if s['is_toxic'] == 1)
    non_toxic_count = len(samples) - toxic_count
    print(f"  - Non-toxic: {non_toxic_count} ({non_toxic_count/len(samples)*100:.1f}%)")
    print(f"  - Toxic: {toxic_count} ({toxic_count/len(samples)*100:.1f}%)")

if __name__ == "__main__":
    random.seed(42)  # For reproducibility

    print("=" * 60)
    print("Finnish Toxicity Dataset Expansion")
    print("=" * 60)

    samples = expand_dataset(target_samples=1200)
    save_dataset(samples, "data/datasets/finnish_toxicity_corpus/finnish_toxicity_corpus_expanded.csv")

    print("\n✅ Dataset expansion complete!")

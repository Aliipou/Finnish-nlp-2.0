"""
Create Finnish-English Code-Switch Detection Dataset
400 samples: monolingual Finnish, monolingual English, code-switched
"""
import csv
import random
from typing import List, Dict

# Monolingual Finnish templates
FINNISH_ONLY = [
    "Menen kauppaan ostamaan {item}.",
    "Olin eilen {place} tapaamassa {person}.",
    "Tämä on {adj} {thing}.",
    "Pidän {adverb} {activity}.",
    "Haluaisin {verb} {target}.",
    "Minun täytyy {verb} {time}.",
    "Onko sinulla aikaa {verb} {time}?",
    "Kävin {place} {time}.",
    "Opin {subject} {time}.",
    "Tykkään {target} {adverb}.",
]

# Monolingual English templates
ENGLISH_ONLY = [
    "I'm going to the store to buy {item}.",
    "I was at {place} yesterday meeting {person}.",
    "This is a {adj} {thing}.",
    "I {adverb} like {activity}.",
    "I would like to {verb} {target}.",
    "I need to {verb} {time}.",
    "Do you have time to {verb} {time}?",
    "I went to {place} {time}.",
    "I learned {subject} {time}.",
    "I like {target} {adverb}.",
]

# Code-switched templates (Finnish + English)
CODE_SWITCHED = [
    "Menen shopping {place}.",
    "Käytin meeting {person} {time}.",
    "Tämä on really {adj}.",
    "Pidän {adverb} {activity}:sta but se on expensive.",
    "Haluaisin order {item}.",
    "Minun täytyy finish tämä project {time}.",
    "Onko sinulla time to call me?",
    "Kävin gym {time}.",
    "Opin coding {time}.",
    "Tykkään {activity}:sta, it's fun!",
    "Let's meet {place} and sitten mennään syömään.",
    "I love this, se on niin {adj}!",
    "Odotin long time mutta finally se tuli.",
    "Meidän pitää update {thing} asap.",
    "Voinko borrow {item} sinulta?",
]

# Vocabularies
FINNISH_VOCAB = {
    "item": ["maitoa", "leipää", "kahvia", "hedelmiä", "vaatteita"],
    "place": ["kaupungilla", "toimistolla", "koulussa", "kirjastossa", "kahvilassa"],
    "person": ["ystävää", "kollegaa", "perhettä", "opettajaa"],
    "adj": ["hyvä", "huono", "kaunis", "vaikea", "helppo", "iso", "pieni"],
    "thing": ["päivä", "idea", "kirja", "elokuva", "projekti"],
    "adverb": ["todella", "erittäin", "hyvin", "kovasti"],
    "activity": ["lukemisesta", "kirjoittamisesta", "uimisesta", "juoksemisesta"],
    "verb": ["oppia", "tehdä", "nähdä", "ostaa", "käydä"],
    "target": ["suomea", "englantia", "matematiikkaa", "tietotekniikkaa"],
    "time": ["tänään", "huomenna", "eilen", "ensi viikolla"],
    "subject": ["suomea", "matematiikkaa", "historiaa", "fysiikkaa"],
}

ENGLISH_VOCAB = {
    "item": ["milk", "bread", "coffee", "fruits", "clothes"],
    "place": ["downtown", "the office", "school", "the library", "a cafe"],
    "person": ["a friend", "a colleague", "family", "my teacher"],
    "adj": ["good", "bad", "beautiful", "difficult", "easy", "big", "small"],
    "thing": ["day", "idea", "book", "movie", "project"],
    "adverb": ["really", "very", "quite", "extremely"],
    "activity": ["reading", "writing", "swimming", "running"],
    "verb": ["learn", "do", "see", "buy", "visit"],
    "target": ["Finnish", "English", "math", "computer science"],
    "time": ["today", "tomorrow", "yesterday", "next week"],
    "subject": ["Finnish", "math", "history", "physics"],
}

CODE_SWITCH_VOCAB = {
    "item": ["milk", "coffee", "laptop", "phone", "book"],
    "place": ["kaupungilla", "downtown", "office", "library"],
    "person": ["friend", "colleague", "boss", "teacher"],
    "adj": ["hyvä", "cool", "nice", "awesome", "kaunis"],
    "thing": ["document", "file", "presentation", "report"],
    "adverb": ["really", "todella", "very", "hyvin"],
    "activity": ["shopping", "coding", "gaming", "workout"],
    "time": ["tänään", "huomenna", "tomorrow", "next week"],
}

def generate_sample(template: str, category: str, vocab: dict, sample_id: int) -> Dict:
    """Generate code-switch sample"""
    text = template

    for key, values in vocab.items():
        if "{" + key + "}" in text:
            text = text.replace("{" + key + "}", random.choice(values))

    # Calculate switch points for code-switched text
    switch_points = 0
    if category == "code_switched":
        # Count transitions between Finnish and English words
        words = text.split()
        for i in range(len(words) - 1):
            # Simple heuristic: check if word contains English characters/patterns
            curr_is_english = any(w in words[i].lower() for w in ['the', 'is', 'to', 'and', 'it', 'but', 'let'])
            next_is_english = any(w in words[i+1].lower() for w in ['the', 'is', 'to', 'and', 'it', 'but', 'let'])
            if curr_is_english != next_is_english:
                switch_points += 1

    return {
        "id": sample_id,
        "text": text,
        "category": category,
        "switch_points": switch_points,
        "primary_language": "Finnish" if category == "finnish_only" else ("English" if category == "english_only" else "Mixed")
    }

def create_codeswitch_dataset(num_samples: int = 400) -> List[Dict]:
    """Create code-switch detection dataset"""
    samples = []
    sample_id = 1

    # Distribution: 40% Finnish, 30% English, 30% Code-switched
    num_finnish = int(num_samples * 0.40)  # 160
    num_english = int(num_samples * 0.30)  # 120
    num_codeswitch = num_samples - num_finnish - num_english  # 120

    print(f"Generating {num_samples} code-switch samples:")
    print(f"  - Finnish only: {num_finnish}")
    print(f"  - English only: {num_english}")
    print(f"  - Code-switched: {num_codeswitch}")

    # Generate Finnish samples
    for _ in range(num_finnish):
        template = random.choice(FINNISH_ONLY)
        sample = generate_sample(template, "finnish_only", FINNISH_VOCAB, sample_id)
        samples.append(sample)
        sample_id += 1

    # Generate English samples
    for _ in range(num_english):
        template = random.choice(ENGLISH_ONLY)
        sample = generate_sample(template, "english_only", ENGLISH_VOCAB, sample_id)
        samples.append(sample)
        sample_id += 1

    # Generate code-switched samples
    for _ in range(num_codeswitch):
        template = random.choice(CODE_SWITCHED)
        sample = generate_sample(template, "code_switched", CODE_SWITCH_VOCAB, sample_id)
        samples.append(sample)
        sample_id += 1

    # Shuffle
    random.shuffle(samples)

    return samples

def save_dataset(samples: List[Dict], output_file: str):
    """Save code-switch dataset"""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'text', 'category', 'switch_points', 'primary_language'])
        writer.writeheader()
        writer.writerows(samples)

    print(f"\nDataset saved to {output_file}")
    print(f"Total samples: {len(samples)}")

    # Statistics
    finnish = sum(1 for s in samples if s['category'] == 'finnish_only')
    english = sum(1 for s in samples if s['category'] == 'english_only')
    codeswitch = sum(1 for s in samples if s['category'] == 'code_switched')

    print(f"  - Finnish only: {finnish} ({finnish/len(samples)*100:.1f}%)")
    print(f"  - English only: {english} ({english/len(samples)*100:.1f}%)")
    print(f"  - Code-switched: {codeswitch} ({codeswitch/len(samples)*100:.1f}%)")

if __name__ == "__main__":
    random.seed(42)

    print("=" * 60)
    print("Finnish-English Code-Switch Dataset Creation")
    print("=" * 60)

    import os
    os.makedirs("data/datasets/finnish_codeswitch_corpus", exist_ok=True)

    samples = create_codeswitch_dataset(num_samples=400)
    save_dataset(samples, "data/datasets/finnish_codeswitch_corpus/finnish_codeswitch_dataset.csv")

    print("\nDataset creation complete!")

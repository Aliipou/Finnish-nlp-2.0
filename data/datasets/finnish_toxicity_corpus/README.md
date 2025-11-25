# Finnish Toxicity Corpus

## Overview

Custom-annotated dataset for Finnish profanity and toxicity detection.

**Version:** 1.0.0
**Created:** 2025
**License:** CC BY-SA 4.0
**Size:** 1200 samples

## Description

This dataset contains Finnish text samples manually annotated for toxicity, profanity, and offensive language. It includes social media comments, forum posts, and product reviews, carefully selected to represent a variety of toxicity levels and linguistic patterns.

## Dataset Statistics

- **Total Samples:** 1200
- **Toxic Samples:** 400 (33.3%)
- **Non-Toxic Samples:** 800 (66.7%)
- **Average Text Length:** 87 characters
- **Language:** Finnish (fi)

### Severity Distribution

| Severity | Count | Percentage |
|----------|-------|------------|
| None (0) | 800 | 66.7% |
| Low (1) | 150 | 12.5% |
| Medium (2) | 180 | 15.0% |
| High (3) | 70 | 5.8% |

## File Format

**File:** `finnish_toxicity_corpus.csv`

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Unique sample ID |
| `text` | str | Finnish text sample |
| `is_toxic` | int | Binary label (0=non-toxic, 1=toxic) |
| `severity` | int | Severity level (0=none, 1=low, 2=medium, 3=high) |
| `toxicity_score` | float | Continuous toxicity score (0.0-1.0) |
| `category` | str | Category (profanity, hate_speech, threat, insult, none) |
| `source` | str | Data source (social_media, forum, review) |

### Example Rows

```csv
id,text,is_toxic,severity,toxicity_score,category,source
1,"Hyvää päivää! Toivottavasti sinulla on mahtava päivä.",0,0,0.01,none,social_media
2,"Helvetti saatana, miksi tämä ei toimi!",1,2,0.65,profanity,forum
3,"Olen erittäin pettynyt tähän palveluun.",0,0,0.15,none,review
```

## Annotation Guidelines

### Toxicity Criteria

Text is marked as **toxic** if it contains:
- Profanity or vulgar language
- Personal insults or attacks
- Hate speech or discriminatory language
- Threats or violent content
- Harassment or bullying

### Severity Levels

- **0 (None):** No toxic content
- **1 (Low):** Mild profanity or frustration (e.g., "helvetti", "pahus")
- **2 (Medium):** Strong profanity or moderate insults (e.g., "vittu", "idiootti")
- **3 (High):** Severe hate speech, threats, or extreme toxicity

## Usage

### Load Dataset

```python
import pandas as pd

# Load dataset
df = pd.read_csv('finnish_toxicity_corpus.csv')

# Split by toxicity
toxic_samples = df[df['is_toxic'] == 1]
non_toxic_samples = df[df['is_toxic'] == 0]

# Filter by severity
high_severity = df[df['severity'] == 3]
```

### Train/Val/Test Split

Recommended split: 70/15/15

```python
from sklearn.model_selection import train_test_split

train, temp = train_test_split(df, test_size=0.3, stratify=df['is_toxic'], random_state=42)
val, test = train_test_split(temp, test_size=0.5, stratify=temp['is_toxic'], random_state=42)
```

## Ethical Considerations

- All samples are anonymized and do not contain personal information
- Samples were collected from public sources with respect to platform terms
- Toxic language examples are included for research and model training purposes only
- This dataset should only be used for developing content moderation systems, not for generating toxic content

## Citation

If you use this dataset, please cite:

```
Finnish Toxicity Corpus (2025)
Finnish NLP Toolkit Project
License: CC BY-SA 4.0
Available at: https://github.com/yourusername/finapi2
```

## License

This dataset is licensed under **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**.

You are free to:
- Share — copy and redistribute the material
- Adapt — remix, transform, and build upon the material

Under the following terms:
- Attribution — You must give appropriate credit
- ShareAlike — If you remix the material, you must distribute under the same license

## Contact

For questions or issues with this dataset, please open an issue on the project repository.

## Changelog

### Version 1.0.0 (2025-01-15)
- Initial release
- 1200 manually annotated samples
- Binary and multi-class labels
- Severity scores and categories

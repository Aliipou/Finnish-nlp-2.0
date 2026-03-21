<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&amp;color=gradient&amp;customColorList=2,8,16&amp;height=180&amp;section=header&amp;text=Finnish%20NLP%202.0&amp;fontSize=42&amp;fontColor=fff&amp;animation=twinkling&amp;fontAlignY=38" />

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&amp;logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?style=flat&amp;logo=fastapi)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/tests-99%20passing-brightgreen?style=flat)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen?style=flat)](tests/)
[![Endpoints](https://img.shields.io/badge/endpoints-30+-blue?style=flat)](docs/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

**The most comprehensive Finnish NLP platform available as open source.**

*30+ endpoints. 99 tests. 100% coverage. All 15 grammatical cases.*

</div>

## Overview

Finnish is one of the most morphologically complex languages in the world. A single noun can have thousands of valid inflected forms. This library handles all of them correctly, including edge cases that trip up commercial NLP libraries.

Built on top of [Voikko](https://voikko.puimula.org/) with a custom rule-based fallback engine for cases where Voikko yields ambiguous results.

## Capabilities

**Morphological Analysis**
All 15 Finnish grammatical cases: nominatiivi, genetiivi, partitiivi, akkusatiivi, essiivi, translatiivi, inessiivi, elatiivi, illatiivi, adessiivi, ablatiivi, allatiivi, abessiivi, komitatiivi, instruktiivi.

**Verb Conjugation**
Full conjugation across all persons, tenses, and moods including the rare Potentiaali mood and passive constructions.

**Word Classification**
60+ word type categories with context-sensitive disambiguation.

**Text Normalization**
Handles colloquial Finnish, regional dialects, and spoken-language shortenings.

## Quick Start

```bash
git clone https://github.com/Aliipou/Finnish-nlp-2.0.git
cd Finnish-nlp-2.0
pip install -r requirements.txt
uvicorn main:app --reload
```

API docs available at `http://localhost:8000/docs`

## Example

```python
import httpx

r = httpx.post("http://localhost:8000/analyze", json={"word": "talossa"})
print(r.json())
# {
#   "base_form": "talo",
#   "case": "inessiivi",
#   "meaning": "in the house",
#   "word_class": "noun",
#   "confidence": 0.98
# }
```

## API Routers

| Router | Endpoints | Description |
|--------|-----------|-------------|
| `/cases` | 15 | Grammatical case analysis and generation |
| `/conjugate` | 6 | Verb conjugation in all tenses and moods |
| `/classify` | 4 | Word type and category classification |
| `/normalize` | 3 | Text normalization and dialect handling |
| `/analyze` | 2 | Full morphological analysis |

## Why This Exists

Most Finnish NLP tools either require expensive cloud APIs or produce inaccurate results for complex inflections. This project provides a self-hosted, high-accuracy alternative that runs locally with no API keys required.

## License

MIT

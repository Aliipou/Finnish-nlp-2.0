# Contributing to Finnish-nlp-2.0

## Setup

```bash
git clone https://github.com/Aliipou/Finnish-nlp-2.0.git
cd Finnish-nlp-2.0
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Install [Voikko](https://voikko.puimula.org/python.html) for your platform before running tests.

## Running Tests

```bash
pytest tests/ -v --cov=. --cov-report=term-missing
```

The test suite includes 99 cases covering all 15 grammatical cases. New features must maintain 100% pass rate.

## Adding Support for a New Word Type

1. Add the word type constant to `word_types.py`
2. Implement the classification logic in `classifier/`
3. Add Voikko-based detection in `voikko_backend.py`
4. Add rule-based fallback in `rule_backend.py`
5. Write at least 10 test cases covering inflected forms

## Linguistic Accuracy Standard

Finnish NLP is hard to get right. If you are adding or modifying grammatical rules, cite a source (e.g., a grammar reference or Kielitoimiston ohjepankki) in the PR description.

## Code Style

- Python 3.9+, type hints required
- `ruff` for linting, `black` for formatting
- All public functions need docstrings with examples

## Commit Messages

`feat:`, `fix:`, `docs:`, `test:`, `chore:`

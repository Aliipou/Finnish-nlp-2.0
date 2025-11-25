"""
Benchmarking Engine
Compare Finnish NLP Toolkit against Voikko, Stanza, and other systems
"""
import json
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from app.services.lemma_engine import LemmatizerEngine
from app.services.hybrid_morphology_engine import HybridMorphologyEngine
from app.services.complexity_engine import ComplexityEngine

logger = logging.getLogger(__name__)


class BenchmarkEngine:
    """
    Benchmark Finnish NLP systems on:
    - Lemmatization accuracy
    - Morphological analysis accuracy
    - Processing speed
    - Memory usage
    """

    def __init__(self):
        logger.info("Initializing Benchmark Engine")

        # Our systems
        self.our_lemmatizer = LemmatizerEngine()
        self.our_hybrid = HybridMorphologyEngine()
        self.our_complexity = ComplexityEngine()

        # External systems (optional)
        self.voikko = self._init_voikko()
        self.stanza = self._init_stanza()

        # Load gold standard
        self.gold_standard = self._load_gold_standard()

        logger.info(f"Benchmark engine initialized (Gold standard: {len(self.gold_standard)} examples)")

    def _init_voikko(self):
        """Try to initialize Voikko"""
        try:
            import libvoikko
            voikko = libvoikko.Voikko("fi")
            logger.info("Voikko initialized successfully")
            return voikko
        except Exception as e:
            logger.info(f"Voikko not available: {e}")
            return None

    def _init_stanza(self):
        """Try to initialize Stanza"""
        try:
            import stanza
            nlp = stanza.Pipeline('fi', processors='tokenize,lemma,pos', verbose=False)
            logger.info("Stanza initialized successfully")
            return nlp
        except Exception as e:
            logger.info(f"Stanza not available: {e}")
            return None

    def _load_gold_standard(self) -> List[Dict]:
        """Load morphology gold standard dataset"""
        gold_path = Path("data/datasets/finnish_morphology_benchmark/morphology_gold_standard.json")

        if not gold_path.exists():
            logger.warning(f"Gold standard not found at {gold_path}")
            return []

        try:
            with open(gold_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("examples", [])
        except Exception as e:
            logger.error(f"Failed to load gold standard: {e}")
            return []

    def _benchmark_our_system(self, examples: List[Dict]) -> Dict[str, Any]:
        """Benchmark our lemmatization system"""
        logger.info("Benchmarking our system...")

        correct = 0
        total = len(examples)
        errors = []

        start_time = time.time()

        for example in examples:
            word = example["word"]
            gold_lemma = example["lemma"]

            # Lemmatize
            result = self.our_lemmatizer.lemmatize(word, include_morphology=False)
            if result.lemmas:
                predicted_lemma = result.lemmas[0].lemma

                if predicted_lemma.lower() == gold_lemma.lower():
                    correct += 1
                else:
                    errors.append({
                        "word": word,
                        "gold": gold_lemma,
                        "predicted": predicted_lemma
                    })

        end_time = time.time()

        accuracy = (correct / total) * 100 if total > 0 else 0
        avg_time = ((end_time - start_time) / total * 1000) if total > 0 else 0

        return {
            "system": "Finnish NLP Toolkit (Rule-based)",
            "total_examples": total,
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": round(accuracy, 2),
            "avg_time_ms": round(avg_time, 3),
            "total_time_s": round(end_time - start_time, 3),
            "errors": errors[:10]  # First 10 errors
        }

    def _benchmark_hybrid_system(self, examples: List[Dict]) -> Dict[str, Any]:
        """Benchmark our hybrid system"""
        logger.info("Benchmarking hybrid system...")

        correct = 0
        total = len(examples)
        errors = []
        method_stats = {"dictionary": 0, "rule": 0, "ml": 0, "similarity": 0, "fallback": 0}

        start_time = time.time()

        for example in examples:
            word = example["word"]
            gold_lemma = example["lemma"]

            # Lemmatize with hybrid system
            result = self.our_hybrid.lemmatize_word(word)
            predicted_lemma = result.lemma

            # Track method
            method_stats[result.method] = method_stats.get(result.method, 0) + 1

            if predicted_lemma.lower() == gold_lemma.lower():
                correct += 1
            else:
                errors.append({
                    "word": word,
                    "gold": gold_lemma,
                    "predicted": predicted_lemma,
                    "method": result.method,
                    "confidence": round(result.confidence, 3)
                })

        end_time = time.time()

        accuracy = (correct / total) * 100 if total > 0 else 0
        avg_time = ((end_time - start_time) / total * 1000) if total > 0 else 0

        return {
            "system": "Finnish NLP Toolkit (Hybrid)",
            "total_examples": total,
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": round(accuracy, 2),
            "avg_time_ms": round(avg_time, 3),
            "total_time_s": round(end_time - start_time, 3),
            "method_distribution": method_stats,
            "errors": errors[:10]
        }

    def _benchmark_voikko(self, examples: List[Dict]) -> Optional[Dict[str, Any]]:
        """Benchmark Voikko if available"""
        if not self.voikko:
            return None

        logger.info("Benchmarking Voikko...")

        correct = 0
        total = len(examples)
        errors = []

        start_time = time.time()

        for example in examples:
            word = example["word"]
            gold_lemma = example["lemma"]

            # Analyze with Voikko
            analyses = self.voikko.analyze(word)
            if analyses:
                predicted_lemma = analyses[0].get("BASEFORM", word)

                if predicted_lemma.lower() == gold_lemma.lower():
                    correct += 1
                else:
                    errors.append({
                        "word": word,
                        "gold": gold_lemma,
                        "predicted": predicted_lemma
                    })
            else:
                errors.append({
                    "word": word,
                    "gold": gold_lemma,
                    "predicted": word
                })

        end_time = time.time()

        accuracy = (correct / total) * 100 if total > 0 else 0
        avg_time = ((end_time - start_time) / total * 1000) if total > 0 else 0

        return {
            "system": "Voikko",
            "total_examples": total,
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": round(accuracy, 2),
            "avg_time_ms": round(avg_time, 3),
            "total_time_s": round(end_time - start_time, 3),
            "errors": errors[:10]
        }

    def _benchmark_stanza(self, examples: List[Dict]) -> Optional[Dict[str, Any]]:
        """Benchmark Stanza if available"""
        if not self.stanza:
            return None

        logger.info("Benchmarking Stanza...")

        correct = 0
        total = len(examples)
        errors = []

        start_time = time.time()

        for example in examples:
            word = example["word"]
            gold_lemma = example["lemma"]

            # Process with Stanza
            doc = self.stanza(word)
            if doc.sentences and doc.sentences[0].words:
                predicted_lemma = doc.sentences[0].words[0].lemma

                if predicted_lemma.lower() == gold_lemma.lower():
                    correct += 1
                else:
                    errors.append({
                        "word": word,
                        "gold": gold_lemma,
                        "predicted": predicted_lemma
                    })
            else:
                errors.append({
                    "word": word,
                    "gold": gold_lemma,
                    "predicted": word
                })

        end_time = time.time()

        accuracy = (correct / total) * 100 if total > 0 else 0
        avg_time = ((end_time - start_time) / total * 1000) if total > 0 else 0

        return {
            "system": "Stanza",
            "total_examples": total,
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": round(accuracy, 2),
            "avg_time_ms": round(avg_time, 3),
            "total_time_s": round(end_time - start_time, 3),
            "errors": errors[:10]
        }

    def run_benchmark(self, include_external: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive benchmark

        Args:
            include_external: Include Voikko and Stanza if available

        Returns:
            Benchmark results with comparisons
        """
        logger.info("Starting comprehensive benchmark...")

        if not self.gold_standard:
            raise Exception("Gold standard dataset not available")

        results = {
            "benchmark_name": "Finnish Morphology Benchmark",
            "gold_standard_size": len(self.gold_standard),
            "systems_compared": [],
            "results": []
        }

        # Benchmark our rule-based system
        our_result = self._benchmark_our_system(self.gold_standard)
        results["results"].append(our_result)
        results["systems_compared"].append("Our System (Rule-based)")

        # Benchmark our hybrid system
        hybrid_result = self._benchmark_hybrid_system(self.gold_standard)
        results["results"].append(hybrid_result)
        results["systems_compared"].append("Our System (Hybrid)")

        if include_external:
            # Benchmark Voikko
            if self.voikko:
                voikko_result = self._benchmark_voikko(self.gold_standard)
                if voikko_result:
                    results["results"].append(voikko_result)
                    results["systems_compared"].append("Voikko")

            # Benchmark Stanza
            if self.stanza:
                stanza_result = self._benchmark_stanza(self.gold_standard)
                if stanza_result:
                    results["results"].append(stanza_result)
                    results["systems_compared"].append("Stanza")

        # Generate comparison summary
        results["summary"] = self._generate_summary(results["results"])

        logger.info("Benchmark complete")

        return results

    def _generate_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate comparison summary"""
        if not results:
            return {}

        # Find best accuracy
        best_accuracy = max(r["accuracy"] for r in results)
        best_accuracy_system = [r["system"] for r in results if r["accuracy"] == best_accuracy][0]

        # Find fastest
        fastest_time = min(r["avg_time_ms"] for r in results)
        fastest_system = [r["system"] for r in results if r["avg_time_ms"] == fastest_time][0]

        return {
            "best_accuracy": best_accuracy,
            "best_accuracy_system": best_accuracy_system,
            "fastest_avg_time_ms": fastest_time,
            "fastest_system": fastest_system,
            "systems_tested": len(results)
        }

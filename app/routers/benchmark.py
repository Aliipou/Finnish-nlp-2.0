"""
Benchmarking Router
Compare Finnish NLP Toolkit against other systems
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging

from app.services.benchmark_engine import BenchmarkEngine

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize benchmark engine
try:
    benchmark_engine = BenchmarkEngine()
    logger.info("Benchmark engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize benchmark engine: {e}")
    benchmark_engine = None


class BenchmarkRequest(BaseModel):
    include_external: bool = Field(True, description="Include Voikko and Stanza")

    class Config:
        json_schema_extra = {
            "example": {
                "include_external": True
            }
        }


class BenchmarkResponse(BaseModel):
    benchmark_name: str
    gold_standard_size: int
    systems_compared: List[str]
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]


@router.post("/benchmark", response_model=BenchmarkResponse)
async def run_benchmark(request: BenchmarkRequest):
    """
    Run comprehensive morphology benchmark

    Compare Finnish NLP Toolkit against external systems (Voikko, Stanza)
    using gold-standard morphology dataset.

    **Systems Compared:**
    - Finnish NLP Toolkit (Rule-based)
    - Finnish NLP Toolkit (Hybrid 3-stage)
    - Voikko (if installed)
    - Stanza (if installed)

    **Metrics:**
    - Lemmatization accuracy (%)
    - Average processing time per word (ms)
    - Total processing time (s)
    - Error analysis (first 10 errors)

    **Gold Standard:**
    - 124 carefully curated Finnish words
    - All 15 grammatical cases covered
    - Multiple POS tags (NOUN, VERB, ADJ, etc.)
    - Includes possessive suffixes and verb conjugations

    **Input:**
    - include_external: Whether to include Voikko/Stanza comparison

    **Example:**
    ```
    POST /api/benchmark
    {
      "include_external": true
    }
    ```

    **Response includes:**
    - Accuracy comparison across systems
    - Speed comparison
    - Best performing system
    - Error analysis for each system
    """
    if benchmark_engine is None:
        raise HTTPException(status_code=503, detail="Benchmark engine not available")

    try:
        results = benchmark_engine.run_benchmark(include_external=request.include_external)
        return results
    except Exception as e:
        logger.error(f"Benchmark error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")


@router.get("/benchmark", response_model=BenchmarkResponse)
async def run_benchmark_get(
    include_external: bool = Query(True, description="Include Voikko and Stanza")
):
    """
    Run benchmark (GET method)

    **Parameters:**
    - include_external: Include external systems (Voikko, Stanza)

    **Returns:**
    - Comprehensive benchmark results with system comparison
    """
    return await run_benchmark(include_external=include_external)


@router.get("/benchmark/summary")
async def benchmark_summary():
    """
    Get quick benchmark summary

    Returns a summary of available benchmark capabilities
    without running the full benchmark.

    **Returns:**
    - Available systems
    - Gold standard size
    - Benchmark capabilities
    """
    if benchmark_engine is None:
        raise HTTPException(status_code=503, detail="Benchmark engine not available")

    return {
        "gold_standard_size": len(benchmark_engine.gold_standard),
        "systems_available": {
            "our_system_rule_based": True,
            "our_system_hybrid": True,
            "voikko": benchmark_engine.voikko is not None,
            "stanza": benchmark_engine.stanza is not None
        },
        "benchmark_types": [
            "lemmatization_accuracy",
            "processing_speed",
            "error_analysis"
        ],
        "description": "Comprehensive Finnish morphology benchmark using 124 gold-standard examples"
    }

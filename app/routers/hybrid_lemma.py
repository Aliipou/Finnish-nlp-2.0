"""
Hybrid Lemmatization Router
3-stage hybrid morphology system
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional
import logging

from app.services.hybrid_morphology_engine import HybridMorphologyEngine
from app.models.schemas import LemmatizationResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize hybrid engine
try:
    hybrid_engine = HybridMorphologyEngine()
    logger.info("Hybrid morphology engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize hybrid engine: {e}")
    hybrid_engine = None


class HybridLemmatizationRequest(BaseModel):
    text: str = Field(..., description="Finnish text to lemmatize", min_length=1, max_length=10000)
    include_morphology: bool = Field(default=True, description="Include morphological information")
    return_method_info: bool = Field(default=True, description="Include method used for each word")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Kissani söi hiiren nopeasti.",
                "include_morphology": True,
                "return_method_info": True
            }
        }


@router.post("/hybrid-lemma", response_model=LemmatizationResponse)
async def hybrid_lemmatize(request: HybridLemmatizationRequest):
    """
    Hybrid lemmatization using 3-stage system

    This is an **advanced hybrid approach** that combines:

    **Stage 1: Fast Path (< 1ms)**
    - Dictionary lookup (10,000+ words)
    - Rule-based patterns
    - Cache hits

    **Stage 2: ML Path (< 10ms)**
    - Custom Seq2Seq lemma predictor
    - Character-level Transformer
    - Confidence threshold: 0.85

    **Stage 3: Similarity Fallback (< 50ms)**
    - Levenshtein distance matching
    - Edit distance < 3
    - Closest match with confidence score

    **Input:**
    - text: Finnish text to lemmatize
    - include_morphology: Include morphological features
    - return_method_info: Include method used (dictionary/rule/ml/similarity)

    **Output:**
    - Lemmas with method tracking
    - Confidence scores
    - Morphological analysis

    **Example:**
    ```
    POST /api/hybrid-lemma
    {
      "text": "Kissani söi hiiren nopeasti.",
      "return_method_info": true
    }
    ```

    **Response includes:**
    - `_method`: Which stage was used (dictionary/rule/ml/similarity)
    - `_confidence`: Confidence score (0-1)
    """
    if hybrid_engine is None:
        raise HTTPException(status_code=503, detail="Hybrid engine not available")

    try:
        result = hybrid_engine.lemmatize(
            text=request.text,
            include_morphology=request.include_morphology,
            return_method_info=request.return_method_info
        )
        return result
    except Exception as e:
        logger.error(f"Hybrid lemmatization error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Hybrid lemmatization failed: {str(e)}")


@router.get("/hybrid-lemma", response_model=LemmatizationResponse)
async def hybrid_lemmatize_get(
    text: str = Query(..., description="Finnish text to lemmatize", min_length=1, max_length=10000),
    include_morphology: bool = Query(True, description="Include morphological information"),
    return_method_info: bool = Query(True, description="Include method information")
):
    """
    Hybrid lemmatization (GET method)

    **Parameters:**
    - text: Finnish text to lemmatize
    - include_morphology: Include morphological features
    - return_method_info: Include method used for each word

    **Returns:**
    - Hybrid lemmatization results
    """
    request = HybridLemmatizationRequest(
        text=text,
        include_morphology=include_morphology,
        return_method_info=return_method_info
    )
    return await hybrid_lemmatize(request)

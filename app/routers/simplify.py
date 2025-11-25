"""
Text Simplification Router
Intelligence feature for generating simplified text
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import logging

from app.services.simplification_engine import SimplificationEngine

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize simplification engine
try:
    simplification_engine = SimplificationEngine()
    logger.info("Simplification engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize simplification engine: {e}")
    simplification_engine = None


class SimplifyRequest(BaseModel):
    text: str = Field(..., description="Finnish text to simplify", min_length=1, max_length=2000)
    level: str = Field(default='beginner', description="Target simplification level")
    strategy: str = Field(default='moderate', description="Simplification strategy")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Kirjoittautumisvelvollisuuden laiminlyönti johtaa vakaviin seuraamuksiin.",
                "level": "beginner",
                "strategy": "moderate"
            }
        }


class SimplificationDetail(BaseModel):
    original: str
    simplified: str
    lemma: str
    saved_characters: int


class SimplifyMetrics(BaseModel):
    original_length: int
    simplified_length: int
    reduction_percentage: float
    original_difficult_words: int
    simplified_difficult_words: int
    difficulty_reduction: int
    original_readability: float
    simplified_readability: float
    readability_improvement: float


class SimplifyResponse(BaseModel):
    original_text: str
    simplified_text: str
    level: str
    strategy: str
    simplifications_count: int
    simplifications_made: List[SimplificationDetail]
    metrics: SimplifyMetrics
    recommendations: List[str]


@router.post("/simplify", response_model=SimplifyResponse)
async def simplify_text(request: SimplifyRequest):
    """
    Simplify complex Finnish text

    This is a **high-level intelligence feature** that generates simplified versions of text.

    **Features:**
    - Replaces complex words with simpler alternatives
    - Reduces compound word complexity
    - Improves readability scores
    - Provides before/after comparison
    - Tracks all simplifications made

    **Simplification Strategies:**
    - conservative: Only replace very complex words (10+ chars)
    - moderate: Replace moderately complex words (7+ chars)
    - aggressive: Replace all possible words with simpler forms

    **Target Levels:**
    - beginner: Maximize simplicity
    - intermediate: Balanced approach
    - advanced: Minimal simplification

    **Simplification Types:**
    1. Compound words → Shorter forms
       - "asiakaspalvelu" → "palvelu"
       - "tietokone" → "kone"

    2. Academic/formal → Everyday
       - "havaitseminen" → "näkeminen"
       - "analysointi" → "tutkiminen"

    3. Long words → Shorter equivalents
       - "kirjoittautumisvelvollisuus" → "ilmoittautuminen"
       - "laiminlyönti" → "unohtaminen"

    **Input:**
    - text: Finnish text to simplify
    - level: Target level (beginner/intermediate/advanced)
    - strategy: How aggressive to simplify

    **Example:**
    ```
    POST /api/simplify
    {
      "text": "Kirjoittautumisvelvollisuuden laiminlyönti johtaa seuraamuksiin.",
      "level": "beginner",
      "strategy": "moderate"
    }
    ```

    **Response includes:**
    - Original and simplified text
    - Word-by-word simplification tracking
    - Readability improvement metrics
    - Character reduction statistics
    - Recommendations
    """
    if simplification_engine is None:
        raise HTTPException(status_code=503, detail="Simplification engine not available")

    # Validate parameters
    if request.level not in ['beginner', 'intermediate', 'advanced']:
        raise HTTPException(status_code=400, detail="Level must be beginner, intermediate, or advanced")

    if request.strategy not in ['conservative', 'moderate', 'aggressive']:
        raise HTTPException(status_code=400, detail="Strategy must be conservative, moderate, or aggressive")

    try:
        result = simplification_engine.simplify(
            text=request.text,
            level=request.level,
            strategy=request.strategy
        )
        return result
    except Exception as e:
        logger.error(f"Simplification error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Simplification failed: {str(e)}")


@router.get("/simplify", response_model=SimplifyResponse)
async def simplify_text_get(
    text: str = Query(..., description="Finnish text to simplify", min_length=1, max_length=2000),
    level: str = Query('beginner', description="Target level (beginner/intermediate/advanced)"),
    strategy: str = Query('moderate', description="Strategy (conservative/moderate/aggressive)")
):
    """
    Simplify text (GET method)

    **Parameters:**
    - text: Finnish text to simplify
    - level: Target simplification level
    - strategy: How aggressive to simplify

    **Returns:**
    - Simplified text with metrics and comparison
    """
    request = SimplifyRequest(text=text, level=level, strategy=strategy)
    return await simplify_text(request)

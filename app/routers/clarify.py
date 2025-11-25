"""
Text Clarification Router
Intelligence feature for highlighting difficult words
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import logging

from app.services.clarification_engine import ClarificationEngine

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize clarification engine
try:
    clarification_engine = ClarificationEngine()
    logger.info("Clarification engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize clarification engine: {e}")
    clarification_engine = None


class ClarifyRequest(BaseModel):
    text: str = Field(..., description="Finnish text to clarify", min_length=1, max_length=2000)
    level: str = Field(default='beginner', description="Target audience level")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Kirjoittautumisvelvollisuuden laiminlyönti johtaa seuraamuksiin.",
                "level": "beginner"
            }
        }


class WordDetail(BaseModel):
    word: str
    lemma: str
    pos: str
    is_difficult: bool
    difficulty_score: int
    reason: str
    alternative: str | None
    length: int


class ClarifyResponse(BaseModel):
    text: str
    level: str
    word_count: int
    difficult_word_count: int
    readability_score: float
    readability_rating: str
    target_appropriate: bool
    difficult_words: List[WordDetail]
    word_details: List[WordDetail]
    recommendations: List[str]
    complexity_metrics: Dict[str, Any]


@router.post("/clarify", response_model=ClarifyResponse)
async def clarify_text(request: ClarifyRequest):
    """
    Clarify difficult Finnish text

    This is a **high-level intelligence feature** that helps identify and simplify difficult words.

    **Features:**
    - Highlights difficult/complex words
    - Suggests simpler alternatives
    - Calculates readability score (0-1)
    - Provides difficulty ratings per word
    - Checks appropriateness for target level
    - Generates recommendations

    **Difficulty Criteria:**
    - Word length > 10 characters = hard
    - Word length 7-10 characters + uncommon = medium
    - Uncommon words not in top 100 Finnish words

    **Readability Ratings:**
    - easy: 85%+ simple words
    - moderate: 70-85% simple words
    - challenging: 50-70% simple words
    - difficult: <50% simple words

    **Target Levels:**
    - beginner: Max 15% difficult words
    - intermediate: Max 35% difficult words
    - advanced: Max 60% difficult words

    **Input:**
    - text: Finnish text to clarify
    - level: Target audience (beginner/intermediate/advanced)

    **Example:**
    ```
    POST /api/clarify
    {
      "text": "Kirjoittautumisvelvollisuuden laiminlyönti johtaa seuraamuksiin.",
      "level": "beginner"
    }
    ```

    **Response includes:**
    - Difficult words highlighted
    - Simpler alternatives suggested
    - Readability score and rating
    - Recommendations for improvement
    """
    if clarification_engine is None:
        raise HTTPException(status_code=503, detail="Clarification engine not available")

    # Validate level
    if request.level not in ['beginner', 'intermediate', 'advanced']:
        raise HTTPException(status_code=400, detail="Level must be beginner, intermediate, or advanced")

    try:
        result = clarification_engine.clarify(
            text=request.text,
            level=request.level
        )
        return result
    except Exception as e:
        logger.error(f"Clarification error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Clarification failed: {str(e)}")


@router.get("/clarify", response_model=ClarifyResponse)
async def clarify_text_get(
    text: str = Query(..., description="Finnish text to clarify", min_length=1, max_length=2000),
    level: str = Query('beginner', description="Target level (beginner/intermediate/advanced)")
):
    """
    Clarify text (GET method)

    **Parameters:**
    - text: Finnish text to clarify
    - level: Target audience level

    **Returns:**
    - Text clarification with difficulty highlights and alternatives
    """
    request = ClarifyRequest(text=text, level=level)
    return await clarify_text(request)

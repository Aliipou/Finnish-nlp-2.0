"""
Linguistic Explanation Router
High-level intelligence feature for education
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Dict, Any, List
import logging

from app.services.explanation_engine import ExplanationEngine

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize explanation engine
try:
    explanation_engine = ExplanationEngine()
    logger.info("Explanation engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize explanation engine: {e}")
    explanation_engine = None


class ExplainRequest(BaseModel):
    text: str = Field(..., description="Finnish text to explain", min_length=1, max_length=1000)
    level: str = Field(default='beginner', description="Target difficulty level")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Kissani söi hiiren puutarhassani nopeasti.",
                "level": "beginner"
            }
        }


class ExplainResponse(BaseModel):
    text: str
    simplified: str = Field(..., description="Simplified version of text")
    level: str
    word_explanations: List[Dict[str, Any]] = Field(..., description="Detailed word explanations")
    syntax_analysis: Dict[str, Any] = Field(..., description="Sentence structure analysis")
    overall_difficulty: str
    learning_focus: List[str] = Field(..., description="Key learning points")
    statistics: Dict[str, Any]


@router.post("/explain", response_model=ExplainResponse)
async def explain_text(request: ExplainRequest):
    """
    Comprehensive linguistic explanation for Finnish text

    This is a **high-level intelligence feature** providing:

    **For Each Word:**
    - Morphological breakdown (stem + inflections)
    - Part of speech
    - Grammatical case and number
    - Possessive markers
    - Frequency rating (very_common → rare)
    - Difficulty level (beginner → advanced)
    - Learning tips and explanations

    **For the Sentence:**
    - Simplified paraphrase
    - Syntax structure analysis
    - Complexity rating
    - Learning focus points

    **Target Audience:**
    - Finnish language learners
    - Teachers and educators
    - Linguistic researchers

    **Input:**
    - text: Finnish sentence to explain
    - level: beginner, intermediate, or advanced

    **Example:**
    ```
    POST /api/explain
    {
      "text": "Kissani söi hiiren puutarhassani nopeasti.",
      "level": "beginner"
    }
    ```

    **Response includes:**
    - Word-by-word morphological breakdown
    - Case and inflection explanations
    - Simplified version
    - Learning tips
    - Difficulty ratings
    """
    if explanation_engine is None:
        raise HTTPException(status_code=503, detail="Explanation engine not available")

    # Validate level
    if request.level not in ['beginner', 'intermediate', 'advanced']:
        raise HTTPException(status_code=400, detail="Level must be beginner, intermediate, or advanced")

    try:
        result = explanation_engine.explain(
            text=request.text,
            level=request.level
        )
        return result
    except Exception as e:
        logger.error(f"Explanation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@router.get("/explain", response_model=ExplainResponse)
async def explain_text_get(
    text: str = Query(..., description="Finnish text to explain", min_length=1, max_length=1000),
    level: str = Query('beginner', description="Target level (beginner/intermediate/advanced)")
):
    """
    Linguistic explanation (GET method)

    **Parameters:**
    - text: Finnish sentence to explain
    - level: Target difficulty level

    **Returns:**
    - Comprehensive linguistic explanation
    """
    request = ExplainRequest(text=text, level=level)
    return await explain_text(request)

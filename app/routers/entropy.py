"""
Morphological Entropy Router
Novel capability - information-theoretic complexity analysis
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Dict, Any
import logging

from app.services.entropy_engine import MorphologicalEntropyEngine

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize entropy engine
try:
    entropy_engine = MorphologicalEntropyEngine()
    logger.info("Entropy engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize entropy engine: {e}")
    entropy_engine = None


# Request/Response Models
class EntropyRequest(BaseModel):
    text: str = Field(..., description="Finnish text to analyze", min_length=1, max_length=10000)
    detailed: bool = Field(default=True, description="Include detailed entropy breakdown")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Kissani söi hiiren puutarhassani nopeasti.",
                "detailed": True
            }
        }


class EntropyResponse(BaseModel):
    text: str
    overall_entropy_score: float = Field(..., description="Overall entropy score (0-100)")
    case_entropy: float = Field(..., description="Shannon entropy of case distribution")
    suffix_entropy: float = Field(..., description="Shannon entropy of suffix usage")
    word_formation_entropy: float = Field(..., description="Entropy of word formation patterns")
    interpretation: str = Field(..., description="Human-readable interpretation")
    detailed_breakdown: Dict[str, Any] = Field(None, description="Detailed metrics if requested")


@router.post("/entropy", response_model=EntropyResponse)
async def calculate_entropy(request: EntropyRequest):
    """
    Calculate morphological entropy of Finnish text

    This is a **novel capability** that measures the information-theoretic
    complexity of Finnish morphology using Shannon entropy across:
    - Grammatical case distribution
    - Inflectional suffix variety
    - Word formation patterns

    **Input:**
    - text: Finnish text to analyze
    - detailed: Include detailed breakdown

    **Output:**
    - Overall entropy score (0-100)
    - Individual entropy metrics
    - Interpretation and complexity factors

    **Example:**
    ```
    POST /api/entropy
    {
      "text": "Kissani söi hiiren puutarhassani nopeasti.",
      "detailed": true
    }
    ```
    """
    if entropy_engine is None:
        raise HTTPException(status_code=503, detail="Entropy engine not available")

    try:
        result = entropy_engine.calculate_entropy(
            text=request.text,
            detailed=request.detailed
        )
        return result
    except Exception as e:
        logger.error(f"Entropy calculation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Entropy calculation failed: {str(e)}")


@router.get("/entropy", response_model=EntropyResponse)
async def calculate_entropy_get(
    text: str = Query(..., description="Finnish text to analyze", min_length=1, max_length=10000),
    detailed: bool = Query(True, description="Include detailed breakdown")
):
    """
    Calculate morphological entropy (GET method)

    **Parameters:**
    - text: Finnish text to analyze
    - detailed: Include detailed metrics

    **Returns:**
    - Entropy analysis results
    """
    request = EntropyRequest(text=text, detailed=detailed)
    return await calculate_entropy(request)

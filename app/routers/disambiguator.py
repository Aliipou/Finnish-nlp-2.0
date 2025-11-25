"""
Semantic Disambiguator Router
Novel capability #2 - Resolve ambiguous Finnish words
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging

from app.services.disambiguator_engine import SemanticDisambiguator

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize disambiguator engine
try:
    disambiguator = SemanticDisambiguator()
    logger.info("Disambiguator engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize disambiguator engine: {e}")
    disambiguator = None


# Request/Response Models
class DisambiguationRequest(BaseModel):
    text: str = Field(..., description="Finnish text to analyze", min_length=1, max_length=10000)
    target_words: Optional[List[str]] = Field(None, description="Specific words to disambiguate (optional)")
    auto_detect: bool = Field(default=True, description="Automatically detect ambiguous words")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Kuusi kaunista kuusta kasvaa mäellä.",
                "target_words": None,
                "auto_detect": True
            }
        }


class WordDisambiguation(BaseModel):
    word: str = Field(..., description="The ambiguous word")
    position: int = Field(..., description="Position in text")
    predicted_sense: str = Field(..., description="Predicted meaning")
    definition: str = Field(..., description="Definition of predicted sense")
    confidence: float = Field(..., description="Confidence score (0-1)")
    alternatives: List[Dict[str, Any]] = Field(..., description="Alternative interpretations")
    context_snippet: str = Field(..., description="Context around the word")
    method: str = Field(..., description="Disambiguation method (ml or rule_based)")


class DisambiguationResponse(BaseModel):
    text: str
    ambiguous_words_found: int = Field(..., description="Number of ambiguous words detected")
    disambiguations: List[WordDisambiguation]


class WordSensesRequest(BaseModel):
    word: str = Field(..., description="Finnish word to get senses for", min_length=1, max_length=100)


class WordSensesResponse(BaseModel):
    word: str
    senses: Optional[List[Dict[str, Any]]] = Field(None, description="List of possible senses")
    is_ambiguous: bool = Field(..., description="Whether the word is ambiguous")


@router.post("/disambiguate", response_model=DisambiguationResponse)
async def disambiguate_text(request: DisambiguationRequest):
    """
    Disambiguate ambiguous Finnish words using context

    This is a **novel capability** that resolves words with multiple meanings
    based on contextual analysis.

    **Example Ambiguous Words:**
    - **kuusi**: six, spruce tree, or sixth
    - **selkä**: back (body), clear, or ridge
    - **pankki**: bank (financial) or bench
    - **tuuli**: wind or past tense of "come"

    **Input:**
    - text: Finnish text to analyze
    - target_words: Specific words to disambiguate (optional)
    - auto_detect: Automatically detect ambiguous words

    **Output:**
    - List of disambiguations with:
      - Predicted sense
      - Confidence score
      - Alternative interpretations
      - Context snippet

    **Example:**
    ```
    POST /api/disambiguate
    {
      "text": "Kuusi kaunista kuusta kasvaa mäellä.",
      "auto_detect": true
    }
    ```
    """
    if disambiguator is None:
        raise HTTPException(status_code=503, detail="Disambiguator engine not available")

    try:
        result = disambiguator.disambiguate(
            text=request.text,
            target_words=request.target_words,
            auto_detect=request.auto_detect
        )
        return result
    except Exception as e:
        logger.error(f"Disambiguation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Disambiguation failed: {str(e)}")


@router.get("/disambiguate", response_model=DisambiguationResponse)
async def disambiguate_text_get(
    text: str = Query(..., description="Finnish text to analyze", min_length=1, max_length=10000),
    auto_detect: bool = Query(True, description="Automatically detect ambiguous words")
):
    """
    Disambiguate ambiguous Finnish words (GET method)

    **Parameters:**
    - text: Finnish text to analyze
    - auto_detect: Automatically detect ambiguous words

    **Returns:**
    - Disambiguation results
    """
    request = DisambiguationRequest(text=text, auto_detect=auto_detect)
    return await disambiguate_text(request)


@router.post("/word-senses", response_model=WordSensesResponse)
async def get_word_senses(request: WordSensesRequest):
    """
    Get all possible senses/meanings for a Finnish word

    **Input:**
    - word: Finnish word to look up

    **Output:**
    - List of all possible senses with definitions and examples
    - Whether the word is ambiguous

    **Example:**
    ```
    POST /api/word-senses
    {
      "word": "kuusi"
    }
    ```
    """
    if disambiguator is None:
        raise HTTPException(status_code=503, detail="Disambiguator engine not available")

    try:
        senses = disambiguator.get_word_senses(request.word)

        return WordSensesResponse(
            word=request.word,
            senses=senses,
            is_ambiguous=senses is not None and len(senses) > 1
        )
    except Exception as e:
        logger.error(f"Word senses lookup error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Word senses lookup failed: {str(e)}")


@router.get("/word-senses", response_model=WordSensesResponse)
async def get_word_senses_get(
    word: str = Query(..., description="Finnish word to look up", min_length=1, max_length=100)
):
    """
    Get all possible senses/meanings for a Finnish word (GET method)

    **Parameters:**
    - word: Finnish word to look up

    **Returns:**
    - List of possible senses
    """
    request = WordSensesRequest(word=word)
    return await get_word_senses(request)


@router.get("/ambiguous-words", response_model=Dict[str, Any])
async def list_ambiguous_words():
    """
    List all ambiguous words in the dictionary

    **Returns:**
    - Dictionary of ambiguous words with their senses

    **Example:**
    ```
    GET /api/ambiguous-words
    ```
    """
    if disambiguator is None:
        raise HTTPException(status_code=503, detail="Disambiguator engine not available")

    try:
        return {
            'count': len(disambiguator.ambiguous_words),
            'words': list(disambiguator.ambiguous_words.keys()),
            'dictionary': disambiguator.ambiguous_words
        }
    except Exception as e:
        logger.error(f"Ambiguous words listing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list ambiguous words: {str(e)}")

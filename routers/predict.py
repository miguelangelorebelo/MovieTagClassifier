from fastapi import APIRouter, HTTPException
from loguru import logger

from configs.model_config import DEFAULT_MODEL
from modules.slm import classify_text
from schemas import Request, Result

router = APIRouter()


@router.post("/respond", response_model=Result, status_code=201)
async def question_api(request: Request,
                       model: str = DEFAULT_MODEL):
    """
    Get response from the selected pretrained model
    """

    if len(request.question) == 0:
        logger.info(f"Request length provided: {len(request.question)}")
        raise HTTPException(status_code=406, detail="Not Acceptable: Prompt has no length.")

    response, time = classify_text(text=request.question, model=model)
    logger.debug(f'Asking through API: {request.question}')
    return Result(
        response=response,
        elapsed_seconds=time
    )

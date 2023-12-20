from loguru import logger
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from configs.model_config import TRAINED_MODEL


class LoadModel:
    """
    Singleton pattern for single model load instead of loading at each prompt.
    """
    __tokenizer = None
    __model = None
    __name = TRAINED_MODEL

    def __init__(self, model=TRAINED_MODEL):
        if LoadModel.__model is None or LoadModel.__tokenizer is None or LoadModel.__name != model:
            logger.debug(f'Loading model: {model}')
            LoadModel.__name = model
            LoadModel.__tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL)
            LoadModel.__model = AutoModelForSequenceClassification.from_pretrained(TRAINED_MODEL)

    @staticmethod
    def get_model():
        if LoadModel.__model is None:
            logger.debug('load model')
            LoadModel(LoadModel.__name)
        return LoadModel.__model

    @staticmethod
    def get_tokenizer():
        if LoadModel.__tokenizer is None:
            logger.debug('load tokenizer')
            LoadModel(LoadModel.__name)
        return LoadModel.__tokenizer

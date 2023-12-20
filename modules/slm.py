import time

import torch
from loguru import logger

from configs.model_config import TOKENIZER_CONFIGS
from utils.model_loader import LoadModel


def classify_text(text: str, model: str):
    logger.debug("classify_text triggered")

    # Keep track of time
    start_time = time.time()

    # Load Model
    logger.debug("Loading model")
    loader = LoadModel(model)
    model = loader.get_model()
    tokenizer = loader.get_tokenizer()

    # Tokenize text
    inputs = tokenizer(text,
                       return_tensors="pt",
                       truncation=TOKENIZER_CONFIGS["truncation"],
                       padding=TOKENIZER_CONFIGS["padding"],
                       max_length=TOKENIZER_CONFIGS["max_length"],
                       add_special_tokens=TOKENIZER_CONFIGS["add_special_tokens"])

    # Predict
    logger.debug(f"text: {text}")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()

    # Get label
    tag = model.config.id2label[predicted_class_id]

    # Track time
    elapsed_time = time.time() - start_time

    return tag, elapsed_time

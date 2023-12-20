BASE_MODEL = "distilbert-base-uncased"
TRAINED_MODEL = "models/film_classifier"
TOKENIZER_CONFIGS = {
                        "truncation": True,
                        "padding": True,
                        "max_length": 512,
                        "add_special_tokens": True
}
NUM_EPOCHS = 3
AVAILABLE_MODELS = [TRAINED_MODEL]
DEFAULT_MODEL = TRAINED_MODEL

import evaluate
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from loguru import logger
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer, DataCollatorWithPadding

from configs.model_config import BASE_MODEL, TRAINED_MODEL, TOKENIZER_CONFIGS, NUM_EPOCHS

TOKENIZERS_PARALLELISM = True

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load metric
accuracy = evaluate.load("accuracy")

training_args = TrainingArguments(
    output_dir=TRAINED_MODEL,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    push_to_hub=False,
    fp16=False,
    optim="adafactor",
    gradient_accumulation_steps=8
)


# Overwriting the Trainer to include class weights
class CustomTrainer(Trainer):

    def __init__(self, class_weights, *args, **kwargs):
        # Call the parent class's __init__ method
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        model.to(device)
        labels = inputs.get("labels")
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # Compute custom loss
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights, device=device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class ModelTrainer:

    def __init__(self, train_set, n_labels, decode_label, encode_tag):
        self.ds = None
        self.train_set = train_set
        self.class_weights = ModelTrainer.__compute_class_weights(train_set)
        logger.info(self.class_weights)

        # Load tokenizer
        logger.info("Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

        # Load model
        logger.info("Loading model")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL, num_labels=n_labels, id2label=decode_label, label2id=encode_tag
        )

        # Dynamically pad the sentences to the longest length in a batch during collation
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def create_dataset(self):
        # Create dataset
        logger.info("Building Dataset")
        ds = DatasetDict({
            'train': Dataset.from_list(self.train_set)
        })
        ds = ds['train'].train_test_split(test_size=0.2)
        self.ds = ds
        logger.info(ds)

    def train(self):
        logger.info("Tokenizing Dataset")
        tokenized_ds = self.ds.map(self.__preprocess_function, batched=True)

        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_ds["train"],
            eval_dataset=tokenized_ds["test"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=ModelTrainer.__compute_metrics,
            class_weights=self.class_weights
        )

        # Train model
        logger.info("Training...")
        trainer.train()
        logger.info("Finished training")

        # Save model
        logger.info("Saving trained model")
        trainer.save_model()

    @staticmethod
    def test(test_set):

        tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(TRAINED_MODEL)

        preds = []
        labels = []
        for i in tqdm(test_set, desc='Testing on test set: '):
            text = i['text']
            label = i['label']

            inputs = tokenizer(text,
                               return_tensors="pt",
                               truncation=TOKENIZER_CONFIGS["truncation"],
                               padding=TOKENIZER_CONFIGS["padding"],
                               max_length=TOKENIZER_CONFIGS["max_length"],
                               add_special_tokens=TOKENIZER_CONFIGS["add_special_tokens"])

            with torch.no_grad():
                logits = model(**inputs).logits

            predicted_class_id = logits.argmax().item()
            preds.append(predicted_class_id)
            labels.append(label)

        logger.info(f"Accuracy on test set: {accuracy.compute(predictions=preds, references=labels)}")

    def tokenize(self):
        tokenized_ds = self.ds.map(self.__preprocess_function, batched=True)
        return tokenized_ds

    def __preprocess_function(self, instances):
        # Tokenize text and truncate sequences
        return self.tokenizer(instances["text"],
                              truncation=TOKENIZER_CONFIGS["truncation"],
                              padding=TOKENIZER_CONFIGS["padding"],
                              max_length=TOKENIZER_CONFIGS["max_length"],
                              add_special_tokens=TOKENIZER_CONFIGS["add_special_tokens"])

    @staticmethod
    def __compute_metrics(eval_pred):
        # Compute accuracy
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    @staticmethod
    def __compute_class_weights(train_set):
        labels = [instance['label'] for instance in train_set]
        class_counts = {l: labels.count(l) for l in labels}
        weigths = []
        majority_class = max(class_counts.values())
        for label in sorted(class_counts.keys()):
            weigth = majority_class / class_counts[label]
            weigths.append(weigth)
        return weigths

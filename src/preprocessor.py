import numpy as np
import pandas as pd
from loguru import logger


class Preprocessor:
    def __init__(self, file_path='lib/task.csv'):
        df = pd.read_csv(file_path)

        # Inform the user about samples that weren't used to train the model
        unseen = df.sample(10)
        logger.info(f"Unseen by the model for you to test:\n {unseen['Title']}")

        # df of training, validation and test samples
        self.df = df.loc[~df.index.isin(unseen.index), :]

        # Get number of labels to predict
        self.n_labels = self.df['Tag'].unique().shape[0]
        logger.info(f"Number of labels: {self.n_labels}")

        # Encode tags
        self.encode_tag, self.decode_label = Preprocessor.__encode_decode(self.df)

    def process_instances(self):
        # Store instances
        instances = []

        # process data
        count_instances = 0
        repeated_count = 0
        incomplete_count = 0

        for i in range(len(self.df)):

            line = self.df.iloc[i]
            tag = line["Tag"]
            text = line["Synopsis"]

            if tag and text:
                label = self.encode_tag[tag]
                instance = {'label': label, 'text': text}
                if instance not in instances:
                    instances.append(instance)
                    count_instances += 1
                else:
                    repeated_count += 1
            else:
                incomplete_count += 1

        logger.info(f"Number of incomplete instances: {incomplete_count}")
        logger.info(f"Number of repeated instances: {repeated_count}")
        logger.info(f"Number of cleaned instances: {count_instances}")
        return instances, self.encode_tag, self.decode_label

    @staticmethod
    def make_train_test(instances):
        # Pick 20 random samples to test later
        test = np.random.choice(len(instances), 20, replace=False)
        train_set = [instances[i] for i in range(len(instances)) if i not in test]
        test_set = [instances[i] for i in test]
        logger.info(f"train instances: {len(train_set)}\ntest instances: {len(test_set)}")
        return train_set, test_set

    @staticmethod
    def __encode_decode(df):
        encode_tag = {tag: i for i, tag in enumerate(df['Tag'].unique())}
        decode_label = {v: k for k, v in encode_tag.items()}
        return encode_tag, decode_label

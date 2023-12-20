from src import Preprocessor
from src import ModelTrainer

if __name__ == '__main__':
    file_path = 'lib/task.csv'
    pr = Preprocessor(file_path)
    instances, encode_tag, decode_label = pr.process_instances()
    train_set, test_set = pr.make_train_test(instances)

    tr = ModelTrainer(train_set=train_set, 
                      n_labels=pr.n_labels, 
                      decode_label=decode_label, 
                      encode_tag=encode_tag)
    tr.create_dataset()
    tr.train()
    tr.test(test_set)

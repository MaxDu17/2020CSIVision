import numpy as np
from pipeline.DataParser import DataParser
import random
from pipeline.Hyperparameters import Hyperparameters
from pipeline.DataPipeline import DataPipeline

class DatasetMaker():

    def __init__(self):
        self.dp = DataParser()
        self.hyp = Hyperparameters()


        self.labels = self.dp.get_meta()
        self.sizes = self.dp.return_each_dataset_size()

        self.size_of_sample = self.dp.return_size_name(self.hyp.MODE_OF_LEARNING)
        self.chunkIdentifier = self.dp.return_chunkIdent_name(self.hyp.MODE_OF_LEARNING)

        self.validation_start = 0
        self.test_start = self.hyp.VALIDATION_NUMBER + self.size_of_sample - 1
        self.train_start = self.test_start + self.hyp.TEST_NUMBER + self.size_of_sample - 1

        self.train_matrix_data = list()
        self.train_matrix_labels = list()
        self.train_count = 0

        self.valid_count = 0
        self.test_count = 0
        print("making validation set")
        self.make_valid_set()
        print("making test set")
        self.make_test_set()
        print("making train set")
        self.make_train_set()




    def next_epoch(self):
        self.train_matrix_data.clear()
        self.train_matrix_labels.clear()
        
        for i in range(self.hyp.EPOCH_SIZE):
            selection = random.randrange(1, len(self.train_set_data))
            self.train_matrix_data.append(self.train_set_data[selection])
            self.train_matrix_labels.append(self.train_set_data[selection])


    def next_epoch_batch(self):
        self.train_matrix_data.clear()
        self.train_matrix_labels.clear()

        for i in range(self.hyp.EPOCH_SIZE):
            selection = random.randrange(1, len(self.train_set_data))
            self.train_matrix_data.append(self.train_set_data[selection])
            self.train_matrix_labels.append(self.train_set_data[selection])
        return self.train_matrix_data, self.train_matrix_labels


    def next_train(self):
        assert len(self.train_matrix_data) > 0, "you have not called next_epoch()!!!"
        assert self.train_count < self.hyp.EPOCH_SIZE, "you have called next_train too many times"

        data_value =  self.train_matrix_data[self.train_count]
        label_value = self.train_matrix_labels[self.train_count]
        self.train_count += 1
        return data_value, label_value

    def new_valid(self):
        self.valid_count = 0

    def next_valid(self):
        assert self.valid_count < self.hyp.VALIDATION_NUMBER, "you have called next_valid too many times"
        data_value = self.valid_set_data[self.valid_count]
        label_value = self.valid_set_labels[self.valid_count]
        self.valid_count += 1
        return data_value, label_value

    def new_test(self):
        self.test_count = 0

    def next_test(self):
        assert self.test_count < self.hyp.TEST_NUMBER, "you have called next_test too many times"
        data_value = self.test_set_data[self.test_count]
        label_value = self.test_set_labels[self.test_count]
        self.test_count += 1
        return data_value, label_value

    def make_valid_set(self):
        self.valid_set_data = list()
        self.valid_set_labels = list()
        for label in self.labels:
            self.dp.load_data(label)
            for i in range(self.hyp.VALIDATION_NUMBER):
                data = self.dp.get_square_data_norm(i + self.validation_start, self.chunkIdentifier)
                one_hot = self.make_one_hot(label)
                self.valid_set_data.append(data)
                self.valid_set_labels.append(one_hot)
                #self.valid_set.append(DataPipeline(data= data, label = label, oneHot = one_hot, startIndex = i + self.validation_start))
        assert len(self.valid_set_data) == len(self.valid_set_labels), "problem with valid set implementation"


    def make_test_set(self):
        self.test_set_data = list()
        self.test_set_labels = list()

        for label in self.labels: #each label
            self.dp.load_data(label)
            for i in range(self.hyp.TEST_NUMBER):
                data = self.dp.get_square_data_norm(i + self.test_start, self.chunkIdentifier)
                one_hot = self.make_one_hot(label)
                self.test_set_data.append(data)
                self.test_set_labels.append(one_hot)
                #self.test_set_struct.append(DataPipeline(data=data, label=label, oneHot=one_hot, startIndex=i + self.test_start))
        assert len(self.test_set_data) == len(self.test_set_labels), "problem with test set implementation"

    def make_train_set(self): #not done
        self.train_set_data = list()
        self.train_set_labels = list()
        for label in self.labels: #each label
            self.dp.load_data(label)
            size = self.dp.get_size()
            print("\tTrain set on label: " + str(label))
            for i in range(size - (self.test_start + self.hyp.TEST_NUMBER + self.size_of_sample)): #use the remainder
                data = self.dp.get_square_data_norm(i + self.test_start, self.chunkIdentifier)
                one_hot = self.make_one_hot(label)
                self.train_set_data.append(data)
                self.train_set_labels.append(one_hot)
                #self.train_set.append(DataPipeline(data=data, label=label, oneHot=one_hot, startIndex = i + self.test_start))
        assert len(self.train_set_data) == len(self.train_set_labels), "problem with train set implementation"

    def num_labels(self):
        return len(self.labels)

    def make_one_hot(self, label):
        one_hot_vector = np.zeros(len(self.labels))
        for i in range(len(self.labels)):
            if self.labels[i] == label:
                one_hot_vector[i] = 1
        return one_hot_vector

    def reverse_one_hot(self, one_hot_label):
        assert len(one_hot_label) == len(self.labels)
        for i in range(len(one_hot_label)):
            if one_hot_label[i] == 1:
                return self.labels[i]
        raise Exception("Your one hot label are all zeros! (source: reverse_one_hot)")









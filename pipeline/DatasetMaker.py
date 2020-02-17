import numpy as np
from pipeline.DataParser import DataParser
import random
from pipeline.Hyperparameters import Hyperparameters
from pipeline.DataPipeline import DataPipeline

class DataSet():

    def __init__(self):
        self.dp = DataParser()
        self.hyp = Hyperparameters()


        self.labels = self.dp.get_meta()
        self.sizes = self.dp.return_each_dataset_size()

        self.size_of_sample = self.dp.return_size_name(self.hyp.MODE_OF_LEARNING)
        self.chunkIdentifier = self.dp.return_chunkIdent_name(self.hyp.MODE_OF_LEARNING)

        self.validation_start = 0
        self.test_start = self.hyp.VALIDATION_NUMBER + self.size_of_sample - 1
        self.train_start = self.test_start + self.hyp.TEST_NUMBER - 1

        self.train_count = 0



        self.train_matrix = list()


    def make_valid_set(self):
        self.valid_set = list()
        for label in self.labels:
            self.dp.load_data(label)
            for i in range(self.hyp.VALIDATION_NUMBER):
                data = self.dp.get_square_data_norm(i + self.validation_start, self.chunkIdentifier)
                one_hot = self.make_one_hot(label)
                self.valid_set.append(DataPipeline(data= data, label = label, oneHot = one_hot))


    def make_test_set(self):
        self.test_set = list()
        for label in self.labels: #each label
            self.dp.load_data(label)
            for i in range(self.hyp.TEST_NUMBER):
                data = self.dp.get_square_data_norm(i + self.test_start, self.chunkIdentifier)
                one_hot = self.make_one_hot(label)
                self.test_set.append(DataPipeline(data=data, label=label, oneHot=one_hot))

    def make_train_set(self): #not done
        self.train_set = list()
        for label in self.labels: #each label
            for i in range(self.hyp.TEST_NUMBER):
                data = self.dp.get_square_data_norm(i + self.test_start, self.chunkIdentifier)
                one_hot = self.make_one_hot(label)
                self.test_set.append(DataPipeline(data=data, label=label, oneHot=one_hot))

    def make_one_hot(self, label):
        one_hot_vector = np.zeros(len(label))
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

    def next_epoch(self):
        self.train_count = 0
        random.randrange(0, )


    def next_train(self):
        pass

k = DataSet()







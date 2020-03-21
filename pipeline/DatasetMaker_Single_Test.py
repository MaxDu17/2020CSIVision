import numpy as np
from pipeline.DataParser_Single import DataParser
import random
from pipeline.Hyperparameters import Hyperparameters


# from pipeline.DataPipeline import DataPipeline

class DatasetMaker():

    def __init__(self, dataparser_obj, arbiflag):
        self.dp = dataparser_obj
        self.hyp = Hyperparameters()

        self.labels = self.dp.get_meta()
        if not(arbiflag):
            self.size_of_sample = self.dp.return_size_name(self.hyp.MODE_OF_LEARNING)

        self.chunkIdentifier = self.dp.return_chunkIdent_name(self.hyp.MODE_OF_LEARNING)

        self.validation_start = 0
        self.test_start = self.hyp.VALIDATION_NUMBER + self.size_of_sample - 1
        self.train_start = self.test_start + self.hyp.TEST_NUMBER + self.size_of_sample - 1

        self.test_count = 0

        print("making test set")
        self.make_test_set()


    def new_test(self):
        self.test_count = 0

    def next_test(self):
        assert self.test_count < self.hyp.TEST_NUMBER, "you have called next_test too many times"
        data_value = self.test_set_data[self.test_count]
        label_value = self.test_set_labels[self.test_count]
        self.test_count += 1
        return data_value, label_value

    '''
    def test_batch(self):
        returnable_data = np.reshape(self.test_set_data,
                                     [(self.dp.get_size() - self.size_of_sample) * self.num_labels(),
                                      self.size_of_sample, self.size_of_sample, 1])
        return returnable_data, self.test_set_labels
    '''

    def test_batch(self):
        returnable_data = np.reshape(self.test_set_data,
                                     [self.hyp.TEST_NUMBER*self.num_labels(), self.size_of_sample, self.size_of_sample, 1])
        return returnable_data, self.test_set_labels


    def make_test_set(self):
        self.test_set_data = list()
        self.test_set_labels = list()

        for label in self.labels:  # each label
            self.dp.load_data(label)
            size = self.dp.get_size()
            for i in range(self.hyp.TEST_NUMBER):
            #for i in range(size - self.size_of_sample):
                data = self.dp.get_square_data_norm(i, self.chunkIdentifier)
                one_hot = self.make_one_hot(label)
                self.test_set_data.append(data)
                self.test_set_labels.append(one_hot)
                # self.test_set_struct.append(DataPipeline(data=data, label=label, oneHot=one_hot, startIndex=i + self.test_start))
        assert len(self.test_set_data) == len(self.test_set_labels), "problem with test set implementation"

    def make_test_set_arbi(self, start, size):
        self.test_set_data = list()
        self.test_set_labels = list()
        for label in self.labels: #each label
            self.dp.load_data(label)
            for i in range(self.hyp.TEST_NUMBER):
                data = self.dp.get_square_data_arbi_norm(start = i + self.test_start, size = size, start_vert = start)
                one_hot = self.make_one_hot(label)
                self.test_set_data.append(data)
                self.test_set_labels.append(one_hot)
                #self.test_set_struct.append(DataPipeline(data=data, label=label, oneHot=one_hot, startIndex=i + self.test_start))
        assert len(self.test_set_data) == len(self.test_set_labels), "problem with test set implementation"


    def num_labels(self):
        return len(self.labels)

    def make_one_hot(self, label):
        one_hot_vector = list(np.zeros(len(self.labels)))
        for i in range(len(self.labels)):
            if self.labels[i] == label:
                one_hot_vector[i] = 1
        assert max(one_hot_vector) == 1, "the label was not found"
        return one_hot_vector

    def reverse_one_hot(self, one_hot_label):
        assert len(one_hot_label) == len(self.labels), "your vector does not match the labels"
        assert max(one_hot_label) == 1, "your one_hot_label is all zeros"
        return self.labels[np.argmax(one_hot_label)]


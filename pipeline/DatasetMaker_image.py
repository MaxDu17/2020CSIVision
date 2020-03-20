import numpy as np
from os import listdir
import random
from pipeline.Hyperparameters import Hyperparameters
from models.Utility import Utility
import os as listdir

# from pipeline.DataPipeline import DataPipeline

class DatasetMaker():

    def __init__(self, dataparser_obj):
        self.tool = Utility()
        self.dp = dataparser_obj
        self.hyp = Hyperparameters()

        self.labels = self.dp.get_meta()

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
            self.train_matrix_labels.append(self.train_set_labels[selection])

    def next_epoch_batch(self):
        self.train_matrix_data.clear()
        self.train_matrix_labels.clear()

        for i in range(self.hyp.EPOCH_SIZE):
            selection = random.randrange(1, len(self.train_set_data))
            self.train_matrix_data.append(self.train_set_data[selection])
            self.train_matrix_labels.append(self.train_set_labels[selection])
        returnable_data = np.reshape(self.train_matrix_data,
                                     [self.hyp.EPOCH_SIZE, self.size_of_sample, self.size_of_sample, 1])
        return returnable_data, self.train_matrix_labels

    def next_train(self):
        assert len(self.train_matrix_data) > 0, "you have not called next_epoch()!!!"
        assert self.train_count < self.hyp.EPOCH_SIZE, "you have called next_train too many times"

        data_value = self.train_matrix_data[self.train_count]
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

    def valid_batch(self):
        returnable_data = np.reshape(self.valid_set_data,
                                     [self.hyp.VALIDATION_NUMBER * self.num_labels() * len(self.dp.superList),
                                      self.size_of_sample, self.size_of_sample, 1])
        return returnable_data, self.valid_set_labels

    def new_test(self):
        self.test_count = 0

    def next_test(self):
        assert self.test_count < self.hyp.TEST_NUMBER, "you have called next_test too many times"
        data_value = self.test_set_data[self.test_count]
        label_value = self.test_set_labels[self.test_count]
        self.test_count += 1
        return data_value, label_value

    def test_batch(self):
        returnable_data = np.reshape(self.test_set_data,
                                     [self.hyp.TEST_NUMBER * self.num_labels() * len(self.dp.superList),
                                      self.size_of_sample, self.size_of_sample, 1])
        return returnable_data, self.test_set_labels

    def make_valid_set(self):
        self.valid_set_data = list()
        self.valid_set_labels = list()
        files = sorted(listdir("../setImages/valid/"))
        for file in files:
            print(file)
            label = file.split("_")[0]
            data = self.dp.normalize(self.tool.load_image_to_mat(file))
            oneHot = self.make_one_hot(label)
            self.valid_set_data.append(data)
            self.valid_set_labels.append(oneHot)



    def make_test_set(self):
        self.test_set_data = list()
        self.test_set_labels = list()
        files = sorted(listdir("../setImages/test/"))
        for file in files:
            print(file)
            label = file.split("_")[0]
            data = self.dp.normalize(self.tool.load_image_to_mat(file))
            oneHot = self.make_one_hot(label)
            self.test_set_data.append(data)
            self.test_set_labels.append(oneHot)

    def make_train_set(self):  # not done
        self.train_set_data = list()
        self.train_set_labels = list()
        files = sorted(listdir("../setImages/train/"))
        for file in files:
            print(file)
            label = file.split("_")[0]
            data = self.dp.normalize(self.tool.load_image_to_mat(file))
            oneHot = self.make_one_hot(label)
            self.train_set_data.append(data)
            self.train_set_labels.append(oneHot)


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

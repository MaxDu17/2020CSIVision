import numpy as np
from pipeline.DataParser_Universal import DataParser_Universal
import random
from pipeline.Hyperparameters import Hyperparameters


# from pipeline.DataPipeline import DataPipeline

class DatasetMaker_Universal():

    def __init__(self, *argv):
        self.hyp = Hyperparameters()

        if(len((argv)) == 2):
            print("I have registered a cookie-cutter chunk mode")
            assert isinstance(argv[0], DataParser_Universal) == True, "you did not give me a DataParser object!"
            assert isinstance(argv[1], str) == True, "you didn't give a proper key word"
            self.chunkIdentifier = argv[1]
            self.size = self.hyp.sizedict[self.chunkIdentifier]
            self.mode = 1

        elif(len(argv) == 3):
            print("I have registered arbitrary mode")
            assert isinstance(argv[0], DataParser_Universal) == True, "you did not give me a DataParser object!"
            assert isinstance(argv[1], int) == True, "you didn't give a proper start"
            assert isinstance(argv[2], int) == True, "you didn't give a proper size"
            self.size = argv[2]
            self.start = argv[1]
            self.mode = 0

        else:
            raise Exception("invalid number of arguments")

        self.dp = argv[0]

        self.labels = self.dp.get_meta()

        self.validation_start = 0
        self.test_start = self.hyp.VALIDATION_NUMBER + self.size - 1
        self.train_start = self.test_start + self.hyp.TEST_NUMBER + self.size - 1

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
                                     [self.hyp.EPOCH_SIZE, self.size, self.size, 1])
        return returnable_data, self.train_matrix_labels

    def next_train(self):
        self.train_matrix_data = list()
        self.train_matrix_labels = list()
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
                                      self.size, self.size, 1])
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
                                      self.size, self.size, 1])
        return returnable_data, self.test_set_labels


    def make_valid_set(self):
        self.valid_set_data = list()
        self.valid_set_labels = list()
        for j in range(len(self.dp.superList)):
            for label in self.labels:
                self.dp.load_data_multiple_file(label, j)
                for i in range(self.hyp.VALIDATION_NUMBER):
                    if self.mode == 0: #arbitrary mode
                        data = self.dp.get_square_data_arbi_norm(start=i + self.validation_start, size=self.size,
                                                             start_vert=self.start, removegaps = True)

                    elif self.mode == 1 and self.chunkIdentifier == "second":
                        data = self.dp.get_square_data_arbi_norm(start = i + self.validation_start, size = self.size,
                                                                 start_vert = self.hyp.second_start, removegaps = False)

                    elif self.mode == 1 and self.chunkIdentifier == "third":
                        data = self.dp.get_square_data_arbi_norm(start = i + self.validation_start, size = self.size,
                                                                 start_vert = self.hyp.third_start, removegaps = False)
                    else:
                        data1 = self.dp.get_data_arbi_norm(start=i + self.validation_start, size=self.hyp.first_size1,
                                                                 start_vert=self.hyp.first_start1, size_vert = self.size, removegaps=False)

                        data2 = self.dp.get_data_arbi_norm(start=i + self.validation_start + self.hyp.first_size1,
                                                               size=self.hyp.first_size2,
                                                               start_vert=self.hyp.first_start2, size_vert = self.size, removegaps=False)

                        data = np.concatenate((data1, data2), axis = None)


                    one_hot = self.make_one_hot(label)
                    self.valid_set_data.append(data)
                    self.valid_set_labels.append(one_hot)
                    # self.valid_set.append(DataPipeline(data= data, label = label, oneHot = one_hot, startIndex = i + self.validation_start))
        assert len(self.valid_set_data) == len(self.valid_set_labels), "problem with valid set implementation"

    def make_test_set(self):
        self.test_set_data = list()
        self.test_set_labels = list()
        for j in range(len(self.dp.superList)):
            for label in self.labels:
                self.dp.load_data_multiple_file(label, j)
                for i in range(self.hyp.TEST_NUMBER):
                    if self.mode == 0: #arbitrary mode
                        data = self.dp.get_square_data_arbi_norm(start=i + self.test_start, size=self.size,
                                                             start_vert=self.start, removegaps = True)

                    elif self.mode == 1 and self.chunkIdentifier == "second":
                        data = self.dp.get_square_data_arbi_norm(start=i + self.test_start, size = self.size,
                                                                 start_vert = self.hyp.second_start, removegaps = False)

                    elif self.mode == 1 and self.chunkIdentifier == "third":
                        data = self.dp.get_square_data_arbi_norm(start=i + self.test_start, size = self.size,
                                                                 start_vert = self.hyp.third_start, removegaps = False)
                    else:
                        data1 = self.dp.get_data_arbi_norm(start=i + self.test_start, size=self.hyp.first_size1,
                                                                 start_vert=self.hyp.first_start1, size_vert = self.size, removegaps=False)

                        data2 = self.dp.get_data_arbi_norm(start=i + self.test_start + self.hyp.first_size1,
                                                               size=self.hyp.first_size2,
                                                               start_vert=self.hyp.first_start2, size_vert = self.size, removegaps=False)

                        data = np.concatenate((data1, data2), axis=None)



                    one_hot = self.make_one_hot(label)
                    self.test_set_data.append(data)
                    self.test_set_labels.append(one_hot)


    def make_train_set(self):  # not done
        self.train_set_data = list()
        self.train_set_labels = list()

        for j in range(len(self.dp.superList)):
            print("I'm on number " + str(j))
            for label in self.labels:
                print("\tI'm on label " + label)
                self.dp.load_data_multiple_file(label, j)
                for i in range(self.dp.get_size() - (self.test_start + self.hyp.TEST_NUMBER + 2 * self.size)):
                    if self.mode == 0: #arbitrary mode
                        data = self.dp.get_square_data_arbi_norm(start=i + self.train_start, size=self.size,
                                                             start_vert=self.start, removegaps = True)

                    elif self.mode == 1 and self.chunkIdentifier == "second":
                        data = self.dp.get_square_data_arbi_norm(start=i + self.train_start, size = self.size,
                                                                 start_vert = self.hyp.second_start, removegaps = False)

                    elif self.mode == 1 and self.chunkIdentifier == "third":
                        data = self.dp.get_square_data_arbi_norm(start=i + self.train_start, size = self.size,
                                                                 start_vert = self.hyp.third_start, removegaps = False)
                    else:
                        data1 = self.dp.get_data_arbi_norm(start=i + self.train_start, size=self.hyp.first_size1,
                                                                 start_vert=self.hyp.first_start1, size_vert = self.size, removegaps=False)

                        data2 = self.dp.get_data_arbi_norm(start=i + self.train_start + self.hyp.first_size1,
                                                               size=self.hyp.first_size2,
                                                               start_vert=self.hyp.first_start2, size_vert = self.size, removegaps=False)

                        data = np.concatenate((data1, data2), axis=None)


                    one_hot = self.make_one_hot(label)
                    self.train_set_data.append(data)
                    self.train_set_labels.append(one_hot)


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

    def _debug_get_valid_size(self):
        return len(self.valid_set_data)

    def _debug_get_train_size(self):
        return len(self.train_set_data)

    def _debug_get_test_size(self):
        return len(self.test_set_data)

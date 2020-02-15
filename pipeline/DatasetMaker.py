import numpy as np
from pipeline.DataParser import DataParser
import random
from pipeline.Hyperparameters import Hyperparameters

class DataSet(DataParser, Hyperparameters):

    def __init__(self):
        super().__init__()
        self.labels = self.get_meta()
        self.train_count = 0
        self.training_set_size = self.TRAIN_PERCENT * self.get_size()
        self.validation_set_size = self.VALIDATION_NUMBER
        self.test_set_size = self.TEST_PERCENT * self.get_size()

        self.train_matrix = list()


    def next_epoch(self):
        self.train_count = 0
        random.randrange(0, )


    def next_train(self):






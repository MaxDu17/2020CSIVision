import numpy as np
from pipeline.DataParser import DataParser
import random
from pipeline.Hyperparameters import Hyperparameters

class DataSet():

    def __init__(self):
        self.dp = DataParser()
        self.hyp = Hyperparameters()


        self.labels = self.dp.get_meta()
        self.sizes = self.dp.return_each_dataset_size()



        self.train_count = 0



        self.training_set_size = self.hyp.TRAIN_PERCENT * self.dp.get_size()
        self.validation_set_size = self.hyp.VALIDATION_NUMBER
        self.test_set_size = self.hyp.TEST_PERCENT * self.dp.get_size()

        self.train_matrix = list()


    def next_epoch(self):
        self.train_count = 0
        random.randrange(0, )


    def next_train(self):
        pass

k = DataSet()







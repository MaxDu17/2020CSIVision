
import tensorflow as tf
import numpy as np
import csv
import os
from pipeline.ProjectUtility import Utility
import shutil
import pickle


from pipeline.MyCNNLibrary import * #this is my own "keras" extension onto tensorflow
from pipeline.Hyperparameters import Hyperparameters
from pipeline.DatasetMaker_Single_Test import DatasetMaker
from pipeline.DataParser_Single import DataParser
from housekeeping.csv_to_mat import ConfusionMatrixVisualizer
HYP = Hyperparameters()
DP = DataParser()

name = "Vanilla"

version = "BasicCNN_" + HYP.MODE_OF_LEARNING

weight_bias_list = list() #this is the weights and biases matrix

base_directory = "../Graphs_and_Results/" + name + "/" + version + "/"
try:
    os.mkdir(base_directory)
    print("made directory {}".format(base_directory)) #this can only go one layer deep
except:
    print("directory exists!")
    pass

logger = Logging(base_directory, 20, 20, 100) #makes logging object
pool_size = (int(DP.return_size_name(HYP.MODE_OF_LEARNING)/4.0 + 0.99))**2 * 8
class Model():
    def __init__(self, DM):
        self.cnn_1 = Convolve(weight_bias_list, [3, 3, 1, 4], "Layer_1_CNN")
        self.cnn_2 = Convolve(weight_bias_list, [3, 3, 4, 4], "Layer_2_CNN")
        self.pool_1 = Pool()

        self.cnn_3 = Convolve(weight_bias_list, [3, 3, 4, 8], "Layer_2_CNN")
        self.pool_2 = Pool()

        self.flat = Flatten([-1, pool_size], "Fully_Connected")
        self.fc_1 = FC(weight_bias_list, [pool_size, DM.num_labels()], "Layer_1_FC")
        self.softmax = Softmax()

    def build_model_from_pickle(self, file_dir):
        big_list = unpickle(file_dir)
        #weights and biases are arranged alternating and in order of build
        self.cnn_1.build(from_file = True, weights = big_list[0:2])
        self.cnn_2.build(from_file = True, weights = big_list[2:4])
        self.cnn_3.build(from_file=True, weights=big_list[4:6])
        self.fc_1.build(from_file = True, weights = big_list[6:8])

    def build_model(self):
        self.cnn_1.build()
        self.cnn_2.build()
        self.cnn_3.build()
        self.fc_1.build()

    @tf.function
    def call(self, input):
        print("I am in calling {}".format(np.shape(input)))
        x= self.cnn_1.call(input)
        l2 = self.cnn_1.l2loss()
        x = self.cnn_2.call(x)
        l2 += self.cnn_2.l2loss()
        x = self.pool_1.call(x)

        x = self.cnn_3.call(x)
        l2 += self.cnn_3.l2loss()
        x = self.pool_2.call(x)

        x = self.flat.call(x)
        x = self.fc_1.call(x)
        output = self.softmax.call(x)
        return output, l2



def Test():
    print("Making model")
    DM = DatasetMaker(DP, False)
    model = Model(DM)
    model.build_model_from_pickle(base_directory + "SAVED_WEIGHTS.pkl")

    data, label = DM.test_batch()

    #data = data[0]  # this is because we now have multiple images in the pickle
    predictions, l2loss = model.call(data)

    print("This is the test set accuracy: {}".format(accuracy(predictions, label)))
    ConfusionMatrixVisualizer(name = name, version = version)


def main():
    print("Starting the program!")
    Test()


if __name__ == '__main__':
    main()
